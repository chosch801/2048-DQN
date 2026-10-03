"""FP32 evaluation with separately reported greedy and shallow search policies."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from model import AfterstateValueNet
from model_v30 import AfterstateValueNetV30
from policy import choose_actions_batch
from search import choose_expectimax_actions_batch
from vector_env import BatchEnv2048


def find_latest_checkpoint(models_dir: str) -> str | None:
    candidates: list[str] = []
    if not os.path.isdir(models_dir):
        return None
    for root, _, files in os.walk(models_dir):
        for filename in files:
            if filename.startswith("dqn_ep") and filename.endswith(".pth"):
                candidates.append(os.path.join(root, filename))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: int(os.path.basename(path).split("ep")[1].split(".pth")[0]),
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    model, version, _ = _load_model_with_config(checkpoint_path, device)
    return model, version


def _load_model_with_config(checkpoint_path: str, device: torch.device) -> tuple:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    version = checkpoint.get("version")
    if version == "v3.1-afterstate-quantile-cnn-dqn":
        model: torch.nn.Module = AfterstateValueNet(
            num_quantiles=checkpoint.get("num_quantiles", 51)
        ).to(device)
    elif version == "v3-afterstate-cnn-dqn":
        model = AfterstateValueNetV30().to(device)
    else:
        raise ValueError(f"unsupported V3 checkpoint version: {version}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, str(version), checkpoint.get("training_config", {})


@torch.inference_mode()
def run_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    episodes: int,
    gamma: float,
    seed: int | None,
    num_envs: int = 64,
    reward_mode: str = "transition",
    policy_name: str = "greedy",
) -> dict:
    """Run complete seeded games with the selected inference policy."""

    episodes = int(episodes)
    num_envs = int(num_envs)
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if policy_name not in ("greedy", "expectimax-2"):
        raise ValueError(f"unsupported policy: {policy_name}")

    base_seed = (
        int(seed)
        if seed is not None
        else int(np.random.default_rng().integers(0, 2**31 - 1))
    )
    scores: list[int] = []
    max_tiles: list[int] = []
    lengths: list[int] = []
    games: list[dict] = []
    action_counts = np.zeros(4, dtype=np.int64)
    inference_seconds = 0.0
    inference_batches = 0
    start_time = time.perf_counter()

    completed = 0
    while completed < episodes:
        chunk_size = min(num_envs, episodes - completed)
        chunk_start = completed
        env = BatchEnv2048(num_envs=chunk_size, seed=base_seed + completed)
        _, masks = env.reset()
        active = np.ones(chunk_size, dtype=bool)
        episode_lengths = np.zeros(chunk_size, dtype=np.int64)

        while np.any(active):
            active_indices = np.flatnonzero(active)
            inference_start = time.perf_counter()
            choose = (
                choose_actions_batch if policy_name == "greedy"
                else choose_expectimax_actions_batch
            )
            policy_options = {"epsilon": 0.0, "amp_enabled": False} if policy_name == "greedy" else {}
            actions = choose(
                model,
                env.boards(),
                masks,
                device,
                gamma,
                active_mask=active,
                reward_mode=reward_mode,
                **policy_options,
            )
            # Include action preparation and device transfer, not environment steps.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            inference_batches += 1
            (
                _,
                _,
                dones,
                _,
                next_masks,
                infos,
            ) = env.step(actions, active_mask=active)
            np.add.at(action_counts, actions[active_indices], 1)
            episode_lengths[active_indices] += 1

            for index in active_indices:
                if not dones[index]:
                    continue
                completed += 1
                scores.append(int(infos[index]["score"]))
                max_tiles.append(int(infos[index]["max_tile"]))
                lengths.append(int(episode_lengths[index]))
                games.append({
                    "episode_id": int(chunk_start + index),
                    "seed": int(base_seed + chunk_start + index),
                    "score": int(infos[index]["score"]),
                    "max_tile": int(infos[index]["max_tile"]),
                    "length": int(episode_lengths[index]),
                })
                active[index] = False

            masks = next_masks

        if completed % 100 == 0 or completed == episodes:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            print(
                f"已评估 {completed}/{episodes} 局 "
                f"({completed / elapsed:.1f} 局/秒)"
            )

    score_array = np.asarray(scores)
    tile_array = np.asarray(max_tiles)
    length_array = np.asarray(lengths)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    return {
        "episodes": int(episodes),
        "mean_score": float(score_array.mean()),
        "median_score": float(np.median(score_array)),
        "max_score": int(score_array.max()),
        "min_score": int(score_array.min()),
        "std_score": float(score_array.std()),
        "rate_512": float(np.mean(tile_array >= 512)),
        "rate_1024": float(np.mean(tile_array >= 1024)),
        "rate_2048": float(np.mean(tile_array >= 2048)),
        "rate_4096": float(np.mean(tile_array >= 4096)),
        "rate_8192": float(np.mean(tile_array >= 8192)),
        "max_tile": int(tile_array.max()),
        "mean_length": float(length_array.mean()),
        "max_length": int(length_array.max()),
        "action_counts": action_counts.tolist(),
        "games": sorted(games, key=lambda game: game["episode_id"]),
        "seed": base_seed,
        "gamma": float(gamma),
        "device": str(device),
        "num_envs": num_envs,
        "reward_mode": reward_mode,
        "policy": policy_name,
        "precision": "float32",
        "search_player_moves": 2 if policy_name == "expectimax-2" else 1,
        "elapsed_seconds": elapsed,
        "inference_seconds": inference_seconds,
        "inference_batches": inference_batches,
        "inference_actions_per_second": float(length_array.sum()) / max(inference_seconds, 1e-6),
        "timing_scope": "batched action selection including preparation and device transfer; throughput is not single-game latency",
    }


def print_report(result: dict, checkpoint_path: str, version: str) -> None:
    print("=" * 62)
    print(f"policy: {result.get('policy', 'greedy')} | precision: float32")
    if version == "v3-afterstate-cnn-dqn":
        print("V3.0 Afterstate CNN-DQN 评估")
    else:
        print("V3.1 Afterstate Quantile CNN-DQN 评估")
    print(f"checkpoint: {checkpoint_path}")
    print("=" * 62)
    print(f"均分:       {result['mean_score']:10.1f}")
    print(f"中位分:     {result['median_score']:10.1f}")
    print(f"最高分:     {result['max_score']:10d}")
    print(f"标准差:     {result['std_score']:10.1f}")
    print(f">=512:      {result['rate_512'] * 100:9.1f}%")
    print(f">=1024:     {result['rate_1024'] * 100:9.1f}%")
    print(f">=2048:     {result['rate_2048'] * 100:9.1f}%")
    print(f">=4096:     {result['rate_4096'] * 100:9.1f}%")
    print(f"实际最高方块: {result['max_tile']}")
    print(f"平均步数:   {result['mean_length']:10.1f}")
    print("动作计数 [上, 下, 左, 右]:", result["action_counts"])
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3.1 FP32 策略评估")
    parser.add_argument("--policy", choices=("greedy", "expectimax-2"), default="greedy")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    if args.output and os.path.exists(args.output):
        parser.error(f"output already exists: {args.output}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.ckpt is None:
        args.ckpt = find_latest_checkpoint(os.path.join(base_dir, "models", "v3_1"))
    if args.ckpt is None:
        print("未找到 V3 checkpoint，请使用 --ckpt 指定路径")
        sys.exit(1)

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"评估设备: {device}")
    model, version, training_config = _load_model_with_config(args.ckpt, device)
    if args.gamma is None:
        args.gamma = training_config.get("gamma", 0.99)
    reward_mode = "log" if version == "v3-afterstate-cnn-dqn" else "transition"
    result = run_evaluation(
        model,
        device,
        args.episodes,
        args.gamma,
        args.seed,
        num_envs=args.num_envs,
        reward_mode=reward_mode,
        policy_name=args.policy,
    )
    result["checkpoint"] = os.path.abspath(args.ckpt)
    result["version"] = version
    print_report(result, args.ckpt, version)

    if args.output:
        with open(args.output, "x", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
        print(f"评估结果已写入: {args.output}")


if __name__ == "__main__":
    main()
