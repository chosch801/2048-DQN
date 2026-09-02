"""Pure-greedy evaluation for V3.

Expectimax is deliberately absent from this file.  It will be added later as
an independent inference policy so that V3's network quality can be measured
without search first.
"""

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
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    version = checkpoint.get("version")
    if version == "v3.1-afterstate-quantile-cnn-dqn":
        model: torch.nn.Module = AfterstateValueNet().to(device)
    elif version == "v3-afterstate-cnn-dqn":
        model = AfterstateValueNetV30().to(device)
    else:
        raise ValueError(f"unsupported V3 checkpoint version: {version}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, str(version)


@torch.inference_mode()
def run_evaluation(
    model: torch.nn.Module,
    device: torch.device,
    episodes: int,
    gamma: float,
    seed: int | None,
    num_envs: int = 64,
    reward_mode: str = "transition",
) -> dict:
    """Run deterministic greedy games in parallel without search."""

    episodes = int(episodes)
    num_envs = int(num_envs)
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")

    base_seed = (
        int(seed)
        if seed is not None
        else int(np.random.default_rng().integers(0, 2**31 - 1))
    )
    scores: list[int] = []
    max_tiles: list[int] = []
    lengths: list[int] = []
    action_counts = np.zeros(4, dtype=np.int64)
    start_time = time.time()

    completed = 0
    while completed < episodes:
        chunk_size = min(num_envs, episodes - completed)
        env = BatchEnv2048(num_envs=chunk_size, seed=base_seed + completed)
        _, masks = env.reset()
        active = np.ones(chunk_size, dtype=bool)
        episode_lengths = np.zeros(chunk_size, dtype=np.int64)

        while np.any(active):
            active_indices = np.flatnonzero(active)
            actions = choose_actions_batch(
                model,
                env.boards(),
                masks,
                device,
                gamma,
                epsilon=0.0,
                active_mask=active,
                amp_enabled=device.type == "cuda",
                reward_mode=reward_mode,
            )
            (
                _,
                _,
                dones,
                _,
                next_masks,
                infos,
            ) = env.step(actions, active_mask=active)
            action_counts[actions[active_indices]] += 1
            episode_lengths[active_indices] += 1

            for index in active_indices:
                if not dones[index]:
                    continue
                completed += 1
                scores.append(int(infos[index]["score"]))
                max_tiles.append(int(infos[index]["max_tile"]))
                lengths.append(int(episode_lengths[index]))
                active[index] = False

            masks = next_masks

        if completed % 100 == 0 or completed == episodes:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"已评估 {completed}/{episodes} 局 "
                f"({completed / elapsed:.1f} 局/秒)"
            )

    score_array = np.asarray(scores)
    tile_array = np.asarray(max_tiles)
    length_array = np.asarray(lengths)
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
        "max_tile": int(tile_array.max()),
        "mean_length": float(length_array.mean()),
        "max_length": int(length_array.max()),
        "action_counts": action_counts.tolist(),
    }


def print_report(result: dict, checkpoint_path: str, version: str) -> None:
    print("=" * 62)
    if version == "v3-afterstate-cnn-dqn":
        print("V3.0 Afterstate CNN-DQN 纯贪心评估")
    else:
        print("V3.1 Afterstate Quantile CNN-DQN 纯贪心评估")
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
    parser = argparse.ArgumentParser(description="V3.1 纯贪心评估")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

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
    model, version = load_model(args.ckpt, device)
    reward_mode = "log" if version == "v3-afterstate-cnn-dqn" else "transition"
    result = run_evaluation(
        model,
        device,
        args.episodes,
        args.gamma,
        args.seed,
        num_envs=args.num_envs,
        reward_mode=reward_mode,
    )
    print_report(result, args.ckpt, version)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
        print(f"评估结果已写入: {args.output}")


if __name__ == "__main__":
    main()
