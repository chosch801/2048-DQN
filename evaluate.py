"""2048 模型评估脚本。

加载训练好的 checkpoint，在纯贪心模式下跑若干局评估，
输出分数分布、方块达成率、移动偏好等完整统计。

用法:
    python evaluate.py                          # 加载默认 checkpoint
    python evaluate.py --ckpt models/5000轮/dqn_ep5000.pth
    python evaluate.py --episodes 100           # 只跑 100 局
    python evaluate.py --no-cuda                # 强制 CPU
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import Env2048
from model import TransformerDQN


def load_model(ckpt_path, device):
    """加载 checkpoint 并返回处于 eval 模式的模型。"""
    print(f"加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = TransformerDQN(num_quant=51).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def run_evaluation(model, device, num_episodes=1000):
    """运行纯贪心评估并收集统计。"""
    env = Env2048()

    scores = []
    max_tiles = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0=上 1=下 2=左 3=右
    tile_counts = {64: 0, 128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}

    print(f"开始评估 {num_episodes} 局...")
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        step = 0
        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_quantiles, _ = model(state_t)
            q_expected = q_quantiles.mean(dim=-1).cpu().numpy().flatten()
            q_expected[~info["valid_actions"]] = -np.inf
            action = int(np.argmax(q_expected))

            action_counts[action] += 1

            state, _, done, _, info = env.step(action)
            step += 1
            if done:
                scores.append(info["score"])
                max_tiles.append(info["max_tile"])
                episode_lengths.append(step)
                break

        if ep % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  已评估 {ep}/{num_episodes} 局  "
                  f"({ep / elapsed:.0f} 局/秒)")

    elapsed = time.time() - start_time
    print(f"评估完成，耗时 {elapsed:.1f} 秒 ({num_episodes / elapsed:.1f} 局/秒)\n")
    return scores, max_tiles, episode_lengths, action_counts


def print_report(scores, max_tiles, episode_lengths, action_counts, num_episodes):
    """打印评估报告。"""
    scores = np.array(scores)
    max_tiles = np.array(max_tiles)
    lengths = np.array(episode_lengths)

    print("=" * 62)
    print(f"  2048 模型评估报告 ({num_episodes} 局)")
    print("=" * 62)

    print(f"\n  分数统计")
    print(f"  {'─' * 42}")
    print(f"  均分:       {scores.mean():10.1f}")
    print(f"  中位分:     {np.median(scores):10.1f}")
    print(f"  最高分:     {scores.max():10.1f}")
    print(f"  最低分:     {scores.min():10.1f}")
    print(f"  标准差:     {scores.std():10.1f}")

    # 分段统计
    for low, high, label in [(0, 1000, "    0–1k"), (1000, 5000, "  1k–5k"),
                              (5000, 10000, " 5k–10k"), (10000, 20000, "10k–20k"),
                              (20000, 50000, "20k+" )]:
        count = int(np.sum((scores >= low) & (scores < high)))
        if count > 0:
            print(f"  {label}: {count:6d} 局  ({count / num_episodes * 100:5.1f}%)")

    print(f"\n  最高方块分布")
    print(f"  {'─' * 42}")
    for tile in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        count = int(np.sum(max_tiles >= tile))
        if count > 0 or tile >= 1024:
            print(f"  >= {tile:4d}: {count:6d} 局  "
                  f"({count / num_episodes * 100:5.1f}%)")
    print(f"  实际最高方块: {max_tiles.max()}")

    print(f"\n  每局步数")
    print(f"  {'─' * 42}")
    print(f"  均步数:     {lengths.mean():10.1f}")
    print(f"  最多步数:   {lengths.max():10.0f}")
    print(f"  最少步数:   {lengths.min():10.0f}")

    print(f"\n  移动方向偏好")
    print(f"  {'─' * 42}")
    total = sum(action_counts.values())
    directions = {0: "上(Up)", 1: "下(Down)", 2: "左(Left)", 3: "右(Right)"}
    for a in range(4):
        count = action_counts[a]
        print(f"  {directions[a]:<12}: {count:8d} 步  "
              f"({count / total * 100:5.1f}%)")

    print(f"\n" + "=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2048 模型评估")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint 路径（默认自动查找 models 下最新的）")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="评估局数（默认 1000）")
    parser.add_argument("--no-cuda", action="store_true",
                        help="强制使用 CPU")
    args = parser.parse_args()

    # 自动查找 checkpoint
    if args.ckpt is None:
        candidates = []
        for root, _, files in os.walk("models"):
            for f in files:
                if f.startswith("dqn_ep") and f.endswith(".pth"):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            print("未找到 checkpoint，请用 --ckpt 指定路径")
            sys.exit(1)
        args.ckpt = max(candidates,
                        key=lambda x: int(
                            os.path.basename(x).split("ep")[1].split(".pth")[0]))
        print(f"自动选择 checkpoint: {args.ckpt}")

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")
    print(f"设备: {device}")

    model = load_model(args.ckpt, device)
    scores, max_tiles, lengths, actions = run_evaluation(
        model, device, args.episodes)
    print_report(scores, max_tiles, lengths, actions, args.episodes)
