"""Plot V3 training history stored in a checkpoint."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return result
    cumulative = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    result[window - 1 :] = (
        cumulative[window:] - cumulative[:-window]
    ) / window
    return result


def plot_training(checkpoint_path: str, output_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    history = checkpoint["history"]
    episodes = np.asarray(history["episodes"], dtype=np.int64)
    scores = np.asarray(history["scores"], dtype=np.float64)
    max_tiles = np.asarray(history["max_tiles"], dtype=np.float64)
    lengths = np.asarray(history["episode_lengths"], dtype=np.float64)
    losses = np.asarray(history["losses"], dtype=np.float64)

    score_mean_100 = rolling_mean(scores, 100)
    score_mean_500 = rolling_mean(scores, 500)
    tile_mean_100 = rolling_mean(max_tiles, 100)
    length_mean_100 = rolling_mean(lengths, 100)
    loss_mean_100 = rolling_mean(losses, 100)

    success_100 = {
        ">=512": rolling_mean((max_tiles >= 512).astype(float), 100),
        ">=1024": rolling_mean((max_tiles >= 1024).astype(float), 100),
        ">=2048": rolling_mean((max_tiles >= 2048).astype(float), 100),
    }

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        f"V3 Afterstate CNN-DQN training history | {len(episodes):,} episodes",
        fontsize=16,
    )

    score_axis = axes[0, 0]
    score_axis.scatter(
        episodes,
        scores,
        s=3,
        alpha=0.12,
        color="#64748b",
        label="episode score",
    )
    score_axis.plot(
        episodes,
        score_mean_100,
        color="#2563eb",
        linewidth=1.8,
        label="100-episode mean",
    )
    score_axis.plot(
        episodes,
        score_mean_500,
        color="#dc2626",
        linewidth=2.2,
        label="500-episode mean",
    )
    score_axis.set_title("Score trend")
    score_axis.set_xlabel("Episode")
    score_axis.set_ylabel("Game score")
    score_axis.legend(loc="upper left")
    score_axis.grid(alpha=0.2)

    tile_axis = axes[0, 1]
    tile_axis.plot(
        episodes,
        tile_mean_100,
        color="#7c3aed",
        linewidth=2.0,
        label="100-episode mean max tile",
    )
    tile_axis.set_title("Max-tile achievement")
    tile_axis.set_xlabel("Episode")
    tile_axis.set_ylabel("Mean max tile")
    tile_axis.set_ylim(bottom=0)
    tile_axis.grid(alpha=0.2)
    rate_axis = tile_axis.twinx()
    rate_colors = {">=512": "#16a34a", ">=1024": "#ea580c", ">=2048": "#be123c"}
    for label, values in success_100.items():
        rate_axis.plot(
            episodes,
            values * 100.0,
            linewidth=1.6,
            color=rate_colors[label],
            label=f"{label} rate",
        )
    rate_axis.set_ylabel("Achievement rate (%)")
    rate_axis.set_ylim(0, 100)
    handles, labels = tile_axis.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    tile_axis.legend(handles + rate_handles, labels + rate_labels, loc="upper left")

    length_axis = axes[1, 0]
    length_axis.scatter(
        episodes,
        lengths,
        s=3,
        alpha=0.12,
        color="#64748b",
        label="episode length",
    )
    length_axis.plot(
        episodes,
        length_mean_100,
        color="#0891b2",
        linewidth=2.0,
        label="100-episode mean",
    )
    length_axis.set_title("Episode length")
    length_axis.set_xlabel("Episode")
    length_axis.set_ylabel("Steps")
    length_axis.legend(loc="upper left")
    length_axis.grid(alpha=0.2)

    loss_axis = axes[1, 1]
    loss_axis.plot(
        episodes,
        losses,
        color="#94a3b8",
        linewidth=0.7,
        alpha=0.55,
        label="episode mean loss",
    )
    loss_axis.plot(
        episodes,
        loss_mean_100,
        color="#9333ea",
        linewidth=2.0,
        label="100-episode mean",
    )
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Episode")
    loss_axis.set_ylabel("Smooth-L1 TD loss")
    loss_axis.set_ylim(bottom=0)
    loss_axis.legend(loc="upper right")
    loss_axis.grid(alpha=0.2)

    for axis in axes.flat:
        axis.axvline(5000, color="#475569", linestyle="--", linewidth=1.0)
        axis.text(
            5000,
            0.98,
            "ep5000",
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8,
            color="#475569",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V3 training history")
    parser.add_argument(
        "--checkpoint",
        default="models/v3/dqn_ep10000.pth",
    )
    parser.add_argument(
        "--output",
        default="training_curve.png",
    )
    args = parser.parse_args()
    plot_training(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
