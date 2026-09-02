"""Plot aggregate metrics from a V3 evaluation JSON file."""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_evaluation_summary(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as input_file:
        result = json.load(input_file)

    tile_labels = [">=512", ">=1024", ">=2048", ">=4096"]
    tile_rates = [result[f"rate_{label[2:]}"] * 100.0 for label in tile_labels]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.8),
        constrained_layout=True,
    )
    figure.suptitle(
        "V3.1 episode 15000 | 1000-game greedy evaluation summary",
        fontsize=16,
    )

    score_axis = axes[0]
    score_axis.hlines(
        0,
        result["min_score"],
        result["max_score"],
        color="#94a3b8",
        linewidth=5,
        label="min–max range",
    )
    score_axis.errorbar(
        result["mean_score"],
        0,
        xerr=result["std_score"],
        fmt="o",
        color="#2563eb",
        ecolor="#2563eb",
        capsize=6,
        markersize=8,
        label="mean ± std",
    )
    score_axis.scatter(
        result["median_score"],
        0,
        color="#dc2626",
        marker="D",
        s=70,
        label="median",
        zorder=3,
    )
    score_axis.set_title("Score summary")
    score_axis.set_xlabel("Game score")
    score_axis.set_yticks([])
    score_axis.grid(axis="x", alpha=0.2)
    score_axis.legend(loc="upper left")
    score_axis.text(
        0.02,
        0.08,
        f"mean  {result['mean_score']:,.1f}\n"
        f"median {result['median_score']:,.1f}\n"
        f"min–max {result['min_score']:,}–{result['max_score']:,}",
        transform=score_axis.transAxes,
        va="bottom",
        fontsize=10,
    )

    tile_axis = axes[1]
    bars = tile_axis.bar(
        tile_labels,
        tile_rates,
        color=["#16a34a", "#ea580c", "#be123c", "#0f766e"],
    )
    tile_axis.set_title("Max-tile achievement")
    tile_axis.set_xlabel("Highest tile reached")
    tile_axis.set_ylabel("Games reaching threshold (%)")
    tile_axis.set_ylim(0, 110)
    tile_axis.grid(axis="y", alpha=0.2)
    for bar, rate in zip(bars, tile_rates):
        tile_axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    length_axis = axes[2]
    length_labels = ["Mean length", "Longest game"]
    length_values = [result["mean_length"], result["max_length"]]
    bars = length_axis.bar(
        length_labels,
        length_values,
        color=["#0891b2", "#7c3aed"],
    )
    length_axis.set_title("Game length")
    length_axis.set_ylabel("Steps")
    length_axis.set_ylim(0, max(length_values) * 1.18)
    length_axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, length_values):
        length_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(length_values) * 0.02,
            f"{value:,.1f}" if isinstance(value, float) else f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aggregate V3 evaluation metrics"
    )
    parser.add_argument(
        "--input",
        default="models/v3_1/eval_ep15000_greedy_1000.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation_summary_ep15000.png",
    )
    args = parser.parse_args()
    plot_evaluation_summary(args.input, args.output)


if __name__ == "__main__":
    main()
