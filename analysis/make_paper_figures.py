#!/usr/bin/env python3
"""Create publication-style static figures from validated GPT-2 LoRA aggregates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#3568B8"
GOLD = "#C58B1B"
INK = "#20242A"
MUTED = "#626A73"
GRID = "#D9DEE5"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, field: str) -> float:
    return float(row[field])


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_count(rows: list[dict]) -> int:
    counts = {int(row["runs"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"Expected one run count across configurations, found {sorted(counts)}")
    return counts.pop()


def performance_figure(rows: list[dict], output_dir: Path) -> None:
    n = run_count(rows)
    labels = [row["configuration"] for row in rows]
    accuracy = np.array([100 * num(row, "test_accuracy_mean") for row in rows])
    accuracy_sd = np.array([100 * num(row, "test_accuracy_std") for row in rows])
    macro_f1 = np.array([100 * num(row, "test_macro_f1_mean") for row in rows])
    macro_f1_sd = np.array([100 * num(row, "test_macro_f1_std") for row in rows])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.22)
    ax.errorbar(
        accuracy,
        y - 0.13,
        xerr=accuracy_sd,
        fmt="o",
        markersize=6,
        linewidth=1.5,
        capsize=3,
        color=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.6,
        label="Test accuracy",
    )
    ax.errorbar(
        macro_f1,
        y + 0.13,
        xerr=macro_f1_sd,
        fmt="s",
        markersize=5.5,
        linewidth=1.5,
        capsize=3,
        color=GOLD,
        markeredgecolor=INK,
        markeredgewidth=0.6,
        label="Test macro-F1",
    )
    for value, spread, row_y in zip(accuracy, accuracy_sd, y - 0.13):
        ax.text(value + spread + 0.28, row_y, f"{value:.2f}", va="center", fontsize=8, color=BLUE)
    for value, spread, row_y in zip(macro_f1, macro_f1_sd, y + 0.13):
        ax.text(value + spread + 0.28, row_y, f"{value:.2f}", va="center", fontsize=8, color=GOLD)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(30, 56)
    ax.set_xlabel("Score (%)", color=INK)
    fig.suptitle(
        "SST-5 test performance across training configurations",
        x=0.14,
        y=0.975,
        ha="left",
        color=INK,
        weight="semibold",
    )
    fig.text(
        0.14,
        0.915,
        f"Points are {n}-seed means; error bars show ±1 sample standard deviation (n={n}).",
        fontsize=9,
        color=MUTED,
        va="top",
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, output_dir, "test_performance")


def efficiency_figure(rows: list[dict], output_dir: Path) -> None:
    n = run_count(rows)
    labels = [row["configuration"] for row in rows]
    trainable_pct = np.array([100 * num(row, "trainable_fraction") for row in rows])
    checkpoint_mb = np.array([num(row, "best_checkpoint_mb_mean") for row in rows])
    peak_vram_gib = np.array([num(row, "peak_cuda_memory_mb_mean") / 1024 for row in rows])
    final_gap = np.array([100 * num(row, "final_generalization_gap_mean") for row in rows])
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.2))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.10, wspace=0.12)
    panels = [
        (trainable_pct, "Trainable parameters (%)", True, BLUE),
        (checkpoint_mb, "Best checkpoint (MiB)", True, GOLD),
        (final_gap, "Final train–validation gap (pp)", False, "#667085"),
    ]
    for index, (ax, (values, title, log_scale, color)) in enumerate(zip(axes, panels)):
        bars = ax.barh(y, values, height=0.58, color=color, edgecolor=INK, linewidth=0.5)
        if log_scale:
            ax.set_xscale("log")
        ax.set_yticks(y, labels if index == 0 else [])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, weight="semibold")
        ax.grid(axis="x", color=GRID, linewidth=0.8, which="both")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=8.5)
        for bar, value in zip(bars, values):
            if index == 0:
                label = f"{value:.3f}%" if value < 1 else f"{value:.0f}%"
            elif index == 1:
                label = f"{value:.2f}"
            else:
                label = f"{value:.1f}"
            ax.text(
                bar.get_width() * (1.10 if log_scale else 1.01) + (0 if log_scale else 0.15),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                fontsize=7.8,
                color=INK,
            )
    axes[0].set_xlim(0.0015, 190)
    axes[1].set_xlim(0.02, 1200)
    axes[2].set_xlim(0, 57)
    fig.suptitle("Parameter, storage, and overfitting costs", x=0.02, y=0.975, ha="left", fontsize=13, color=INK, weight="semibold")
    fig.text(
        0.01,
        0.925,
        f"Log scales are used for trainable share and checkpoint size; all values are {n}-seed means.",
        ha="left",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    save(fig, output_dir, "efficiency_and_overfitting")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.aggregate_csv)
    performance_figure(rows, args.output_dir)
    efficiency_figure(rows, args.output_dir)


if __name__ == "__main__":
    main()
