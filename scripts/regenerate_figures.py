"""Regenerate the three README figures strictly from committed JSON/artifacts.

Usage:
    python -m scripts.regenerate_figures

Outputs (written to repo root, overwriting):
    heuristic_scores_ci.png      - 4 policies x 3 tasks with 95% bootstrap CI
    ablation_env.png             - 6 env ablations x 3 tasks with 95% bootstrap CI
    before_after_scores_7b.png   - Heuristic vs SFT (Qwen2.5-7B), single-pass n=5

Also refreshes the TRAINING_LOGS/ copies of the last two figures.

No extra inputs beyond heuristic_scores.json, ablation_env.json, and
TRAINING_LOGS/training_run.json. Any number in the README is recomputable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _task_order() -> list[str]:
    return ["easy", "medium", "hard"]


def _pretty_task(t: str) -> str:
    return {"easy": "easy (30d)", "medium": "medium (60d)", "hard": "hard (90d)"}[t]


def plot_baselines() -> Path:
    data = _load("heuristic_scores.json")
    policies = ["heuristic", "greedy_apr", "random_valid", "do_nothing"]
    pretty = {
        "heuristic": "heuristic",
        "greedy_apr": "greedy_apr",
        "random_valid": "random_valid",
        "do_nothing": "do_nothing",
    }
    colors = {
        "heuristic": "#E67E22",
        "greedy_apr": "#16A085",
        "random_valid": "#3498DB",
        "do_nothing": "#7F8C8D",
    }
    tasks = _task_order()
    n_seeds = data[policies[0]][tasks[0]]["summary"]["n"]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(tasks))
    width = 0.2

    for i, pol in enumerate(policies):
        means = [data[pol][t]["summary"]["mean"] for t in tasks]
        lows = [data[pol][t]["summary"]["ci_low"] for t in tasks]
        highs = [data[pol][t]["summary"]["ci_high"] for t in tasks]
        yerr = np.array([
            [m - lo for m, lo in zip(means, lows)],
            [hi - m for m, hi in zip(means, highs)],
        ])
        offset = (i - (len(policies) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=yerr, capsize=3,
            label=pretty[pol], color=colors[pol], edgecolor="black", linewidth=0.6,
        )
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks])
    ax.set_ylabel("Mean episode score (0–1)")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Reference policies — mean episode score with 95% bootstrap CI "
        f"(n={n_seeds} seeds per cell)",
        fontsize=12,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out = ROOT / "heuristic_scores_ci.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ablations() -> Path:
    data = _load("ablation_env.json")
    order = [
        "full",
        "no_medical_emergency",
        "no_interest_accrual",
        "no_upi_micro_spend",
        "no_peer_pressure",
        "no_festival",
    ]
    pretty = {
        "full": "full env",
        "no_medical_emergency": "no medical emergency",
        "no_interest_accrual": "no interest accrual",
        "no_upi_micro_spend": "no UPI micro-spend",
        "no_peer_pressure": "no peer pressure",
        "no_festival": "no festival",
    }
    colors = {
        "full": "#2C3E50",
        "no_medical_emergency": "#E74C3C",
        "no_interest_accrual": "#E67E22",
        "no_upi_micro_spend": "#F1C40F",
        "no_peer_pressure": "#9B59B6",
        "no_festival": "#3498DB",
    }
    tasks = _task_order()
    n_seeds = data["full"][tasks[0]]["summary"]["n"]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(tasks))
    width = 0.13

    for i, ab in enumerate(order):
        means = [data[ab][t]["summary"]["mean"] for t in tasks]
        lows = [data[ab][t]["summary"]["ci_low"] for t in tasks]
        highs = [data[ab][t]["summary"]["ci_high"] for t in tasks]
        yerr = np.array([
            [m - lo for m, lo in zip(means, lows)],
            [hi - m for m, hi in zip(means, highs)],
        ])
        offset = (i - (len(order) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=yerr, capsize=2,
            label=pretty[ab], color=colors[ab], edgecolor="black", linewidth=0.5,
        )
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks])
    ax.set_ylabel("Heuristic mean episode score (0–1)")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Environment ablation: score when a configured mechanic is disabled "
        f"(heuristic policy, n={n_seeds} seeds per cell)",
        fontsize=12,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    out = ROOT / "ablation_env.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_before_after() -> Path:
    tr = _load("TRAINING_LOGS/training_run.json")
    heur = tr["scores_from_notebook_run"]["heuristic_baseline_n5"]
    sft = tr["scores_from_notebook_run"]["sft_n5"]
    tasks = _task_order()

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(tasks))
    width = 0.35

    h_vals = [heur[t] for t in tasks]
    s_vals = [sft[t] for t in tasks]

    b1 = ax.bar(x - width / 2, h_vals, width, label="Heuristic (rule-based teacher)",
                color="#E67E22", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, s_vals, width, label="SFT on Qwen2.5-7B (LoRA, 60 steps)",
                color="#3498DB", edgecolor="black", linewidth=0.6)
    for bar_group, vals in [(b1, h_vals), (b2, s_vals)]:
        for b, v in zip(bar_group, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_task(t) for t in tasks])
    ax.set_ylabel("Episode score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Heuristic vs SFT (Qwen2.5-7B) — single evaluation pass, n=5 seeds per cell\n"
        "(60 SFT steps on 180 heuristic trajectories; GRPO numbers not plotted — not preserved in committed JSON)",
        fontsize=10.5,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out = ROOT / "before_after_scores_7b.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    a = plot_baselines()
    b = plot_ablations()
    c = plot_before_after()
    shutil.copy2(c, ROOT / "TRAINING_LOGS" / "before_after_scores.png")
    shutil.copy2(ROOT / "training_loss_7b.png", ROOT / "TRAINING_LOGS" / "training_loss.png")
    print("Wrote:")
    for p in (a, b, c):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
