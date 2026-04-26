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


def _paired_diff_ci(
    full_scores: list[float],
    abl_scores: list[float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_low, ci_high) of (abl - full) via paired bootstrap."""
    rng = rng or np.random.default_rng(0)
    diffs = np.asarray(abl_scores) - np.asarray(full_scores)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(boot_means, alpha / 2)), float(np.quantile(boot_means, 1 - alpha / 2))


def plot_ablations() -> Path:
    data = _load("ablation_env.json")
    ablations = [
        "no_medical_emergency",
        "no_interest_accrual",
        "no_upi_micro_spend",
        "no_peer_pressure",
        "no_festival",
    ]
    pretty = {
        "no_medical_emergency": "no medical emergency",
        "no_interest_accrual": "no interest accrual",
        "no_upi_micro_spend": "no UPI micro-spend",
        "no_peer_pressure": "no peer pressure",
        "no_festival": "no festival",
    }
    full_scores = data["full"]["hard"]["scores"]
    full_mean = float(np.mean(full_scores))
    n_seeds = data["full"]["hard"]["summary"]["n"]

    rows = []
    for ab in ablations:
        abl_scores = data[ab]["hard"]["scores"]
        abl_mean = float(np.mean(abl_scores))
        delta, lo, hi = _paired_diff_ci(full_scores, abl_scores)
        rows.append((ab, abl_mean, delta, lo, hi))
    rows.sort(key=lambda r: r[2], reverse=True)

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    y = np.arange(len(rows))
    deltas = [r[2] for r in rows]
    lows = [r[2] - r[3] for r in rows]
    highs = [r[4] - r[2] for r in rows]
    # Color the top bar as the accent; fade the rest in a single neutral.
    bar_colors = ["#E74C3C"] + ["#34495E"] * (len(rows) - 1)

    ax.barh(
        y, deltas, xerr=np.array([lows, highs]), capsize=4,
        color=bar_colors, edgecolor="black", linewidth=0.6, height=0.62,
    )
    # Pad so labels sit right next to each bar end (uniformly), not far to the right.
    max_right = max(d + h for d, h in zip(deltas, highs))
    pad = max_right * 0.015
    for i, (_, abl_mean, d, _, _) in enumerate(rows):
        bar_end = d + highs[i]
        ax.text(
            bar_end + pad, i,
            f"Δ {d:+.3f}   ({full_mean:.3f} → {abl_mean:.3f})",
            va="center", fontsize=10,
        )

    ax.axvline(0, color="#2C3E50", linestyle="--", linewidth=1.1)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty[r[0]] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"Δ mean episode score vs full env  (paired bootstrap, 95% CI; full env baseline = {full_mean:.3f})"
    )
    # Leave ~38% horizontal headroom for the right-hand text labels.
    ax.set_xlim(min(-0.015, min(lo for _, _, _, lo, _ in rows) * 1.2),
                max_right * 1.55)
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.suptitle(
        f"Environment ablation — hard task (heuristic policy, n={n_seeds} seeds, one mechanic nulled per run)",
        fontsize=12, y=0.995,
    )
    ax.set_title(
        "Positive Δ means the mechanic is binding on the baseline. "
        "Medium-task deltas are in ablation_env.json; only no_interest_accrual (+0.124) moves the medium grade.",
        fontsize=9.5, loc="left", color="#555", pad=6,
    )
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
