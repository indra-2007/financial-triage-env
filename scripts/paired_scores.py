"""Seed-matched paired-score figure for the hard task.

The multi-seed bar chart in the README hides seed-level behavior — on hard
the heuristic mean only beats `random_valid` by Δ ≈ 0.002, but that rounded
summary does not show *which* policy wins on each individual seed. This
script produces a per-seed scatter / violin figure so a judge can see the
overlap directly, using the already-committed `heuristic_scores.json`.

Run: python -m scripts.paired_scores
Writes `paired_scores_hard.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=_ROOT / "heuristic_scores.json")
    ap.add_argument("--out", type=Path, default=_ROOT / "paired_scores_hard.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(args.src.read_text())

    policies = [p for p in ("heuristic", "greedy_apr", "random_valid", "do_nothing") if p in data]
    colors = {
        "heuristic": "#ff8c42",
        "greedy_apr": "#2a9b8c",
        "random_valid": "#67a4d9",
        "do_nothing": "#888888",
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [2, 1.3]}
    )

    for i, p in enumerate(policies):
        ys = data[p]["hard"]["scores"]
        xs = np.full_like(ys, i, dtype=float) + (np.random.default_rng(i).uniform(-0.15, 0.15, size=len(ys)))
        ax_top.scatter(xs, ys, s=18, alpha=0.75, color=colors[p], edgecolor="black", linewidth=0.3)
        ax_top.plot([i - 0.25, i + 0.25],
                    [np.mean(ys)] * 2, color="black", linewidth=2.5)
    ax_top.set_xticks(range(len(policies)))
    ax_top.set_xticklabels([p.replace("_", " ") for p in policies])
    ax_top.set_ylabel("Episode score (0-1) — hard task")
    ax_top.set_title("Per-seed episode score on hard (n=60). Black bars = means.")
    ax_top.set_ylim(0.24, 0.55)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)

    if "heuristic" in data and "random_valid" in data:
        h = data["heuristic"]["hard"]["scores"]
        r = data["random_valid"]["hard"]["scores"]
        n = min(len(h), len(r))
        deltas = [h[i] - r[i] for i in range(n)]
        wins = sum(1 for d in deltas if d > 0)
        ax_bot.hist(deltas, bins=18, color="#ff8c42", edgecolor="black", linewidth=0.4, alpha=0.85)
        ax_bot.axvline(0, color="black", linewidth=1)
        ax_bot.axvline(float(np.mean(deltas)), color="#d9534f", linewidth=2, linestyle="--",
                       label=f"mean Δ = {float(np.mean(deltas)):+.4f}")
        ax_bot.set_xlabel("heuristic - random_valid (per seed, hard)")
        ax_bot.set_ylabel("seeds")
        ax_bot.set_title(
            f"Seed-matched heuristic − random_valid on hard: "
            f"heuristic wins on {wins}/{n} seeds, loses on {n - wins}"
        )
        ax_bot.legend(loc="upper left", frameon=False)
        ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
