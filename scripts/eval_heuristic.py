"""Multi-seed evaluation of heuristic, do-nothing, and random baselines.

Writes:
  - `heuristic_scores.json` with full per-seed results + summary.
  - `heuristic_scores_ci.png` bar chart with 95% bootstrap CIs.

Why this exists:
  The notebook bar chart is a *single* eval pass; this script is the number
  every README claim about the baseline is computed from, and can be rerun
  in a few seconds to reproduce every figure. See README "How to re-run
  every claim".

Usage:
  python -m scripts.eval_heuristic --seeds 60 --out heuristic_scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_utils import (  # noqa: E402
    do_nothing_policy,
    greedy_apr_policy,
    heuristic_policy,
    random_valid_policy,
    run_episode,
    summarize,
)


TASKS = ("easy", "medium", "hard")
POLICIES = {
    "heuristic": heuristic_policy,
    "greedy_apr": greedy_apr_policy,
    "do_nothing": do_nothing_policy,
    "random_valid": random_valid_policy,
}


def run_all(seeds: int) -> Dict[str, Dict[str, Dict]]:
    """Return {policy: {task: {scores, returns, summary}}}."""
    results: Dict[str, Dict[str, Dict]] = {}
    for policy_name, policy in POLICIES.items():
        results[policy_name] = {}
        for task in TASKS:
            scores: List[float] = []
            returns: List[float] = []
            for seed in range(seeds):
                out = run_episode(policy, task_id=task, seed=seed)
                scores.append(out["score"])
                returns.append(out["total_return"])
            results[policy_name][task] = {
                "scores": scores,
                "returns": returns,
                "summary": summarize(scores),
                "return_summary": summarize(returns),
            }
            s = results[policy_name][task]["summary"]
            print(
                f"{policy_name:>12} {task:>6}  "
                f"mean={s['mean']:.4f}  std={s['std']:.4f}  "
                f"CI95=[{s['ci_low']:.4f}, {s['ci_high']:.4f}]  "
                f"min={s['min']:.4f}  max={s['max']:.4f}  n={s['n']}"
            )
    return results


def save_figure(results: Dict[str, Dict[str, Dict]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(TASKS)
    order = ["heuristic", "greedy_apr", "random_valid", "do_nothing"]
    colors = {
        "heuristic": "#ff8c42",
        "greedy_apr": "#2a9b8c",
        "random_valid": "#67a4d9",
        "do_nothing": "#888888",
    }

    means = {p: [results[p][t]["summary"]["mean"] for t in tasks] for p in order}
    lows = {p: [results[p][t]["summary"]["ci_low"] for t in tasks] for p in order}
    highs = {p: [results[p][t]["summary"]["ci_high"] for t in tasks] for p in order}

    x = np.arange(len(tasks))
    width = 0.20
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, p in enumerate(order):
        xs = x + (i - 1.5) * width
        y = means[p]
        lo = [y[j] - lows[p][j] for j in range(len(tasks))]
        hi = [highs[p][j] - y[j] for j in range(len(tasks))]
        ax.bar(xs, y, width, label=p.replace("_", " "),
               color=colors[p], edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, y, yerr=[lo, hi], fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3)
        for xi, yi in zip(xs, y):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t + "\n(30d / 60d / 90d)".split(" / ")[i] for i, t in enumerate(tasks)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean episode score (0-1)")
    ax.set_title("Reference policies — mean episode score with 95% bootstrap CI")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=60,
                        help="Seeds per (policy, task). Default: 60")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "heuristic_scores.json")
    parser.add_argument("--figure", type=Path,
                        default=_ROOT / "heuristic_scores_ci.png")
    args = parser.parse_args()

    results = run_all(seeds=args.seeds)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON    -> {args.out}")
    save_figure(results, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
