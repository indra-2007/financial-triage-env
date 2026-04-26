"""Environment-mechanic ablation study.

For each difficulty, run the heuristic baseline under four env variants:
  - full               (the shipping env)
  - no_upi_micro_spend (disable daily chai / Swiggy leaks)
  - no_medical_emergency
  - no_festival

If scores *rise* when a mechanic is disabled, that mechanic is actually
binding on the baseline. This supports the README claim that the 14
reward terms and the partial-observability shocks are not decorative.

Outputs:
  - `ablation_env.json`
  - `ablation_env.png`

Usage:
  python -m scripts.ablation_env --seeds 40
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

from scripts.eval_utils import ABLATIONS, heuristic_policy, run_episode, summarize  # noqa: E402


TASKS = ("easy", "medium", "hard")


def run(seeds: int) -> Dict[str, Dict[str, Dict]]:
    results: Dict[str, Dict[str, Dict]] = {k: {} for k in ABLATIONS}
    for name, patch in ABLATIONS.items():
        for task in TASKS:
            scores: List[float] = []
            for seed in range(seeds):
                out = run_episode(heuristic_policy, task_id=task, seed=seed, env_patch=patch)
                scores.append(out["score"])
            results[name][task] = {
                "scores": scores,
                "summary": summarize(scores),
            }
            s = results[name][task]["summary"]
            print(
                f"{name:>22} {task:>6}  "
                f"mean={s['mean']:.4f}  std={s['std']:.4f}  "
                f"CI95=[{s['ci_low']:.4f}, {s['ci_high']:.4f}]  n={s['n']}"
            )
    return results


def save_figure(results: Dict[str, Dict[str, Dict]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(TASKS)
    order = ["full", "no_upi_micro_spend", "no_medical_emergency", "no_festival"]
    palette = {
        "full": "#2a9b8c",
        "no_upi_micro_spend": "#ff8c42",
        "no_medical_emergency": "#e85d6a",
        "no_festival": "#9b6bd1",
    }

    x = np.arange(len(tasks))
    width = 0.20
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, name in enumerate(order):
        xs = x + (i - 1.5) * width
        y = [results[name][t]["summary"]["mean"] for t in tasks]
        lo = [results[name][t]["summary"]["mean"] - results[name][t]["summary"]["ci_low"] for t in tasks]
        hi = [results[name][t]["summary"]["ci_high"] - results[name][t]["summary"]["mean"] for t in tasks]
        ax.bar(xs, y, width, label=name.replace("_", " "),
               color=palette[name], edgecolor="black", linewidth=0.4)
        ax.errorbar(xs, y, yerr=[lo, hi], fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3)
        for xi, yi in zip(xs, y):
            ax.text(xi, yi + 0.008, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Heuristic mean episode score (0-1)")
    ax.set_title("Env ablation: scores rise when a configured mechanic is disabled")
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--out", type=Path, default=_ROOT / "ablation_env.json")
    parser.add_argument("--figure", type=Path, default=_ROOT / "ablation_env.png")
    args = parser.parse_args()

    results = run(seeds=args.seeds)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON    -> {args.out}")
    save_figure(results, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
