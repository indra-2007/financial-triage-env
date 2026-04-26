"""Prove the environment is genuinely stochastic across seeds — even where the
rounded grader lands on the same value.

A careful reader of `heuristic_scores.json` will notice that `easy` and `medium`
give identical scores for every seed. That is *not* a seed-propagation bug;
the reward / grader are driven by discrete payment events which the heuristic
always resolves the same way, while the underlying RNG still drives daily UPI
spends, unexpected expenses, and salary jitter. This script makes that
distinction concrete by showing:

  - variance of `checking_balance` at day 10 across N seeds (non-zero)
  - variance of `_upi_total_micro_spend` across N seeds (non-zero)
  - variance of the episode `score` across N seeds (zero on easy/medium)

Run: python -m scripts.verify_stochasticity --seeds 30
Writes `stochasticity_report.json` and prints a human-readable table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.my_env_environment import MyEnvironment  # noqa: E402
from inference import _heuristic_action  # noqa: E402


def _std(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def collect(task: str, seeds: int, probe_day: int) -> Dict[str, object]:
    checkings: List[float] = []
    upi_totals: List[float] = []
    net_worth: List[float] = []
    scores: List[float] = []
    for seed in range(seeds):
        env = MyEnvironment()
        obs = env.reset(seed=seed, task_id=task)
        for _ in range(probe_day):
            obs = env.step(_heuristic_action(obs))
        checkings.append(env._checking)  # type: ignore[attr-defined]
        upi_totals.append(env._upi_total_micro_spend)  # type: ignore[attr-defined]
        net_worth.append(
            env._checking + env._savings - sum(d["principal"] for d in env._debts)  # type: ignore[attr-defined]
        )
        while not obs.done:
            obs = env.step(_heuristic_action(obs))
        scores.append(float(env.get_episode_score()))
    return {
        "probe_day": probe_day,
        "n": seeds,
        "checking_balance": {"std": _std(checkings), "min": min(checkings), "max": max(checkings)},
        "upi_total_micro_spend": {"std": _std(upi_totals), "min": min(upi_totals), "max": max(upi_totals)},
        "net_worth": {"std": _std(net_worth), "min": min(net_worth), "max": max(net_worth)},
        "episode_score": {"std": _std(scores), "min": min(scores), "max": max(scores)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--probe-day", type=int, default=10,
                    help="Snapshot env state after this many heuristic steps.")
    ap.add_argument("--out", type=Path, default=_ROOT / "stochasticity_report.json")
    args = ap.parse_args()

    report: Dict[str, object] = {}
    print(
        f"\n{'task':>6}  {'probe_day':>9}  "
        f"{'checking std':>14}  {'UPI_total std':>14}  "
        f"{'net_worth std':>14}  {'score std':>11}"
    )
    print("-" * 80)
    for task in ("easy", "medium", "hard"):
        r = collect(task, seeds=args.seeds, probe_day=args.probe_day)
        report[task] = r
        cs = r["checking_balance"]["std"]  # type: ignore[index]
        us = r["upi_total_micro_spend"]["std"]  # type: ignore[index]
        ns = r["net_worth"]["std"]  # type: ignore[index]
        ss = r["episode_score"]["std"]  # type: ignore[index]
        print(f"{task:>6}  {args.probe_day:>9}  {cs:>14,.2f}  {us:>14,.2f}  {ns:>14,.2f}  {ss:>11.4f}")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {args.out}")
    print(
        "\nRead: large checking / UPI / net_worth std with zero (or near-zero) score std on easy/medium\n"
        "means the *env* is stochastic but the *grader* is event-driven — exactly what the README says."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
