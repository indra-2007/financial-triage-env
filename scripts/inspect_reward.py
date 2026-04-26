"""Reward inspector — print every breakdown term for a chosen (task, seed, day, action).

A judge reading the README wants to see the 14 reward terms *actually fire* on a
concrete state, not only in prose. This script makes that a one-liner:

  python -m scripts.inspect_reward --task medium --seed 7 --day 5 --action pay_bill_full --arg rent
  python -m scripts.inspect_reward --task hard   --seed 0 --day 0 --action do_nothing
  python -m scripts.inspect_reward --task hard   --seed 0 --day 10 --action pay_extra_debt --arg 'loan_shark,5000'

It reconstructs the state deterministically by running the heuristic as a prefix,
then strictly parses the action string, runs one env.step, and prints the full
`_last_breakdown` dict with every term keyed by name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import ActionType, FinancialAction  # noqa: E402
from server.my_env_environment import MyEnvironment  # noqa: E402
from inference import _heuristic_action  # noqa: E402


def _parse_action(name: str, arg: str) -> FinancialAction:
    """Minimal parser for the action strings we expose via CLI."""
    args: List[str] = [a.strip() for a in arg.split(",") if a.strip()]
    if name == "do_nothing":
        return FinancialAction(action_type=ActionType.DO_NOTHING)
    if name == "pay_bill_full":
        return FinancialAction(action_type=ActionType.PAY_BILL_FULL, bill_id=args[0])
    if name == "pay_minimum":
        return FinancialAction(action_type=ActionType.PAY_MINIMUM, debt_id=args[0])
    if name == "pay_extra_debt":
        return FinancialAction(
            action_type=ActionType.PAY_EXTRA_DEBT,
            debt_id=args[0],
            amount=float(args[1]),
        )
    if name == "transfer_to_savings":
        return FinancialAction(
            action_type=ActionType.TRANSFER_TO_SAVINGS, amount=float(args[0]),
        )
    if name == "withdraw_emergency":
        return FinancialAction(
            action_type=ActionType.WITHDRAW_EMERGENCY, amount=float(args[0]),
        )
    raise SystemExit(f"unknown action: {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("easy", "medium", "hard"), default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--day", type=int, default=0,
                    help="Run the heuristic as a prefix for this many days before the inspected step.")
    ap.add_argument("--action", default="do_nothing",
                    help="Action name (do_nothing, pay_bill_full, pay_minimum, pay_extra_debt, …).")
    ap.add_argument("--arg", default="", help="Comma-separated action args, e.g. 'rent' or 'loan_shark,5000'.")
    args = ap.parse_args()

    env = MyEnvironment()
    obs = env.reset(seed=args.seed, task_id=args.task)
    for _ in range(args.day):
        obs = env.step(_heuristic_action(obs))

    action = _parse_action(args.action, args.arg)

    print(f"\n=== state BEFORE the inspected step (task={args.task}, seed={args.seed}, day={env._current_day}) ===")
    print(f"  checking   : ₹{env._checking:>12,.2f}")
    print(f"  savings    : ₹{env._savings:>12,.2f}")
    print(f"  total_debt : ₹{sum(d['principal'] for d in env._debts):>12,.2f}")
    print(f"  credit     : {env._credit_score}")
    print(f"  action     : {args.action}({args.arg})")

    obs_after = env.step(action)
    breakdown = env._last_breakdown or {}
    total = breakdown.get("total", obs_after.reward)

    print(f"\n=== reward breakdown AFTER the step (day={env._current_day}) ===")
    nonzero = {k: v for k, v in breakdown.items() if k != "total"}
    if not nonzero:
        print("  (no reward term fired — action produced a flat step)")
    else:
        for k, v in sorted(nonzero.items(), key=lambda kv: -abs(kv[1])):
            sign = "+" if v >= 0 else "−"
            print(f"  {sign}{abs(v):>8.3f}  {k}")
    print(f"  {'=' * 10}")
    print(f"  total       = {total:+.3f}   (observation.reward = {obs_after.reward:+.3f})")
    print(f"  done?       = {obs_after.done}")
    if obs_after.done:
        print(f"  episode_score = {env.get_episode_score():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
