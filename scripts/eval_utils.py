# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Shared helpers for the judge-facing evaluation scripts. Keeps the eval
# definition of "a run" in one place so every figure and number the README
# cites is produced from the same code path.

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import ActionType, FinancialAction  # noqa: E402
from server.my_env_environment import MyEnvironment  # noqa: E402
from inference import _heuristic_action  # noqa: E402


Policy = Callable[[MyEnvironment, "object"], FinancialAction]
"""A policy is a function (env, observation) -> FinancialAction."""


def heuristic_policy(env: MyEnvironment, obs) -> FinancialAction:
    """The same rule-based baseline used to build SFT data and the 'Fill baseline' button."""
    return _heuristic_action(obs)


def do_nothing_policy(env: MyEnvironment, obs) -> FinancialAction:
    """Null baseline — skipping every day. Any non-trivial env must penalize this."""
    return FinancialAction(action_type=ActionType.DO_NOTHING)


def random_valid_policy(env: MyEnvironment, obs) -> FinancialAction:
    """Uniform random across 'safe' actions (do_nothing + pay_bill_full / pay_minimum if available).

    Kept intentionally conservative so the random baseline is *not* catastrophic —
    this makes it a fair lower bound on "a policy that at least doesn't skip every day".
    """
    pool: List[FinancialAction] = [FinancialAction(action_type=ActionType.DO_NOTHING)]
    for b in obs.bills or []:
        if not b.is_paid:
            pool.append(
                FinancialAction(action_type=ActionType.PAY_BILL_FULL, bill_id=b.id)
            )
    for d in obs.debts or []:
        if d.principal > 0:
            pool.append(
                FinancialAction(action_type=ActionType.PAY_MINIMUM, debt_id=d.id)
            )
    rng = random.Random(env._current_day * 1009 + 17)  # type: ignore[attr-defined]
    return rng.choice(pool)


def run_episode(
    policy: Policy,
    task_id: str,
    seed: int,
    env_patch: Optional[Callable[[MyEnvironment], None]] = None,
) -> Dict[str, float]:
    """Run one full episode under `policy` and return {score, total_return, days}.

    `env_patch` runs right after reset — used for ablations that null out a
    configured mechanic (e.g. disable UPI micro-spend) before the first step.
    """
    env = MyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    if env_patch is not None:
        env_patch(env)
    total_return = 0.0
    for _ in range(obs.episode_length):
        action = policy(env, obs)
        obs = env.step(action)
        total_return += float(getattr(obs, "reward", 0.0) or 0.0)
    return {
        "score": float(env.get_episode_score()),
        "total_return": float(total_return),
        "days": float(env._current_day),  # type: ignore[attr-defined]
    }


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def bootstrap_ci(
    values: Sequence[float],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Return a (low, high) percentile-bootstrap CI for the mean. Pure-python."""
    if len(values) == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Deterministic, judge-inspectable summary of an array of episode scores."""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "ci_low": 0.0, "ci_high": 0.0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    std = var ** 0.5
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ---------- Env ablation patches ----------

def _noop(env: MyEnvironment) -> None:
    return None


def ablate_upi(env: MyEnvironment) -> None:
    env._upi_config = None  # type: ignore[attr-defined]


def ablate_medical(env: MyEnvironment) -> None:
    env._medical_emergencies_config = []  # type: ignore[attr-defined]
    env._active_medical_emergency = None  # type: ignore[attr-defined]


def ablate_festival(env: MyEnvironment) -> None:
    env._festival_windows = []  # type: ignore[attr-defined]
    env._active_festival = None  # type: ignore[attr-defined]


ABLATIONS: Dict[str, Callable[[MyEnvironment], None]] = {
    "full": _noop,
    "no_upi_micro_spend": ablate_upi,
    "no_medical_emergency": ablate_medical,
    "no_festival": ablate_festival,
}
