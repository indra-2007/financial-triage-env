"""Property tests for the reward function — the 14 terms have to do what the
README says they do.

Run:  pytest -q tests/test_reward_properties.py
(standard-library `unittest` fallback works too:
   python -m unittest tests.test_reward_properties)

Each test sets up a deterministic (task, seed, prefix) state, runs one step
under two alternative actions, and asserts the reward-breakdown inequality
the README advertises. No real ML dependencies are required.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import ActionType, FinancialAction  # noqa: E402
from server.my_env_environment import MyEnvironment  # noqa: E402
from inference import _heuristic_action  # noqa: E402


def _env_at(task: str, seed: int, prefix_steps: int = 0) -> tuple[MyEnvironment, object]:
    env = MyEnvironment()
    obs = env.reset(seed=seed, task_id=task)
    for _ in range(prefix_steps):
        obs = env.step(_heuristic_action(obs))
    return env, obs


class TestRewardProperties(unittest.TestCase):

    def test_on_time_bill_pay_beats_do_nothing(self) -> None:
        """Paying rent on day 0 should produce a strictly higher reward than doing nothing."""
        env_a, _ = _env_at("medium", seed=7, prefix_steps=0)
        env_b, _ = _env_at("medium", seed=7, prefix_steps=0)
        env_a.step(FinancialAction(action_type=ActionType.PAY_BILL_FULL, bill_id="rent"))
        env_b.step(FinancialAction(action_type=ActionType.DO_NOTHING))
        r_pay = env_a._last_breakdown["total"]
        r_skip = env_b._last_breakdown["total"]
        self.assertGreater(r_pay, r_skip,
                           f"bill pay should beat do_nothing; got pay={r_pay} skip={r_skip}")
        self.assertIn("bill_payment", env_a._last_breakdown)
        self.assertAlmostEqual(env_a._last_breakdown["bill_payment"], 10.0, places=3)
        self.assertNotIn("bill_payment", env_b._last_breakdown)

    def test_overdraft_penalty_fires_on_negative_checking(self) -> None:
        """Property of the reward function itself: whenever `_checking < 0` at
        compute time the `overdraft_penalty` term must be exactly -25.0, and
        `no_overdraft` must NOT fire. We call `_compute_reward` directly so this
        is a property of the reward dict, not of the transition rails."""
        env, _ = _env_at("hard", seed=0, prefix_steps=0)
        env._checking = -100.0
        _, breakdown = env._compute_reward()
        self.assertIn("overdraft_penalty", breakdown,
                      f"expected overdraft_penalty when checking<0; breakdown={breakdown}")
        self.assertAlmostEqual(breakdown["overdraft_penalty"], -25.0, places=3)
        self.assertNotIn("no_overdraft", breakdown,
                         "no_overdraft must not fire when checking<0")

    def test_do_nothing_streak_eventually_penalised(self) -> None:
        """Running `do_nothing` for six consecutive days must trigger the inaction penalty term."""
        env, obs = _env_at("hard", seed=0, prefix_steps=0)
        final_breakdown = None
        for _ in range(6):
            env.step(FinancialAction(action_type=ActionType.DO_NOTHING))
            final_breakdown = env._last_breakdown
        assert final_breakdown is not None
        self.assertIn("inaction_penalty", final_breakdown,
                      f"expected inaction_penalty after do-nothing streak; breakdown={final_breakdown}")
        self.assertLess(final_breakdown["inaction_penalty"], 0.0)

    def test_anti_churn_savings_no_same_day_credit(self) -> None:
        """If you move ₹1000 out of savings and then put ₹1000 back in on the same day,
        the `savings_growth` term must NOT fire — this is the same-day-churn guard."""
        env, obs = _env_at("hard", seed=0, prefix_steps=0)
        env._savings = 5000.0
        env._savings_withdrawn_today = True
        env._savings_deposited_today = True
        env._yesterday_savings = 5000.0
        env.step(FinancialAction(action_type=ActionType.DO_NOTHING))
        self.assertNotIn("savings_growth", env._last_breakdown,
                         f"savings_growth should be suppressed by anti-churn; got {env._last_breakdown}")

    def test_episode_score_is_in_unit_interval(self) -> None:
        """`grade_episode` must return a value strictly inside [0, 1] after a
        plain heuristic rollout on every difficulty."""
        for task in ("easy", "medium", "hard"):
            env, obs = _env_at(task, seed=0, prefix_steps=0)
            while not obs.done:
                obs = env.step(_heuristic_action(obs))
            score = env.get_episode_score()
            self.assertGreater(score, 0.0, f"{task} score collapsed to 0: {score}")
            self.assertLess(score, 1.0, f"{task} score pinned at 1: {score}")


if __name__ == "__main__":
    unittest.main()
