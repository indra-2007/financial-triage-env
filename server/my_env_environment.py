# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Personal Financial Triage & Life Budget Advisor — Core Environment.

Simulates a realistic personal finance system where an AI agent manages
cash flow, bills, debts, credit score, and savings over multi-day episodes.

1 step = 1 day. State evolves every day with interest accrual, bill deadlines,
salary deposits, emergency expenses, and credit score dynamics.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, FinancialAction, FinancialObservation
    from ..models import AccountInfo, BillInfo, DebtInfo, RiskSignals
    from ..tasks import get_task_config, grade_episode
except ImportError:
    from models import ActionType, FinancialAction, FinancialObservation
    from models import AccountInfo, BillInfo, DebtInfo, RiskSignals
    from tasks import get_task_config, grade_episode


class MyEnvironment(Environment):
    """
    Production-grade personal finance RL environment.

    Implements the full OpenEnv interface: reset(), step(), state.
    Supports 3 task difficulties via task_id kwarg in reset().
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._task_id: str = "easy"
        self._episode_length: int = 30
        self._current_day: int = 0

        # Finances
        self._checking: float = 0.0
        self._savings: float = 0.0
        self._credit_score: int = 700

        # Bill system
        self._bill_templates: List[Dict[str, Any]] = []
        self._active_bills: List[Dict[str, Any]] = []
        self._current_cycle: int = 0

        # Debt system
        self._debts: List[Dict[str, Any]] = []
        self._total_credit_limit: float = 0.0

        # Income & events
        self._salary_schedule: List[Dict[str, Any]] = []
        self._expense_events: List[Dict[str, Any]] = []
        self._bill_changes: List[Dict[str, Any]] = []

        # Per-step tracking
        self._today_events: List[str] = []
        self._yesterday_savings: float = 0.0
        self._yesterday_credit_score: int = 700
        self._today_interest: float = 0.0
        self._today_on_time_payments: int = 0
        self._today_late_payments: int = 0
        self._today_defaults: int = 0
        self._today_debt_reward: float = 0.0

        # Episode history for grading (initialized with defaults for safety)
        self._history: Dict[str, Any] = {
            "overdraft_days": 0,
            "total_late_fees": 0.0,
            "total_interest_paid": 0.0,
            "missed_payment_count": 0,
            "on_time_payment_count": 0,
            "total_bills_generated": 0,
            "defaults": 0,
            "initial_credit_score": 700,
            "initial_savings": 0.0,
            "daily_rewards": [],
        }

    # -------------------------------------------------------------------------
    # OpenEnv API: reset
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FinancialObservation:
        """Reset environment with a task configuration."""
        # Reproducibility: set RNG seed if provided
        self._rng = random.Random(seed)

        self._task_id = kwargs.get("task_id", "easy")
        config = get_task_config(self._task_id)

        self._episode_id = episode_id or str(uuid4())
        self._episode_length = config["episode_length"]
        self._current_day = 0

        # Initialize finances
        self._checking = config["initial_checking"]
        self._savings = config["initial_savings"]
        self._credit_score = config["initial_credit_score"]

        # Store templates
        self._bill_templates = config["bill_templates"]
        self._bill_changes = config["bill_changes"]
        self._expense_events = copy.deepcopy(config["expense_events"])
        self._total_credit_limit = config["total_credit_limit"]
        
        # Real-world uncertainty: Apply slight variations to salary schedule
        self._salary_schedule = copy.deepcopy(config["salary_schedule"])
        for s in self._salary_schedule:
            # 20% chance of a 1-day delay; +/- 5% amount variation
            if self._rng.random() < 0.2:
                s["day"] += 1
            s["amount"] = round(s["amount"] * self._rng.uniform(0.95, 1.05), 2)

        # Initialize debts from templates
        self._debts = []
        for dt in config["debt_templates"]:
            self._debts.append({
                "id": dt["id"],
                "principal": dt["principal"],
                "apr": dt["apr"],
                "minimum_due": dt["minimum_due"],
                "credit_limit": dt.get("credit_limit"),
                "min_payment_due_day": dt["min_payment_due_day"],
                "is_credit_card": dt.get("is_credit_card", False),
                "missed_payments": 0,
                "min_paid_this_cycle": False,
                "original_principal": dt["principal"],
            })

        # Initialize history BEFORE bill generation (bills increment counter)
        self._history = {
            "overdraft_days": 0,
            "total_late_fees": 0.0,
            "total_interest_paid": 0.0,
            "missed_payment_count": 0,
            "on_time_payment_count": 0,
            "total_bills_generated": 0,
            "defaults": 0,
            "initial_credit_score": self._credit_score,
            "initial_savings": self._savings,
            "daily_rewards": [],
        }

        # Generate cycle-0 bills
        self._current_cycle = 0
        self._active_bills = []
        self._generate_cycle_bills(0)

        # Snapshot for delta tracking
        self._yesterday_savings = self._savings
        self._yesterday_credit_score = self._credit_score
        self._today_events = []
        self._last_breakdown = {}

        return self._build_observation(reward=0.0, done=False)

    # -------------------------------------------------------------------------
    # OpenEnv API: step
    # -------------------------------------------------------------------------

    def step(
        self,
        action: FinancialAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FinancialObservation:
        """Execute one day step in the financial simulation."""
        # Reset per-step counters
        self._today_events = []
        self._today_interest = 0.0
        self._today_on_time_payments = 0
        self._today_late_payments = 0
        self._today_defaults = 0
        self._today_debt_reward = 0.0
        self._yesterday_savings = self._savings
        self._yesterday_credit_score = self._credit_score

        # 1. Process begin-of-day events (salary, expenses, new cycle)
        self._process_begin_of_day()

        # 2. Execute agent's action
        self._process_action(action)

        # 3. End-of-day: bill deadlines, interest, debt checks, credit score
        self._process_end_of_day()

        # 4. Compute dense reward
        reward, breakdown = self._compute_reward()

        # 5. Advance day
        self._current_day += 1
        done = self._current_day >= self._episode_length

        # 6. End-of-episode bonus
        if done:
            credit_improvement = self._credit_score - self._history["initial_credit_score"]
            if credit_improvement > 0:
                score_bonus = min(20.0, credit_improvement * 0.2)  # Cap the bonus
                reward += score_bonus
                breakdown["credit_improvement"] = breakdown.get("credit_improvement", 0.0) + score_bonus

        self._history["daily_rewards"].append(reward)
        self._last_breakdown = breakdown

        return self._build_observation(reward=reward, done=done)

    # -------------------------------------------------------------------------
    # OpenEnv API: state
    # -------------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return internal environment state for debugging / introspection."""
        return State(
            episode_id=self._episode_id,
            step_count=self._current_day,
            task_id=self._task_id,
            checking=round(self._checking, 2),
            savings=round(self._savings, 2),
            credit_score=self._credit_score,
            total_interest_paid=round(self._history.get("total_interest_paid", 0), 2),
            total_late_fees=round(self._history.get("total_late_fees", 0), 2),
            overdraft_days=self._history.get("overdraft_days", 0),
            defaults=self._history.get("defaults", 0),
        )

    # -------------------------------------------------------------------------
    # Grading API
    # -------------------------------------------------------------------------

    def get_episode_score(self) -> float:
        """Grade the current episode. Call after done=True."""
        final = {
            "checking": self._checking,
            "savings": self._savings,
            "credit_score": self._credit_score,
            "debts": [{"id": d["id"], "principal": d["principal"],
                        "missed_payments": d["missed_payments"]} for d in self._debts],
        }
        return grade_episode(self._task_id, self._history, final)

    # -------------------------------------------------------------------------
    # Begin-of-day processing
    # -------------------------------------------------------------------------

    def _process_begin_of_day(self) -> None:
        """Credit salary, process expense events, start new billing cycle."""
        day = self._current_day

        # Credit salary / income
        for entry in self._salary_schedule:
            if entry["day"] == day:
                self._checking += entry["amount"]
                self._today_events.append(
                    f"Income: ${entry['amount']:.2f} ({entry['label']})"
                )

        # Process expense events
        for evt in self._expense_events:
            if evt["day"] == day:
                self._checking -= evt["amount"]
                self._today_events.append(
                    f"Expense: ${evt['amount']:.2f} ({evt['description']})"
                )

        # Real-world uncertainty: occasional random unexpected expense
        if self._rng.random() < 0.02:  # 2% chance each day for a minor emergency
            expense = round(self._rng.uniform(20.0, 150.0), 2)
            self._checking -= expense
            self._today_events.append(f"Unexpected random expense: ${expense:.2f}")

        # New billing cycle check (every 30 days)
        new_cycle = day // 30
        if new_cycle > self._current_cycle:
            self._current_cycle = new_cycle
            self._generate_cycle_bills(new_cycle)
            # Reset debt minimum-payment tracking
            for debt in self._debts:
                debt["min_paid_this_cycle"] = False
            self._today_events.append(f"New billing cycle {new_cycle} started")

        # Small savings interest (0.5% APY → daily rate)
        if self._savings > 0:
            daily_savings_rate = 0.005 / 365.0
            interest_earned = self._savings * daily_savings_rate
            self._savings += interest_earned

    def _generate_cycle_bills(self, cycle: int) -> None:
        """Generate active bills for a 30-day cycle from templates."""
        for tmpl in self._bill_templates:
            amount = tmpl["amount"]
            # Apply bill changes (e.g., rent increase)
            for change in self._bill_changes:
                if change["bill_id"] == tmpl["id"] and (cycle * 30) >= change["from_day"]:
                    amount = change["new_amount"]
            self._active_bills.append({
                "id": tmpl["id"],
                "amount": amount,
                "due_day": cycle * 30 + tmpl["base_due_day"],
                "category": tmpl["category"],
                "is_paid": False,
            })
            self._history["total_bills_generated"] += 1

    # -------------------------------------------------------------------------
    # Action processing
    # -------------------------------------------------------------------------

    def _process_action(self, action: FinancialAction) -> None:
        """Execute the agent's chosen action with full validation."""
        at = action.action_type

        if at == ActionType.PAY_BILL_FULL:
            self._action_pay_bill(action.bill_id)
        elif at == ActionType.PAY_MINIMUM:
            self._action_pay_minimum(action.debt_id)
        elif at == ActionType.DEFER_BILL:
            self._action_defer_bill(action.bill_id)
        elif at == ActionType.PAY_EXTRA_DEBT:
            self._action_pay_extra_debt(action.debt_id, action.amount)
        elif at == ActionType.TRANSFER_TO_SAVINGS:
            self._action_transfer_to_savings(action.amount)
        elif at == ActionType.WITHDRAW_EMERGENCY:
            self._action_withdraw_emergency(action.amount)
        elif at == ActionType.DO_NOTHING:
            self._today_events.append("No action taken")

    def _action_pay_bill(self, bill_id: str) -> None:
        bill = self._find_active_bill(bill_id)
        if bill is None:
            self._today_events.append(f"Bill '{bill_id}' not found or not in current cycle")
            return
        if bill["is_paid"]:
            self._today_events.append(f"Bill '{bill_id}' already paid this cycle")
            return
        if self._checking < bill["amount"]:
            self._today_events.append(
                f"Insufficient funds for bill '{bill_id}' "
                f"(need ${bill['amount']:.2f}, have ${self._checking:.2f})"
            )
            return

        self._checking -= bill["amount"]
        bill["is_paid"] = True
        on_time = self._current_day <= bill["due_day"]
        if on_time:
            self._today_on_time_payments += 1
            self._history["on_time_payment_count"] += 1
            self._today_events.append(
                f"Paid bill '{bill_id}' (${bill['amount']:.2f}) ON TIME"
            )
        else:
            self._today_events.append(
                f"Paid bill '{bill_id}' (${bill['amount']:.2f}) LATE"
            )

    def _action_pay_minimum(self, debt_id: str) -> None:
        debt = self._find_debt(debt_id)
        if debt is None:
            self._today_events.append(f"Debt '{debt_id}' not found")
            return
        if debt["principal"] <= 0:
            self._today_events.append(f"Debt '{debt_id}' already paid off")
            return
        if debt["min_paid_this_cycle"]:
            self._today_events.append(f"Minimum already paid for '{debt_id}' this cycle")
            return

        payment = min(debt["minimum_due"], debt["principal"])
        if self._checking < payment:
            self._today_events.append(
                f"Insufficient funds for minimum on '{debt_id}' "
                f"(need ${payment:.2f}, have ${self._checking:.2f})"
            )
            return

        self._checking -= payment
        debt["principal"] = max(0.0, debt["principal"] - payment)
        debt["min_paid_this_cycle"] = True
        self._today_debt_reward += payment * (debt["apr"] / 100.0) * 0.5
        self._today_events.append(
            f"Paid minimum ${payment:.2f} on debt '{debt_id}' "
            f"(remaining: ${debt['principal']:.2f})"
        )

    def _action_defer_bill(self, bill_id: str) -> None:
        bill = self._find_active_bill(bill_id)
        if bill is None:
            self._today_events.append(f"Bill '{bill_id}' not found")
            return
        self._today_events.append(
            f"Deferred bill '{bill_id}' (${bill['amount']:.2f}, due day {bill['due_day']})"
        )

    def _action_pay_extra_debt(self, debt_id: str, amount: float) -> None:
        debt = self._find_debt(debt_id)
        if debt is None:
            self._today_events.append(f"Debt '{debt_id}' not found")
            return
        if debt["principal"] <= 0:
            self._today_events.append(f"Debt '{debt_id}' already paid off")
            return

        actual = min(amount, debt["principal"])
        if self._checking < actual:
            self._today_events.append(
                f"Insufficient funds for extra payment on '{debt_id}'"
            )
            return

        self._checking -= actual
        debt["principal"] = max(0.0, debt["principal"] - actual)
        # Extra payment covers minimum if not yet paid
        if not debt["min_paid_this_cycle"] and actual >= debt["minimum_due"]:
            debt["min_paid_this_cycle"] = True
        self._today_debt_reward += actual * (debt["apr"] / 100.0) * 0.5
        self._today_events.append(
            f"Extra payment ${actual:.2f} on '{debt_id}' "
            f"(remaining: ${debt['principal']:.2f})"
        )

    def _action_transfer_to_savings(self, amount: float) -> None:
        if self._checking < amount:
            self._today_events.append(
                f"Insufficient funds to transfer ${amount:.2f} to savings"
            )
            return
        self._checking -= amount
        self._savings += amount
        self._today_events.append(f"Transferred ${amount:.2f} to savings")

    def _action_withdraw_emergency(self, amount: float) -> None:
        actual = min(amount, self._savings)
        if actual <= 0:
            self._today_events.append("No savings available for withdrawal")
            return
        self._savings -= actual
        self._checking += actual
        self._today_events.append(f"Emergency withdrawal ${actual:.2f} from savings")

    # -------------------------------------------------------------------------
    # End-of-day processing
    # -------------------------------------------------------------------------

    def _process_end_of_day(self) -> None:
        """Run all end-of-day financial events."""
        self._process_bill_deadlines()
        self._process_debt_deadlines()
        self._accrue_interest()
        self._detect_overdraft()
        self._update_credit_score()

    def _process_bill_deadlines(self) -> None:
        """Apply late fees for bills due today that weren't paid."""
        for bill in self._active_bills:
            if not bill["is_paid"] and self._current_day >= bill["due_day"]:
                days_late = self._current_day - bill["due_day"]
                if days_late == 0:
                    late_fee = max(35.0, bill["amount"] * 0.05)
                    msg = "LATE FEE"
                    self._today_late_payments += 1
                    self._history["missed_payment_count"] += 1
                else:
                    late_fee = 5.0  # Escalating $5 daily penalty for unpaid bills
                    msg = "ESCALATING LATE FEE"
                
                self._checking -= late_fee
                self._history["total_late_fees"] += late_fee
                self._today_events.append(
                    f"⚠ {msg}: Bill '{bill['id']}' ({bill['category']}) "
                    f"${bill['amount']:.2f} overdue — fee ${late_fee:.2f}"
                )

    def _process_debt_deadlines(self) -> None:
        """Check for missed minimum payments on debt due dates."""
        for debt in self._debts:
            if debt["principal"] <= 0:
                continue
            cycle = self._current_day // 30
            due_day = cycle * 30 + debt["min_payment_due_day"]
            if self._current_day == due_day and not debt["min_paid_this_cycle"]:
                debt["missed_payments"] += 1
                self._history["missed_payment_count"] += 1
                self._today_events.append(
                    f"⚠ MISSED minimum payment on '{debt['id']}' "
                    f"(consecutive: {debt['missed_payments']})"
                )
                # Default triggers exactly when missed_payments first hits 3
                if debt["missed_payments"] == 3:
                    self._today_defaults += 1
                    self._history["defaults"] += 1
                    self._today_events.append(
                        f"🚨 DEFAULT: Debt '{debt['id']}' has defaulted!"
                    )

    def _accrue_interest(self) -> None:
        """Accrue daily interest on all outstanding debts."""
        total_interest = 0.0
        for debt in self._debts:
            if debt["principal"] > 0 and debt["apr"] > 0:
                daily_rate = debt["apr"] / 100.0 / 365.0
                interest = debt["principal"] * daily_rate
                debt["principal"] += interest
                total_interest += interest
        self._today_interest = total_interest
        self._history["total_interest_paid"] += total_interest
        if total_interest > 0.01:
            self._today_events.append(f"Interest accrued: ${total_interest:.2f}")

    def _detect_overdraft(self) -> None:
        """Track overdraft days and apply $35 fee on first day of overdraft."""
        if self._checking < 0:
            was_already_overdrawn = self._history["overdraft_days"] > 0
            self._history["overdraft_days"] += 1
            # Apply $35 fee once when first entering overdraft
            if not was_already_overdrawn:
                self._checking -= 35.0
                self._history["total_late_fees"] += 35.0
                self._today_events.append(
                    f"⚠ OVERDRAFT FEE: $35.00 charged. Balance: ${self._checking:.2f}"
                )
            else:
                self._today_events.append(
                    f"⚠ OVERDRAFT: Checking balance ${self._checking:.2f}"
                )
        else:
            # Reset overdraft tracking when balance recovers
            # (so fee triggers again if they re-enter overdraft)
            if self._history["overdraft_days"] > 0:
                self._history["_prev_overdraft_days"] = self._history["overdraft_days"]
                # Don't reset — keep cumulative count for grading

    def _update_credit_score(self) -> None:
        """Update credit score based on today's financial activity."""
        delta = 0

        # Payment history impact (35% of FICO weight)
        if self._today_late_payments > 0:
            delta -= 15 * self._today_late_payments

        if self._today_on_time_payments > 0:
            delta += 5 * min(self._today_on_time_payments, 3)

        if self._today_defaults > 0:
            delta -= 50 * self._today_defaults

        # Credit utilization impact (30% of FICO weight)
        utilization = self._compute_utilization()
        if utilization > 0.75:
            delta -= 8
        elif utilization > 0.50:
            delta -= 4
        elif utilization > 0.30:
            delta -= 1
        elif 0.01 < utilization <= 0.10:
            delta += 2
        elif utilization <= 0.01 and self._total_credit_limit > 0:
            delta += 1

        # Stability bonus (no negative events today)
        if (
            self._today_late_payments == 0
            and self._today_defaults == 0
            and self._checking >= 0
        ):
            delta += 1

        # Missed debt payments drag
        total_missed = sum(d["missed_payments"] for d in self._debts)
        if total_missed > 0:
            delta -= min(total_missed, 5)

        # Gradual credit score change (prevent wild swings)
        delta = max(-10, min(10, delta))
        self._credit_score = max(300, min(850, self._credit_score + delta))

    # -------------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------------

    def _compute_reward(self) -> tuple[float, Dict[str, float]]:
        """
        Dense reward computed every step. Combines positive and negative signals.

        Positive:
          +5   no overdraft
          +10  per bill paid on time (this step)
          +var high-APR debt payment (scaled by APR)
          +6   savings increased
          +var credit score improvement (proportional)

        Negative:
          -25  per late payment
          -40  overdraft
          -var interest accrued (proportional)
          -80  per default
          -10  zero savings
        """
        breakdown = {
            "no_overdraft": 0.0,
            "bill_payment": 0.0,
            "debt_payment": 0.0,
            "savings_growth": 0.0,
            "credit_improvement": 0.0,
            "late_payment": 0.0,
            "overdraft_penalty": 0.0,
            "interest_penalty": 0.0,
            "default_penalty": 0.0,
            "zero_savings": 0.0,
        }

        # --- Positive signals ---

        # No overdraft
        if self._checking >= 0:
            breakdown["no_overdraft"] = 5.0

        # Bills paid on time
        breakdown["bill_payment"] = self._today_on_time_payments * 10.0

        # High-APR debt payment reward (from action processing)
        breakdown["debt_payment"] = self._today_debt_reward

        # Savings growth
        if self._savings > self._yesterday_savings + 0.01:
            breakdown["savings_growth"] = 6.0

        # Credit score daily delta
        credit_delta = self._credit_score - self._yesterday_credit_score
        breakdown["credit_improvement"] = credit_delta * 0.5

        # --- Negative signals ---

        # Late payments (Reduced extreme penalty)
        breakdown["late_payment"] = self._today_late_payments * -15.0

        # Overdraft (Reduced to allow partial success in hard task)
        if self._checking < 0:
            breakdown["overdraft_penalty"] = -25.0

        # Interest accrued (proportional penalty)
        breakdown["interest_penalty"] = self._today_interest * -2.0

        # Defaults (Reduced penalty to avoid tanking score completely)
        breakdown["default_penalty"] = self._today_defaults * -50.0

        # Zero savings penalty
        if self._savings <= 0.01:
            breakdown["zero_savings"] = -5.0

        reward = sum(breakdown.values())
        return round(reward, 4), {k: round(v, 4) for k, v in breakdown.items() if v != 0.0}

    # -------------------------------------------------------------------------
    # Observation builder
    # -------------------------------------------------------------------------

    def _build_observation(self, reward: float, done: bool) -> FinancialObservation:
        """Construct the full observation from current state."""
        total_debt = sum(d["principal"] for d in self._debts)
        net_worth = self._checking + self._savings - total_debt

        return FinancialObservation(
            account=AccountInfo(
                checking_balance=round(self._checking, 2),
                savings_balance=round(self._savings, 2),
                credit_utilization=round(self._compute_utilization(), 4),
                next_salary_day=self._next_salary_day(),
            ),
            bills=[
                BillInfo(
                    id=b["id"],
                    amount=round(b["amount"], 2),
                    due_day=b["due_day"],
                    category=b["category"],
                    is_paid=b["is_paid"],
                )
                for b in self._active_bills
                # Show all bills from current cycle (paid or unpaid)
                if b["due_day"] >= (self._current_cycle * 30)
            ],
            debts=[
                DebtInfo(
                    id=d["id"],
                    principal=round(d["principal"], 2),
                    apr=d["apr"],
                    minimum_due=round(d["minimum_due"], 2),
                    missed_payments=d["missed_payments"],
                )
                for d in self._debts
                if d["principal"] > 0
            ],
            risk=RiskSignals(
                days_to_overdraft=self._estimate_days_to_overdraft(),
                interest_today=round(self._today_interest, 2),
                late_fee_risk=round(self._compute_late_fee_risk(), 2),
                credit_score=self._credit_score,
            ),
            current_day=self._current_day,
            episode_length=self._episode_length,
            net_worth=round(net_worth, 2),
            daily_summary=" | ".join(self._today_events) if self._today_events else "No events",
            done=done,
            reward=reward,
            metadata={
                "task_id": self._task_id,
                "episode_id": self._episode_id,
                "step": self._current_day,
                "score": self.get_episode_score() if done else None,
                "info": {
                    "reward_breakdown": self._last_breakdown
                }
            },
        )

    # -------------------------------------------------------------------------
    # Helper / utility methods
    # -------------------------------------------------------------------------

    def _find_active_bill(self, bill_id: str) -> Optional[Dict[str, Any]]:
        """Find an active (current cycle) bill by ID."""
        for b in self._active_bills:
            if b["id"] == bill_id and b["due_day"] >= (self._current_cycle * 30):
                return b
        return None

    def _find_debt(self, debt_id: str) -> Optional[Dict[str, Any]]:
        """Find a debt by ID."""
        for d in self._debts:
            if d["id"] == debt_id:
                return d
        return None

    def _compute_utilization(self) -> float:
        """Compute credit utilization ratio across all credit cards."""
        if self._total_credit_limit <= 0:
            return 0.0
        total_cc_balance = sum(
            d["principal"] for d in self._debts if d.get("is_credit_card")
        )
        return min(1.0, total_cc_balance / self._total_credit_limit)

    def _next_salary_day(self) -> int:
        """Find the next salary day from current_day."""
        future = [
            s["day"] for s in self._salary_schedule if s["day"] > self._current_day
        ]
        if future:
            return min(future)
        # No more salary — return episode end
        return self._episode_length

    def _estimate_days_to_overdraft(self) -> int:
        """Estimate days until checking goes negative based on upcoming bills."""
        if self._checking <= 0:
            return 0

        # Estimate daily outflow from upcoming unpaid bills over next 30 days
        upcoming_bills_total = 0.0
        for b in self._active_bills:
            if not b["is_paid"] and b["due_day"] > self._current_day:
                upcoming_bills_total += b["amount"]

        # Daily interest on debts
        daily_interest = sum(
            d["principal"] * d["apr"] / 100.0 / 365.0
            for d in self._debts
            if d["principal"] > 0 and d["apr"] > 0
        )

        # Average daily outflow
        days_window = 30
        daily_outflow = (upcoming_bills_total / max(1, days_window)) + daily_interest

        if daily_outflow <= 0.001:
            return -1  # -1 means safe / no significant outflow

        days = int(self._checking / daily_outflow)
        return max(0, min(days, 999))

    def _compute_late_fee_risk(self) -> float:
        """Total late fees at risk from unpaid bills due in next 5 days."""
        risk = 0.0
        for b in self._active_bills:
            if (
                not b["is_paid"]
                and self._current_day < b["due_day"] <= self._current_day + 5
            ):
                risk += max(35.0, b["amount"] * 0.05)
        return risk
