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
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ActionType, FinancialAction, FinancialObservation
    from ..models import AccountInfo, BillInfo, DebtInfo, RiskSignals
    from ..models import LoanOffer, MedicalEmergencyInfo, FestivalInfo
    from ..tasks import get_task_config, grade_episode
except ImportError:
    from models import ActionType, FinancialAction, FinancialObservation
    from models import AccountInfo, BillInfo, DebtInfo, RiskSignals
    from models import LoanOffer, MedicalEmergencyInfo, FestivalInfo
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
        self._rng = random.Random(0)

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

        # v2: Loan system
        self._loan_config: Optional[Dict[str, Any]] = None
        self._pending_formal_loans: List[Dict[str, Any]] = []

        # v2: Medical emergency system
        self._medical_emergencies_config: List[Dict[str, Any]] = []
        self._active_medical_emergency: Optional[Dict[str, Any]] = None

        # v2: Festival system
        self._festival_windows: List[Dict[str, Any]] = []
        self._active_festival: Optional[Dict[str, Any]] = None

        # v2: Anti-reward-hacking
        self._consecutive_do_nothing: int = 0
        self._action_history: List[str] = []
        self._savings_withdrawn_today: bool = False
        self._savings_deposited_today: bool = False
        self._negotiated_bills: set = set()

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
            "informal_loans_taken": 0,
            "formal_loans_taken": 0,
            "festive_loans_taken": 0,
            "emergencies_survived": 0,
            "emergencies_failed": 0,
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
            "informal_loans_taken": 0,
            "formal_loans_taken": 0,
            "festive_loans_taken": 0,
            "emergencies_survived": 0,
            "emergencies_failed": 0,
        }

        # Generate cycle-0 bills
        self._current_cycle = 0
        self._active_bills = []
        self._generate_cycle_bills(0)

        # v2: Initialize loan system
        self._loan_config = config.get("loan_config")
        self._pending_formal_loans = []

        # v2: Initialize medical emergency system
        self._medical_emergencies_config = copy.deepcopy(config.get("medical_emergencies", []))
        self._active_medical_emergency = None

        # v2: Initialize festival system
        self._festival_windows = copy.deepcopy(config.get("festival_windows", []))
        self._active_festival = None

        # v2: Anti-reward-hacking state
        self._consecutive_do_nothing = 0
        self._action_history = []
        self._savings_withdrawn_today = False
        self._savings_deposited_today = False
        self._negotiated_bills = set()

        # UPI: Initialize micro-transaction simulation
        self._upi_config = config.get("upi_config")
        self._upi_total_micro_spend = 0.0

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
        t_wall0 = time.perf_counter()

        def _wall_timeout_exceeded() -> bool:
            if timeout_s is None or float(timeout_s) <= 0:
                return False
            return (time.perf_counter() - t_wall0) > float(timeout_s)

        # Reset per-step counters
        self._today_events = []
        self._today_interest = 0.0
        self._today_on_time_payments = 0
        self._today_late_payments = 0
        self._today_defaults = 0
        self._today_debt_reward = 0.0
        self._yesterday_savings = self._savings
        self._yesterday_credit_score = self._credit_score
        # v2: Reset daily anti-churn flags
        self._savings_withdrawn_today = False
        self._savings_deposited_today = False

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

        if _wall_timeout_exceeded():
            raise TimeoutError(
                f"env.step exceeded timeout_s={timeout_s!r} (wall time {time.perf_counter() - t_wall0:.4f}s)"
            )

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
                    f"Income: ₹{entry['amount']:.0f} ({entry['label']})"
                )

        # Process expense events
        for evt in self._expense_events:
            if evt["day"] == day:
                self._checking -= evt["amount"]
                self._today_events.append(
                    f"Expense: ₹{evt['amount']:.0f} ({evt['description']})"
                )

        # Real-world uncertainty: occasional random unexpected expense (INR)
        if self._rng.random() < 0.02:  # 2% chance each day
            expense = round(self._rng.uniform(500.0, 5000.0), 2)
            self._checking -= expense
            self._today_events.append(f"Unexpected expense: ₹{expense:.0f}")

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

        # UPI: Daily micro-transactions (chai, auto, Swiggy, etc.)
        if self._upi_config:
            prob = self._upi_config.get("micro_spend_probability", 0.5)
            if self._rng.random() < prob:
                lo, hi = self._upi_config.get("daily_micro_spend_range", [100, 400])
                spend = round(self._rng.uniform(lo, hi), 0)
                cats = self._upi_config.get("categories", ["upi_merchant"])
                cat = cats[day % len(cats)]
                self._checking -= spend
                self._upi_total_micro_spend += spend
                self._today_events.append(
                    f"📱 UPI: ₹{spend:.0f} ({cat})"
                )
            # P2P pressure: friends/family asking for money
            p2p_days = self._upi_config.get("p2p_pressure_days", [])
            p2p_amounts = self._upi_config.get("p2p_amounts", [])
            for i, pd in enumerate(p2p_days):
                if day == pd and i < len(p2p_amounts):
                    amt = p2p_amounts[i]
                    self._checking -= amt
                    self._today_events.append(
                        f"📱 UPI P2P: Friend/family requested ₹{amt:.0f} — sent via PhonePe"
                    )

        # v2: Deliver pending formal loans
        delivered = []
        for loan in self._pending_formal_loans:
            if day >= loan["delivery_day"]:
                self._checking += loan["amount"]
                self._today_events.append(
                    f"💰 Bank loan of ₹{loan['amount']:.0f} deposited (APR: {loan['apr']}%)"
                )
                delivered.append(loan)
        for loan in delivered:
            self._pending_formal_loans.remove(loan)

        # v2: Trigger medical emergencies
        for emg in self._medical_emergencies_config:
            if emg["trigger_day"] == day and self._active_medical_emergency is None:
                deadline = day + emg["deadline_days"]
                self._active_medical_emergency = {
                    "id": f"medical_emergency_{day}",
                    "amount": emg["amount"],
                    "deadline_day": deadline,
                    "description": emg["description"],
                    "is_paid": False,
                }
                # Add as a special urgent bill
                self._active_bills.append({
                    "id": f"medical_emergency_{day}",
                    "amount": emg["amount"],
                    "due_day": deadline,
                    "category": "medical",
                    "is_paid": False,
                })
                self._history["total_bills_generated"] += 1
                self._today_events.append(
                    f"🚨 MEDICAL EMERGENCY: {emg['description']} — "
                    f"₹{emg['amount']:.0f} due by day {deadline}"
                )

        # v2: Check medical emergency deadline failure
        if (self._active_medical_emergency
                and not self._active_medical_emergency["is_paid"]
                and day > self._active_medical_emergency["deadline_day"]):
            self._history["emergencies_failed"] += 1
            self._credit_score = max(300, self._credit_score - 30)
            penalty = self._active_medical_emergency["amount"] * 0.20
            self._checking -= penalty
            self._history["total_late_fees"] += penalty
            self._today_events.append(
                f"🚨 EMERGENCY UNPAID — penalty ₹{penalty:.0f}, credit score -30"
            )
            self._active_medical_emergency = None

        # v2: Festival season activation and social costs
        for fw in self._festival_windows:
            if fw["start_day"] <= day <= fw["end_day"]:
                self._active_festival = fw
                # Mandatory daily social cost
                social_cost = fw["daily_social_cost"]
                if social_cost > 0:
                    self._checking -= social_cost
                    self._today_events.append(
                        f"🎉 {fw['name']}: social obligations cost ₹{social_cost:.0f}"
                    )
                # Pressure message
                msgs = fw.get("pressure_messages", [])
                if msgs:
                    msg = msgs[day % len(msgs)]
                    self._today_events.append(f"📢 {msg}")
                break
            elif day > fw["end_day"] and self._active_festival == fw:
                self._active_festival = None

    def _generate_cycle_bills(self, cycle: int) -> None:
        """Generate active bills for a 30-day cycle from templates.
        Bills have ±15% stochastic variance to prevent memorization."""
        for tmpl in self._bill_templates:
            base_amount = tmpl["amount"]
            # Apply bill changes (e.g., rent increase)
            for change in self._bill_changes:
                if change["bill_id"] == tmpl["id"] and (cycle * 30) >= change["from_day"]:
                    base_amount = change["new_amount"]
            # ±15% stochastic variance (seeded for reproducibility)
            variance = self._rng.uniform(-0.15, 0.15)
            amount = round(base_amount * (1.0 + variance), 0)
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

        # v2: Track action history for anti-reward-hacking
        self._action_history.append(at.value)
        if len(self._action_history) > 10:
            self._action_history = self._action_history[-10:]

        if at == ActionType.PAY_BILL_FULL:
            self._action_pay_bill(action.bill_id)
            self._consecutive_do_nothing = 0
        elif at == ActionType.PAY_MINIMUM:
            self._action_pay_minimum(action.debt_id)
            self._consecutive_do_nothing = 0
        elif at == ActionType.DEFER_BILL:
            self._action_defer_bill(action.bill_id)
            self._consecutive_do_nothing = 0
        elif at == ActionType.PAY_EXTRA_DEBT:
            self._action_pay_extra_debt(action.debt_id, action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.TRANSFER_TO_SAVINGS:
            self._action_transfer_to_savings(action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.WITHDRAW_EMERGENCY:
            self._action_withdraw_emergency(action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.DO_NOTHING:
            self._consecutive_do_nothing += 1
            self._today_events.append("No action taken")
        # v2: New actions
        elif at == ActionType.TAKE_FORMAL_LOAN:
            self._action_take_formal_loan(action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.TAKE_INFORMAL_LOAN:
            self._action_take_informal_loan(action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.TAKE_FESTIVE_LOAN:
            self._action_take_festive_loan(action.amount)
            self._consecutive_do_nothing = 0
        elif at == ActionType.NEGOTIATE_BILL:
            self._action_negotiate_bill(action.bill_id)
            self._consecutive_do_nothing = 0

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
                f"(need ₹{bill['amount']:.0f}, have ₹{self._checking:.0f})"
            )
            return

        self._checking -= bill["amount"]
        bill["is_paid"] = True
        on_time = self._current_day <= bill["due_day"]
        if on_time:
            self._today_on_time_payments += 1
            self._history["on_time_payment_count"] += 1
            self._today_events.append(
                f"Paid bill '{bill_id}' (₹{bill['amount']:.0f}) ON TIME"
            )
        else:
            self._today_events.append(
                f"Paid bill '{bill_id}' (₹{bill['amount']:.0f}) LATE"
            )

        # v2: Mark medical emergency as paid if this was an emergency bill
        if (self._active_medical_emergency
                and self._active_medical_emergency["id"] == bill_id):
            self._active_medical_emergency["is_paid"] = True
            self._history["emergencies_survived"] += 1
            self._today_events.append("✅ Medical emergency resolved!")
            self._active_medical_emergency = None

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
                f"(need ₹{payment:.0f}, have ₹{self._checking:.0f})"
            )
            return

        self._checking -= payment
        debt["principal"] = max(0.0, debt["principal"] - payment)
        debt["min_paid_this_cycle"] = True
        self._today_debt_reward += payment * (debt["apr"] / 100.0) * 0.5
        self._today_events.append(
            f"Paid minimum ₹{payment:.0f} on debt '{debt_id}' "
            f"(remaining: ₹{debt['principal']:.0f})"
        )

    def _action_defer_bill(self, bill_id: str) -> None:
        bill = self._find_active_bill(bill_id)
        if bill is None:
            self._today_events.append(f"Bill '{bill_id}' not found")
            return
        self._today_events.append(
            f"Deferred bill '{bill_id}' (₹{bill['amount']:.0f}, due day {bill['due_day']})"
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
            f"Extra payment ₹{actual:.0f} on '{debt_id}' "
            f"(remaining: ₹{debt['principal']:.0f})"
        )

    def _action_transfer_to_savings(self, amount: float) -> None:
        if self._checking < amount:
            self._today_events.append(
                f"Insufficient funds to transfer ₹{amount:.0f} to savings"
            )
            return
        self._checking -= amount
        self._savings += amount
        self._savings_deposited_today = True
        self._today_events.append(f"Transferred ₹{amount:.0f} to savings")

    def _action_withdraw_emergency(self, amount: float) -> None:
        actual = min(amount, self._savings)
        if actual <= 0:
            self._today_events.append("No savings available for withdrawal")
            return
        self._savings -= actual
        self._checking += actual
        self._savings_withdrawn_today = True
        self._today_events.append(f"Emergency withdrawal ₹{actual:.0f} from savings")

    # -------------------------------------------------------------------------
    # v2: New Action Handlers — Loans, Negotiation
    # -------------------------------------------------------------------------

    def _action_take_formal_loan(self, amount: float) -> None:
        """Apply for a formal bank loan — delayed delivery, reasonable APR."""
        if not self._loan_config or "formal" not in self._loan_config:
            self._today_events.append("No formal loan options available")
            return
        cfg = self._loan_config["formal"]
        if self._credit_score < cfg["min_credit_score"]:
            self._today_events.append(
                f"Loan DENIED — credit score {self._credit_score} below {cfg['min_credit_score']}"
            )
            return
        actual = min(amount, cfg["max_amount"])
        delivery_day = self._current_day + cfg["processing_days"]
        self._pending_formal_loans.append({"amount": actual, "apr": cfg["apr"], "delivery_day": delivery_day})
        min_due = round(actual * cfg.get("minimum_due_pct", 0.05), 2)
        self._debts.append({
            "id": f"bank_loan_{self._current_day}", "principal": actual, "apr": cfg["apr"],
            "minimum_due": min_due, "credit_limit": None, "min_payment_due_day": 15,
            "is_credit_card": False, "missed_payments": 0, "min_paid_this_cycle": False,
            "original_principal": actual,
        })
        self._history["formal_loans_taken"] += 1
        self._today_events.append(f"✅ Bank loan approved: ₹{actual:.0f} at {cfg['apr']}% APR — arrives day {delivery_day}")

    def _action_take_informal_loan(self, amount: float) -> None:
        """Take a predatory informal loan — instant cash, devastating APR."""
        if not self._loan_config or "informal" not in self._loan_config:
            self._today_events.append("No informal loan options available")
            return
        cfg = self._loan_config["informal"]
        actual = min(amount, cfg["max_amount"])
        self._checking += actual
        min_due = round(actual * cfg.get("minimum_due_pct", 0.10), 2)
        self._debts.append({
            "id": f"loan_shark_{self._current_day}", "principal": actual, "apr": cfg["apr"],
            "minimum_due": min_due, "credit_limit": None, "min_payment_due_day": 10,
            "is_credit_card": False, "missed_payments": 0, "min_paid_this_cycle": False,
            "original_principal": actual, "is_informal": True,
        })
        self._history["informal_loans_taken"] += 1
        self._today_events.append(f"⚠️ Informal loan taken: ₹{actual:.0f} INSTANT — but APR is {cfg['apr']}%!")

    def _action_take_festive_loan(self, amount: float) -> None:
        """Take a festive season loan — only during festival windows."""
        if self._active_festival is None:
            self._today_events.append("No festival season active — festive loan unavailable")
            return
        fw = self._active_festival
        actual = min(amount, fw.get("max_festive_loan", 3000.0))
        apr = fw.get("festive_loan_apr", 28.0)
        self._checking += actual
        min_due = round(actual * 0.08, 2)
        self._debts.append({
            "id": f"festive_loan_{self._current_day}", "principal": actual, "apr": apr,
            "minimum_due": min_due, "credit_limit": None, "min_payment_due_day": 20,
            "is_credit_card": False, "missed_payments": 0, "min_paid_this_cycle": False,
            "original_principal": actual,
        })
        self._history["festive_loans_taken"] += 1
        self._today_events.append(f"🎉 Festive loan taken: ₹{actual:.0f} — 'low EMI' but {apr}% APR")

    def _action_negotiate_bill(self, bill_id: str) -> None:
        """Attempt to negotiate a bill reduction — 40% success chance."""
        bill = self._find_active_bill(bill_id)
        if bill is None:
            self._today_events.append(f"Bill '{bill_id}' not found")
            return
        if bill["is_paid"]:
            self._today_events.append(f"Bill '{bill_id}' already paid")
            return
        if bill_id in self._negotiated_bills:
            self._today_events.append(f"Already negotiated '{bill_id}' this episode")
            return
        self._negotiated_bills.add(bill_id)
        if self._rng.random() < 0.40:
            discount = round(bill["amount"] * 0.15, 2)
            bill["amount"] = round(bill["amount"] - discount, 2)
            self._today_events.append(f"✅ Negotiation SUCCESS: '{bill_id}' reduced by ₹{discount:.0f}")
        else:
            self._today_events.append(f"❌ Negotiation FAILED: '{bill_id}' unchanged")

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
                    late_fee = max(500.0, bill["amount"] * 0.05)  # ₹500 or 5%
                    msg = "LATE FEE"
                    self._today_late_payments += 1
                    self._history["missed_payment_count"] += 1
                else:
                    late_fee = 100.0  # Escalating ₹100/day penalty for unpaid bills
                    msg = "ESCALATING LATE FEE"
                
                self._checking -= late_fee
                self._history["total_late_fees"] += late_fee
                self._today_events.append(
                    f"⚠ {msg}: Bill '{bill['id']}' ({bill['category']}) "
                    f"₹{bill['amount']:.0f} overdue — fee ₹{late_fee:.0f}"
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
            self._today_events.append(f"Interest accrued: ₹{total_interest:.0f}")

    def _detect_overdraft(self) -> None:
        """Track overdraft days and apply ₹500 fee on first day of overdraft."""
        if self._checking < 0:
            was_already_overdrawn = self._history["overdraft_days"] > 0
            self._history["overdraft_days"] += 1
            # Apply ₹500 fee once when first entering overdraft
            if not was_already_overdrawn:
                self._checking -= 500.0
                self._history["total_late_fees"] += 500.0
                self._today_events.append(
                    f"⚠ OVERDRAFT FEE: ₹500 charged. Balance: ₹{self._checking:.0f}"
                )
            else:
                self._today_events.append(
                    f"⚠ OVERDRAFT: Checking balance ₹{self._checking:.0f}"
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
            "predatory_loan_penalty": 0.0,
            "emergency_buffer_bonus": 0.0,
            "inaction_penalty": 0.0,
            "action_diversity": 0.0,
        }

        # --- Positive signals ---

        if self._checking >= 0:
            breakdown["no_overdraft"] = 5.0

        breakdown["bill_payment"] = self._today_on_time_payments * 10.0
        breakdown["debt_payment"] = self._today_debt_reward

        # v2: Anti-churn savings growth — no reward if withdrew and deposited same day
        if (self._savings > self._yesterday_savings + 0.01
                and not (self._savings_withdrawn_today and self._savings_deposited_today)
                and (self._savings - self._yesterday_savings) >= 500.0):
            breakdown["savings_growth"] = 6.0

        credit_delta = self._credit_score - self._yesterday_credit_score
        breakdown["credit_improvement"] = credit_delta * 0.5

        # v2: Emergency buffer bonus — reward maintaining a cushion
        total_upcoming = sum(b["amount"] for b in self._active_bills if not b["is_paid"])
        if self._checking + self._savings > total_upcoming * 1.2 and total_upcoming > 0:
            breakdown["emergency_buffer_bonus"] = 3.0

        # v2: Action diversity bonus
        if len(self._action_history) >= 7:
            unique_recent = len(set(self._action_history[-7:]))
            if unique_recent >= 3:
                breakdown["action_diversity"] = 1.0

        # --- Negative signals ---

        breakdown["late_payment"] = self._today_late_payments * -15.0

        if self._checking < 0:
            breakdown["overdraft_penalty"] = -25.0

        # Interest penalty scaled for INR (₹470/day interest → ~-9.4 penalty)
        breakdown["interest_penalty"] = (self._today_interest / 100.0) * -2.0
        breakdown["default_penalty"] = self._today_defaults * -50.0

        if self._savings <= 0.01:
            breakdown["zero_savings"] = -5.0

        # v2: Predatory loan penalty — taking informal/festive loans hurts
        if self._history.get("informal_loans_taken", 0) > 0:
            # Ongoing penalty for each informal loan still active
            shark_debt = sum(d["principal"] for d in self._debts if "loan_shark" in d["id"])
            if shark_debt > 0:
                breakdown["predatory_loan_penalty"] = -8.0

        # v2: Consecutive do-nothing penalty (anti-reward-hacking)
        if self._consecutive_do_nothing >= 3:
            breakdown["inaction_penalty"] = -2.0 * (self._consecutive_do_nothing - 2)

        reward = sum(breakdown.values())
        rounded = round(reward, 4)
        out: Dict[str, float] = {
            k: round(v, 4) for k, v in breakdown.items() if v != 0.0
        }
        out["total"] = rounded
        return rounded, out

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
                    # Partial observability: informal lender hides true APR
                    apr=d["apr"] if not d.get("is_informal") else round(d["apr"] * 0.1, 1),
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
            # v2: Loan offers, emergencies, festivals
            loan_offers=self._build_loan_offers(),
            active_emergency=self._build_emergency_obs(),
            festival=self._build_festival_obs(),
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
                risk += max(500.0, b["amount"] * 0.05)
        return risk

    # -------------------------------------------------------------------------
    # v2: Observation helpers for loans, emergencies, festivals
    # -------------------------------------------------------------------------

    def _build_loan_offers(self) -> list:
        """Build loan offer observations based on task config."""
        if not self._loan_config:
            return []
        offers = []
        if "formal" in self._loan_config:
            cfg = self._loan_config["formal"]
            offers.append(LoanOffer(
                loan_type="formal_bank",
                label=cfg.get("label", f"Bank Loan — {cfg['apr']}% APR"),
                max_amount=cfg["max_amount"],
                apr=cfg["apr"],
                processing_days=cfg["processing_days"],
                min_credit_score=cfg["min_credit_score"],
                available=self._credit_score >= cfg["min_credit_score"],
                warning=f"Requires credit score ≥ {cfg['min_credit_score']}. {cfg['processing_days']}-day processing.",
            ))
        if "informal" in self._loan_config:
            cfg = self._loan_config["informal"]
            offers.append(LoanOffer(
                loan_type="informal_lender",
                label=cfg.get("label", "Quick Cash — no paperwork"),
                max_amount=cfg["max_amount"],
                apr=cfg["apr"],
                processing_days=0,
                min_credit_score=0,
                available=True,
                warning="Instant approval! Easy terms!",  # Deliberately misleading
            ))
        if self._active_festival:
            fw = self._active_festival
            offers.append(LoanOffer(
                loan_type="festive",
                label=fw.get("festive_loan_label", f"Festival Offer — celebrate now!"),
                max_amount=fw.get("max_festive_loan", 3000.0),
                apr=fw.get("festive_loan_apr", 28.0),
                processing_days=0,
                min_credit_score=0,
                available=True,
                warning="Limited time festive offer! Low EMI!",
            ))
        return offers

    def _build_emergency_obs(self):
        """Build medical emergency observation."""
        if not self._active_medical_emergency:
            return None
        e = self._active_medical_emergency
        return MedicalEmergencyInfo(
            id=e["id"],
            amount=e["amount"],
            deadline_day=e["deadline_day"],
            description=e["description"],
            is_paid=e["is_paid"],
        )

    def _build_festival_obs(self):
        """Build festival observation."""
        if not self._active_festival:
            return None
        fw = self._active_festival
        return FestivalInfo(
            name=fw["name"],
            days_remaining=max(0, fw["end_day"] - self._current_day),
            daily_social_cost=fw.get("daily_social_cost", 0.0),
            festive_loan_available=True,
            festive_loan_apr=fw.get("festive_loan_apr", 28.0),
            pressure_message=fw.get("pressure_messages", [""])[self._current_day % max(1, len(fw.get("pressure_messages", [""])))],
        )
