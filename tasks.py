# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and deterministic graders for the Financial Triage Environment.

All amounts are in Indian Rupees (₹ / INR).
Calibrated against real Indian household finance data:
  - RBI Financial Literacy survey (2024): 27% adult literacy
  - NSSO health expenditure: mean OOP ₹1,69,504 private hospitalization
  - Credit card outstanding: ₹2,70,000 crore (June 2024)
  - Average household debt service ratio: 25.7% of disposable income

Three difficulty levels:
  - EASY  (30 days): Stable income, few bills, no debt. Goal: avoid overdraft.
  - MEDIUM (60 days): Multiple debts, irregular income, loan shark trap,
                      medical emergency, UPI micro-leaks. Goal: minimize interest.
  - HARD  (90 days): Job loss, predatory lenders, BNPL trap, medical emergencies,
                     Diwali pressure, UPI drain. Goal: survive without destroying credit.

Each grader returns a float score in [0.0, 1.0].

v2 additions:
  - Loan Shark Trap (medium, hard) — formal vs predatory lending
  - Medical Emergency Shock (medium, hard) — sudden mandatory expenses
  - Festival Season Pressure (hard) — Diwali social spending + temptation loans
  - UPI Micro-Transaction Drain (medium, hard) — daily small spends that compound
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# Task Configuration Builder
# =============================================================================


def _make_task(
    task_id: str,
    episode_length: int,
    checking: float,
    savings: float,
    credit_score: int,
    salary_schedule: List[Dict[str, Any]],
    bill_templates: List[Dict[str, Any]],
    debt_templates: List[Dict[str, Any]],
    expense_events: List[Dict[str, Any]],
    bill_changes: List[Dict[str, Any]],
    total_credit_limit: float,
    description: str,
    # --- v2 fields ---
    loan_config: Dict[str, Any] | None = None,
    medical_emergencies: List[Dict[str, Any]] | None = None,
    festival_windows: List[Dict[str, Any]] | None = None,
    # --- UPI simulation ---
    upi_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a validated task configuration dictionary."""
    return {
        "task_id": task_id,
        "episode_length": episode_length,
        "initial_checking": checking,
        "initial_savings": savings,
        "initial_credit_score": credit_score,
        "salary_schedule": salary_schedule,
        "bill_templates": bill_templates,
        "debt_templates": debt_templates,
        "expense_events": expense_events,
        "bill_changes": bill_changes,
        "total_credit_limit": total_credit_limit,
        "description": description,
        # v2
        "loan_config": loan_config,
        "medical_emergencies": medical_emergencies or [],
        "festival_windows": festival_windows or [],
        # UPI
        "upi_config": upi_config,
    }


# =============================================================================
# EASY TASK — 30 days (Fresh Graduate)
# =============================================================================

EASY_TASK = _make_task(
    task_id="easy",
    episode_length=30,
    checking=50000.0,     # ₹50,000 — 1-2 months buffer for a fresh graduate
    savings=10000.0,      # ₹10,000 — small savings
    credit_score=720,
    salary_schedule=[
        # Standard Indian salary: ₹30,000/month, paid monthly
        {"day": 0, "amount": 30000.0, "label": "Monthly salary (UPI credit)"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 12000.0, "base_due_day": 5, "category": "housing"},
        {"id": "electricity", "amount": 1800.0, "base_due_day": 10, "category": "utilities"},
        {"id": "mobile_recharge", "amount": 599.0, "base_due_day": 20, "category": "phone"},
    ],
    debt_templates=[],
    expense_events=[],
    bill_changes=[],
    total_credit_limit=0.0,
    description=(
        "Fresh graduate with ₹30,000/month salary via UPI. Three bills (rent, electricity, "
        "mobile recharge). No debt. Goal: pay all bills on time and build savings."
    ),
    # No v2 features for easy — keep it learnable
    loan_config=None,
    medical_emergencies=None,
    festival_windows=None,
    upi_config={
        "daily_micro_spend_range": [100, 400],  # Chai, auto, snacks
        "micro_spend_probability": 0.6,
        "categories": ["chai_stall", "auto_rickshaw", "street_food", "swiggy", "zepto"],
    },
)


# =============================================================================
# MEDIUM TASK — 60 days (Mid-Career, Debt Juggling)
# =============================================================================

MEDIUM_TASK = _make_task(
    task_id="medium",
    episode_length=60,
    checking=50000.0,     # ₹50,000 — moderate liquidity
    savings=12000.0,      # ₹12,000 — small buffer
    credit_score=660,
    salary_schedule=[
        {"day": 0, "amount": 55000.0, "label": "Monthly salary (UPI credit)"},
        {"day": 18, "amount": 20000.0, "label": "Freelance payment (UPI P2P)"},
        {"day": 30, "amount": 55000.0, "label": "Monthly salary (UPI credit)"},
        {"day": 45, "amount": 20000.0, "label": "Freelance payment (UPI P2P)"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 18000.0, "base_due_day": 3, "category": "housing"},
        {"id": "electricity", "amount": 2500.0, "base_due_day": 8, "category": "utilities"},
        {"id": "mobile_postpaid", "amount": 999.0, "base_due_day": 15, "category": "phone"},
        {"id": "health_insurance", "amount": 3500.0, "base_due_day": 20, "category": "insurance"},
        {"id": "ott_subscriptions", "amount": 800.0, "base_due_day": 25, "category": "subscription"},
    ],
    debt_templates=[
        {
            "id": "cc_hdfc",
            "principal": 85000.0,       # ₹85,000 credit card debt
            "apr": 42.0,                # Indian CC APR: 36-42%
            "minimum_due": 4500.0,      # ~5% of outstanding
            "credit_limit": 200000.0,   # ₹2,00,000 limit
            "min_payment_due_day": 12,
            "is_credit_card": True,
        },
        {
            "id": "personal_loan_sbi",
            "principal": 300000.0,      # ₹3,00,000 personal loan
            "apr": 14.0,                # SBI personal loan rate
            "minimum_due": 8000.0,      # EMI
            "credit_limit": None,
            "min_payment_due_day": 18,
            "is_credit_card": False,
        },
        {
            "id": "bnpl_flipkart",
            "principal": 25000.0,       # ₹25,000 Buy Now Pay Later
            "apr": 0.0,                 # 0% if paid on time, but penalty if missed
            "minimum_due": 5000.0,
            "credit_limit": None,
            "min_payment_due_day": 22,
            "is_credit_card": False,
        },
    ],
    expense_events=[],
    bill_changes=[],
    total_credit_limit=200000.0,
    description=(
        "Mid-career with ₹55,000/month salary + ₹20,000 irregular freelance via UPI. "
        "Five recurring bills. Three debts — HDFC credit card at 42% APR, "
        "SBI personal loan EMI ₹8,000, Flipkart BNPL ₹25,000. "
        "Loan shark trap active — local moneylender offers instant cash at 240% APR. "
        "Medical emergency strikes mid-episode. "
        "Goal: minimize interest, survive the shock, avoid predatory debt."
    ),
    # --- v2: Loan Shark Trap + Medical Emergency ---
    loan_config={
        "formal": {
            "apr": 14.0,
            "max_amount": 200000.0,     # ₹2,00,000 max bank loan
            "processing_days": 3,
            "min_credit_score": 650,
            "label": "SBI Personal Loan — 14% APR, 3-day processing",
            "minimum_due_pct": 0.05,
        },
        "informal": {
            "apr": 240.0,
            "max_amount": 50000.0,      # ₹50,000 max from moneylender
            "processing_days": 0,
            "min_credit_score": 0,
            "label": "Instant cash — no paperwork, settle today",
            "minimum_due_pct": 0.10,
        },
    },
    medical_emergencies=[
        {
            "trigger_day": 35,
            "amount": 45000.0,          # ₹45,000 — emergency dental surgery
            "deadline_days": 3,
            "description": "Emergency dental surgery — hospital demands advance payment",
        },
    ],
    festival_windows=None,
    upi_config={
        "daily_micro_spend_range": [150, 600],
        "micro_spend_probability": 0.75,
        "categories": ["swiggy", "zomato", "uber_auto", "chai_tapri", "amazon_quick", "blinkit"],
        "p2p_pressure_days": [20, 40],  # Friends asking for UPI transfers
        "p2p_amounts": [2000, 3000],
    },
)


# =============================================================================
# HARD TASK — 90 days (Job Loss Crisis)
# =============================================================================

HARD_TASK = _make_task(
    task_id="hard",
    episode_length=90,
    checking=40000.0,     # ₹40,000 — about 1.5 month essential expenses
    savings=10000.0,      # ₹10,000 — minimal buffer
    credit_score=640,
    salary_schedule=[
        # Full employment: day 0 only
        {"day": 0, "amount": 65000.0, "label": "Monthly salary (UPI credit)"},
        # Job loss at day 30 — gig economy income (Swiggy/Uber/freelance)
        {"day": 30, "amount": 28000.0, "label": "Gig income — Swiggy/Uber/freelance (UPI P2P)"},
        {"day": 60, "amount": 28000.0, "label": "Gig income — Swiggy/Uber/freelance (UPI P2P)"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 15000.0, "base_due_day": 3, "category": "housing"},
        {"id": "electricity", "amount": 3000.0, "base_due_day": 8, "category": "utilities"},
        {"id": "mobile_postpaid", "amount": 799.0, "base_due_day": 15, "category": "phone"},
        {"id": "health_insurance", "amount": 4500.0, "base_due_day": 20, "category": "insurance"},
        {"id": "bike_emi", "amount": 4200.0, "base_due_day": 22, "category": "transportation"},
        {"id": "ott_subscriptions", "amount": 500.0, "base_due_day": 25, "category": "subscription"},
    ],
    debt_templates=[
        {
            "id": "cc_hdfc_regalia",
            "principal": 150000.0,      # ₹1,50,000 — maxed out credit card
            "apr": 42.0,                # Indian CC APR
            "minimum_due": 7500.0,
            "credit_limit": 200000.0,
            "min_payment_due_day": 10,
            "is_credit_card": True,
        },
        {
            "id": "cc_icici_amazon",
            "principal": 65000.0,       # ₹65,000 — Amazon pay ICICI
            "apr": 39.6,                # Indian CC APR
            "minimum_due": 3250.0,
            "credit_limit": 100000.0,
            "min_payment_due_day": 14,
            "is_credit_card": True,
        },
        {
            "id": "personal_loan_hdfc",
            "principal": 500000.0,      # ₹5,00,000 personal loan
            "apr": 16.0,                # HDFC personal loan rate
            "minimum_due": 12000.0,     # EMI
            "credit_limit": None,
            "min_payment_due_day": 18,
            "is_credit_card": False,
        },
        {
            "id": "bnpl_simpl",
            "principal": 15000.0,       # ₹15,000 — Simpl/LazyPay BNPL
            "apr": 36.0,                # BNPL penalty APR when overdue
            "minimum_due": 3000.0,
            "credit_limit": None,
            "min_payment_due_day": 24,
            "is_credit_card": False,
        },
    ],
    expense_events=[
        {"day": 15, "amount": 8000.0, "description": "Bike service & repair (UPI payment)"},
    ],
    bill_changes=[
        # Rent increases starting cycle 2 (day 60+)
        {"bill_id": "rent", "from_day": 60, "new_amount": 17000.0},
    ],
    total_credit_limit=300000.0,
    description=(
        "Income drops from ₹65,000 to ₹28,000 (gig work: Swiggy/Uber/freelance) at day 30. "
        "Six bills including rent hike at day 60. Four debts totaling ₹7,30,000. "
        "Local moneylender circles at 365% APR. Two medical emergencies. "
        "Diwali season with social pressure and temptation loans. "
        "Goal: avoid default, maintain CIBIL score > 650, resist the loan shark."
    ),
    # --- v2: Full feature set ---
    loan_config={
        "formal": {
            "apr": 18.0,
            "max_amount": 150000.0,     # ₹1,50,000 max emergency bank loan
            "processing_days": 3,
            "min_credit_score": 550,    # Emergency loan — relaxed criteria
            "label": "SBI Emergency Loan — 18% APR, 3-day approval",
            "minimum_due_pct": 0.05,
        },
        "informal": {
            "apr": 365.0,               # Daily interest = 1% — real moneylender rate
            "max_amount": 50000.0,      # ₹50,000
            "processing_days": 0,
            "min_credit_score": 0,
            "label": "Instant cash — local lender, no questions asked",
            "minimum_due_pct": 0.10,
        },
    },
    medical_emergencies=[
        {
            "trigger_day": 20,
            "amount": 50000.0,          # ₹50,000 — emergency hospitalization
            "deadline_days": 5,         # 5 days to arrange funds
            "description": "Emergency hospitalization — hospital demands advance deposit",
        },
        {
            "trigger_day": 72,
            "amount": 35000.0,          # ₹35,000 — specialist consultation + tests
            "deadline_days": 5,
            "description": "Urgent specialist consultation — cannot be deferred",
        },
    ],
    festival_windows=[
        {
            "start_day": 42,
            "end_day": 48,
            "name": "Diwali Season",
            "daily_social_cost": 1500.0,    # ₹1,500/day — gifts, sweets, clothes
            "festive_loan_apr": 28.0,
            "max_festive_loan": 50000.0,    # ₹50,000 — festive personal loan
            "festive_loan_label": "Diwali Dhamaka Loan — Low EMI, celebrate now!",
            "pressure_messages": [
                "Family expects gifts — everyone around you is shopping on Amazon and Flipkart",
                "Festive loan offers flooding your inbox — '0% EMI for first 3 months!'",
                "Neighbors are bursting crackers and wearing new clothes — social pressure mounting",
                "Limited-time Diwali deals on electronics — fear of missing out on discounts",
                "Cultural obligations: skipping Diwali celebrations feels socially unacceptable",
                "WhatsApp groups full of Diwali party plans — everyone's spending ₹10,000+",
                "Parents expect you to send money home for Diwali — ₹5,000 minimum",
            ],
        },
    ],
    upi_config={
        "daily_micro_spend_range": [150, 600],
        "micro_spend_probability": 0.70,    # Most days
        "categories": ["swiggy", "zomato", "uber", "rapido", "chai", "blinkit",
                        "amazon_pay", "phone_pe_merchant", "paytm_mall"],
        "p2p_pressure_days": [10, 25, 40, 55, 75],
        "p2p_amounts": [2000, 5000, 3000, 2000, 5000],
        "autopay_bills": ["ott_subscriptions", "mobile_postpaid"],  # Auto-deduct via UPI
    },
)


# =============================================================================
# Task Registry
# =============================================================================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}


def get_task_config(task_id: str = "easy") -> Dict[str, Any]:
    """Get a task configuration by ID. Raises ValueError for unknown tasks."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    # Return a deep copy to prevent mutation
    import copy
    return copy.deepcopy(TASK_REGISTRY[task_id])


# =============================================================================
# Deterministic Graders — each returns float in [0.0, 1.0]
# =============================================================================


def _clamp(value: float) -> float:
    """Clamp to strict (0, 1) open interval — evaluator rejects 0.0 and 1.0."""
    return max(0.001, min(0.999, value))


def grade_easy(history: Dict[str, Any], final: Dict[str, Any]) -> float:
    """
    EASY grader — 30 days, stable income, no debt.

    Weights:
      30% — Overdraft avoidance
      35% — Bills paid on time
      20% — Savings maintained or grown
      15% — Credit score maintained
    """
    ep_len = 30

    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 2.0)

    total_bills = max(1, history["total_bills_generated"])
    on_time_ratio = history["on_time_payment_count"] / total_bills
    bill_score = _clamp(on_time_ratio)

    initial_savings = max(1.0, history["initial_savings"])
    savings_ratio = final["savings"] / initial_savings
    savings_score = _clamp(savings_ratio / 1.5)

    credit_delta = final["credit_score"] - history["initial_credit_score"]
    if credit_delta >= 0:
        credit_score = 1.0
    else:
        credit_score = _clamp(1.0 + credit_delta / 80.0)

    total = (
        0.30 * overdraft_score
        + 0.35 * bill_score
        + 0.20 * savings_score
        + 0.15 * credit_score
    )
    return round(_clamp(total), 4)


def grade_medium(history: Dict[str, Any], final: Dict[str, Any]) -> float:
    """
    MEDIUM grader — 60 days, debts, irregular income, loan trap, medical shock.

    Weights:
      15% — Overdraft avoidance
      20% — Bills paid on time
      20% — Interest minimization
      15% — Savings
      15% — Credit score
      15% — Financial wisdom (avoided predatory loans, survived emergencies)
    """
    ep_len = 60

    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 2.5)

    total_bills = max(1, history["total_bills_generated"])
    bill_score = _clamp(history["on_time_payment_count"] / total_bills)

    # Interest minimization: calibrated to INR worst-case
    # ~60 days on ₹4,10,000 blended ~30% APR ≈ ₹20,000
    worst_case_interest = 20000.0
    interest_ratio = history["total_interest_paid"] / max(1.0, worst_case_interest)
    interest_score = _clamp(1.0 - interest_ratio)

    initial_savings = max(1.0, history["initial_savings"])
    savings_score = _clamp(final["savings"] / (initial_savings * 2.0))

    credit_delta = final["credit_score"] - history["initial_credit_score"]
    if credit_delta >= 10:
        credit_score = 1.0
    elif credit_delta >= 0:
        credit_score = 0.8 + 0.2 * (credit_delta / 10.0)
    else:
        credit_score = _clamp(0.8 + credit_delta / 100.0)

    # Financial wisdom: penalize predatory loan usage, reward emergency survival
    wisdom_score = 1.0
    informal_loans = history.get("informal_loans_taken", 0)
    festive_loans = history.get("festive_loans_taken", 0)
    wisdom_score -= 0.3 * informal_loans
    wisdom_score -= 0.15 * festive_loans
    emergencies_survived = history.get("emergencies_survived", 0)
    emergencies_failed = history.get("emergencies_failed", 0)
    total_emergencies = emergencies_survived + emergencies_failed
    if total_emergencies > 0:
        wisdom_score *= (emergencies_survived / total_emergencies)
    wisdom_score = _clamp(wisdom_score)

    total = (
        0.15 * overdraft_score
        + 0.20 * bill_score
        + 0.20 * interest_score
        + 0.15 * savings_score
        + 0.15 * credit_score
        + 0.15 * wisdom_score
    )
    return round(_clamp(total), 4)


def grade_hard(history: Dict[str, Any], final: Dict[str, Any]) -> float:
    """
    HARD grader — 90 days, job loss, emergencies, predatory lenders, Diwali.

    Weights:
      10% — Overdraft avoidance
      15% — No defaults (missed_payments < 3 for all debts)
      15% — Credit score > 650
      15% — Bills paid on time
      10% — Savings rebuilt
      10% — Interest minimization
      15% — Crisis management (emergencies survived, predatory loans avoided)
      10% — Temptation resistance (festival loans avoided)
    """
    ep_len = 90

    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 1.2)

    defaults = history.get("defaults", 0)
    default_score = 1.0 if defaults == 0 else _clamp(1.0 - defaults * 0.15)

    if final["credit_score"] >= 700:
        credit_score = 1.0
    elif final["credit_score"] >= 650:
        credit_score = 0.6 + 0.4 * ((final["credit_score"] - 650) / 50.0)
    else:
        credit_score = _clamp((final["credit_score"] - 300) / 350.0)

    total_bills = max(1, history["total_bills_generated"])
    bill_score = _clamp(history["on_time_payment_count"] / total_bills)

    # Savings rebuilt: any savings is good, ₹20,000+ is great
    savings_score = _clamp(final["savings"] / 20000.0)

    # Interest: worst-case ~90 days on ₹7,30,000 blended ~30% APR ≈ ₹54,000
    worst_case = 54000.0
    interest_ratio = history["total_interest_paid"] / max(1.0, worst_case)
    interest_score = _clamp(1.0 - interest_ratio)

    crisis_score = 1.0
    informal_loans = history.get("informal_loans_taken", 0)
    crisis_score -= 0.25 * informal_loans
    emergencies_survived = history.get("emergencies_survived", 0)
    emergencies_failed = history.get("emergencies_failed", 0)
    total_emergencies = emergencies_survived + emergencies_failed
    if total_emergencies > 0:
        crisis_score *= (emergencies_survived / total_emergencies)
    crisis_score = _clamp(crisis_score)

    festive_loans = history.get("festive_loans_taken", 0)
    temptation_score = _clamp(1.0 - festive_loans * 0.4)

    total = (
        0.10 * overdraft_score
        + 0.15 * default_score
        + 0.15 * credit_score
        + 0.15 * bill_score
        + 0.10 * savings_score
        + 0.10 * interest_score
        + 0.15 * crisis_score
        + 0.10 * temptation_score
    )
    return round(_clamp(total), 4)


# Grader dispatch
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_episode(
    task_id: str, history: Dict[str, Any], final_state: Dict[str, Any]
) -> float:
    """
    Grade a completed episode. Returns score in [0.0, 1.0].

    Args:
        task_id: One of 'easy', 'medium', 'hard'
        history: Episode history dict from the environment
        final_state: Final state snapshot {checking, savings, credit_score, debts}

    Returns:
        Deterministic score between 0.0 and 1.0
    """
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task '{task_id}'")
    return GRADERS[task_id](history, final_state)
