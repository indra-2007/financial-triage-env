# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and deterministic graders for the Financial Triage Environment.

Three difficulty levels:
  - EASY  (30 days): Stable income, few bills, no debt. Goal: avoid overdraft.
  - MEDIUM (60 days): Multiple debts, irregular income. Goal: minimize interest.
  - HARD  (90 days): Job loss mid-episode, multiple creditors. Goal: survive.

Each grader returns a float score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# Task Configuration Dataclass
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
    }


# =============================================================================
# EASY TASK — 30 days
# =============================================================================

EASY_TASK = _make_task(
    task_id="easy",
    episode_length=30,
    checking=3000.0,
    savings=500.0,
    credit_score=720,
    salary_schedule=[
        {"day": 0, "amount": 2500.0, "label": "Bi-weekly salary"},
        {"day": 14, "amount": 2500.0, "label": "Bi-weekly salary"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 1200.0, "base_due_day": 5, "category": "housing"},
        {"id": "utilities", "amount": 150.0, "base_due_day": 10, "category": "utilities"},
        {"id": "phone", "amount": 80.0, "base_due_day": 20, "category": "phone"},
    ],
    debt_templates=[],
    expense_events=[],
    bill_changes=[],
    total_credit_limit=0.0,
    description=(
        "Stable bi-weekly income of $2,500. Three simple bills (rent, utilities, phone). "
        "No debt. Goal: pay all bills on time and avoid overdraft."
    ),
)


# =============================================================================
# MEDIUM TASK — 60 days
# =============================================================================

MEDIUM_TASK = _make_task(
    task_id="medium",
    episode_length=60,
    checking=2000.0,
    savings=300.0,
    credit_score=660,
    salary_schedule=[
        {"day": 0, "amount": 2200.0, "label": "Monthly salary"},
        {"day": 18, "amount": 800.0, "label": "Freelance payment"},
        {"day": 30, "amount": 2200.0, "label": "Monthly salary"},
        {"day": 45, "amount": 800.0, "label": "Freelance payment"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 1400.0, "base_due_day": 3, "category": "housing"},
        {"id": "utilities", "amount": 200.0, "base_due_day": 8, "category": "utilities"},
        {"id": "phone", "amount": 90.0, "base_due_day": 15, "category": "phone"},
        {"id": "insurance", "amount": 350.0, "base_due_day": 20, "category": "insurance"},
        {"id": "streaming", "amount": 50.0, "base_due_day": 25, "category": "subscription"},
    ],
    debt_templates=[
        {
            "id": "cc_visa",
            "principal": 4500.0,
            "apr": 22.0,
            "minimum_due": 90.0,
            "credit_limit": 10000.0,
            "min_payment_due_day": 12,
            "is_credit_card": True,
        },
        {
            "id": "personal_loan",
            "principal": 8000.0,
            "apr": 12.0,
            "minimum_due": 200.0,
            "credit_limit": None,
            "min_payment_due_day": 18,
            "is_credit_card": False,
        },
        {
            "id": "medical_debt",
            "principal": 2000.0,
            "apr": 0.0,
            "minimum_due": 100.0,
            "credit_limit": None,
            "min_payment_due_day": 22,
            "is_credit_card": False,
        },
    ],
    expense_events=[],
    bill_changes=[],
    total_credit_limit=10000.0,
    description=(
        "Monthly salary $2,200 plus irregular $800 freelance. Five recurring bills. "
        "Three debts (credit card 22% APR, personal loan 12%, medical 0%). "
        "Goal: minimize interest paid and avoid missed payments."
    ),
)


# =============================================================================
# HARD TASK — 90 days
# =============================================================================

HARD_TASK = _make_task(
    task_id="hard",
    episode_length=90,
    checking=1500.0,
    savings=200.0,
    credit_score=640,
    salary_schedule=[
        # Full employment: days 0 and 30
        {"day": 0, "amount": 2800.0, "label": "Monthly salary"},
        # Job loss at day 30 — unemployment benefits only
        {"day": 30, "amount": 1200.0, "label": "Unemployment benefit"},
        {"day": 60, "amount": 1200.0, "label": "Unemployment benefit"},
    ],
    bill_templates=[
        {"id": "rent", "amount": 1500.0, "base_due_day": 3, "category": "housing"},
        {"id": "utilities", "amount": 250.0, "base_due_day": 8, "category": "utilities"},
        {"id": "phone", "amount": 100.0, "base_due_day": 15, "category": "phone"},
        {"id": "insurance", "amount": 400.0, "base_due_day": 20, "category": "insurance"},
        {"id": "car_payment", "amount": 350.0, "base_due_day": 22, "category": "transportation"},
        {"id": "streaming", "amount": 30.0, "base_due_day": 25, "category": "subscription"},
    ],
    debt_templates=[
        {
            "id": "cc_platinum",
            "principal": 6000.0,
            "apr": 24.0,
            "minimum_due": 120.0,
            "credit_limit": 10000.0,
            "min_payment_due_day": 10,
            "is_credit_card": True,
        },
        {
            "id": "cc_rewards",
            "principal": 3000.0,
            "apr": 19.0,
            "minimum_due": 60.0,
            "credit_limit": 5000.0,
            "min_payment_due_day": 14,
            "is_credit_card": True,
        },
        {
            "id": "personal_loan",
            "principal": 12000.0,
            "apr": 14.0,
            "minimum_due": 300.0,
            "credit_limit": None,
            "min_payment_due_day": 18,
            "is_credit_card": False,
        },
        {
            "id": "medical_debt",
            "principal": 5000.0,
            "apr": 8.0,
            "minimum_due": 150.0,
            "credit_limit": None,
            "min_payment_due_day": 24,
            "is_credit_card": False,
        },
    ],
    expense_events=[
        {"day": 15, "amount": 800.0, "description": "Emergency car repair"},
        {"day": 45, "amount": 600.0, "description": "Medical co-pay"},
        {"day": 70, "amount": 400.0, "description": "Home appliance replacement"},
    ],
    bill_changes=[
        # Rent increases starting cycle 2 (day 60+)
        {"bill_id": "rent", "from_day": 60, "new_amount": 1700.0},
    ],
    total_credit_limit=15000.0,
    description=(
        "Income drops from $2,800 to $1,200 (unemployment) at day 30. "
        "Six bills including rent increase at day 60. Four debts totaling $26,000. "
        "Emergency expenses on days 15, 45, 70. "
        "Goal: avoid default, maintain credit score > 650, rebuild savings."
    ),
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

    # Overdraft avoidance: 1.0 if zero overdraft days, degrades linearly
    # Softer penalty to allow partial success
    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 2.0)

    # Bill payment: ratio of on-time vs total generated
    total_bills = max(1, history["total_bills_generated"])
    on_time_ratio = history["on_time_payment_count"] / total_bills
    bill_score = _clamp(on_time_ratio)

    # Savings: reward for maintaining/growing from initial
    initial_savings = max(1.0, history["initial_savings"])
    savings_ratio = final["savings"] / initial_savings
    savings_score = _clamp(savings_ratio / 1.5)  # Full marks at 1.5x initial

    # Credit score: penalize drops from initial
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
    MEDIUM grader — 60 days, debts, irregular income.

    Weights:
      20% — Overdraft avoidance
      25% — Bills paid on time
      25% — Interest minimization
      15% — Savings
      15% — Credit score
    """
    ep_len = 60

    # Softer overdraft penalty
    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 2.5)

    total_bills = max(1, history["total_bills_generated"])
    bill_score = _clamp(history["on_time_payment_count"] / total_bills)

    # Interest minimization: compare against worst-case (all debts untouched)
    # Worst-case ~60 days of interest on $14,500 at blended ~13% APR ≈ $310
    worst_case_interest = 310.0
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

    total = (
        0.20 * overdraft_score
        + 0.25 * bill_score
        + 0.25 * interest_score
        + 0.15 * savings_score
        + 0.15 * credit_score
    )
    return round(_clamp(total), 4)


def grade_hard(history: Dict[str, Any], final: Dict[str, Any]) -> float:
    """
    HARD grader — 90 days, job loss, emergencies.

    Weights:
      15% — Overdraft avoidance
      20% — No defaults (missed_payments < 3 for all debts)
      20% — Credit score > 650
      20% — Bills paid on time
      15% — Savings rebuilt
      10% — Interest minimization
    """
    ep_len = 90

    # Softer penalty for overdrafts to allow partial success
    overdraft_score = _clamp(1.0 - (history["overdraft_days"] / ep_len) * 1.5)

    # Default avoidance
    defaults = history.get("defaults", 0)
    # Reduced penalty per default to avoid tanking the score completely
    default_score = 1.0 if defaults == 0 else _clamp(1.0 - defaults * 0.15)

    # Credit score target: > 650
    if final["credit_score"] >= 700:
        credit_score = 1.0
    elif final["credit_score"] >= 650:
        credit_score = 0.6 + 0.4 * ((final["credit_score"] - 650) / 50.0)
    else:
        credit_score = _clamp((final["credit_score"] - 300) / 350.0)

    total_bills = max(1, history["total_bills_generated"])
    bill_score = _clamp(history["on_time_payment_count"] / total_bills)

    # Savings rebuilt: any savings is good, $500+ is great
    savings_score = _clamp(final["savings"] / 500.0)

    # Interest: worst-case ~90 days on $26,000 blended ~17% APR ≈ $1,090
    worst_case = 1090.0
    interest_ratio = history["total_interest_paid"] / max(1.0, worst_case)
    interest_score = _clamp(1.0 - interest_ratio)

    total = (
        0.15 * overdraft_score
        + 0.20 * default_score
        + 0.20 * credit_score
        + 0.20 * bill_score
        + 0.15 * savings_score
        + 0.10 * interest_score
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
