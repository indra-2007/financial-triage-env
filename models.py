# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Personal Financial Triage & Life Budget Advisor Environment.

Defines structured Pydantic models for observations (full financial state),
actions (7 financial decision types), and supporting data structures for
accounts, bills, debts, and risk signals.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from openenv.core.env_server.types import Action, Observation


# =============================================================================
# Enums
# =============================================================================


class ActionType(str, Enum):
    """All available financial actions the agent can take each day."""

    PAY_BILL_FULL = "pay_bill_full"
    PAY_MINIMUM = "pay_minimum"
    DEFER_BILL = "defer_bill"
    PAY_EXTRA_DEBT = "pay_extra_debt"
    TRANSFER_TO_SAVINGS = "transfer_to_savings"
    WITHDRAW_EMERGENCY = "withdraw_emergency"
    DO_NOTHING = "do_nothing"


class BillCategory(str, Enum):
    """Bill categories for financial tracking."""

    HOUSING = "housing"
    UTILITIES = "utilities"
    PHONE = "phone"
    INSURANCE = "insurance"
    TRANSPORTATION = "transportation"
    SUBSCRIPTION = "subscription"
    MEDICAL = "medical"
    OTHER = "other"


# =============================================================================
# Nested Data Models (used inside Observation)
# =============================================================================


class BillInfo(BaseModel):
    """A recurring bill with due date and payment status."""

    id: str = Field(..., description="Unique bill identifier")
    amount: float = Field(..., ge=0, description="Bill amount in dollars")
    due_day: int = Field(..., ge=0, description="Absolute day the bill is due")
    category: str = Field(..., description="Bill category")
    is_paid: bool = Field(default=False, description="Whether bill is paid this cycle")


class DebtInfo(BaseModel):
    """An outstanding debt with interest accrual tracking."""

    id: str = Field(..., description="Unique debt identifier")
    principal: float = Field(..., ge=0, description="Remaining principal balance")
    apr: float = Field(..., ge=0, le=100, description="Annual percentage rate")
    minimum_due: float = Field(..., ge=0, description="Minimum monthly payment")
    missed_payments: int = Field(default=0, ge=0, description="Consecutive missed cycles")


class AccountInfo(BaseModel):
    """Current account balances and income schedule."""

    checking_balance: float = Field(..., description="Checking account balance (can be negative)")
    savings_balance: float = Field(..., ge=0, description="Savings account balance")
    credit_utilization: float = Field(..., ge=0, le=1.0, description="Credit utilization ratio (0-1)")
    next_salary_day: int = Field(..., ge=0, description="Next day salary will be deposited")


class RiskSignals(BaseModel):
    """Risk assessment metrics computed each day."""

    days_to_overdraft: int = Field(..., ge=-1, description="Estimated days to overdraft (-1=safe)")
    interest_today: float = Field(..., ge=0, description="Interest accrued today across all debts")
    late_fee_risk: float = Field(..., ge=0, description="Total late fees at risk in next 5 days")
    credit_score: int = Field(..., ge=300, le=850, description="Current FICO-like credit score")


# =============================================================================
# Action Model
# =============================================================================


class FinancialAction(Action):
    """
    Agent action for the financial triage environment.

    Each step, the agent selects one action type with optional parameters.
    Validation ensures required parameters are present for each action type.
    """

    action_type: ActionType = Field(..., description="Type of financial action")
    bill_id: Optional[str] = Field(default=None, description="Target bill ID")
    debt_id: Optional[str] = Field(default=None, description="Target debt ID")
    amount: Optional[float] = Field(default=None, ge=0.01, description="Dollar amount")

    @model_validator(mode="after")
    def validate_action_params(self) -> "FinancialAction":
        """Ensure required parameters are provided for each action type."""
        at = self.action_type
        if at in (ActionType.PAY_BILL_FULL, ActionType.DEFER_BILL):
            if self.bill_id is None:
                raise ValueError(f"{at.value} requires bill_id")
        if at in (ActionType.PAY_MINIMUM, ActionType.PAY_EXTRA_DEBT):
            if self.debt_id is None:
                raise ValueError(f"{at.value} requires debt_id")
        if at == ActionType.PAY_EXTRA_DEBT:
            if self.amount is None or self.amount <= 0:
                raise ValueError("pay_extra_debt requires a positive amount")
        if at in (ActionType.TRANSFER_TO_SAVINGS, ActionType.WITHDRAW_EMERGENCY):
            if self.amount is None or self.amount <= 0:
                raise ValueError(f"{at.value} requires a positive amount")
        return self


# =============================================================================
# Observation Model
# =============================================================================


class FinancialObservation(Observation):
    """
    Rich financial state observation returned every step.

    Contains full visibility into accounts, bills, debts, risk signals,
    and a human-readable daily summary for agent reasoning.
    """

    account: AccountInfo = Field(..., description="Account balances and dates")
    bills: List[BillInfo] = Field(default_factory=list, description="Active bills")
    debts: List[DebtInfo] = Field(default_factory=list, description="Outstanding debts")
    risk: RiskSignals = Field(..., description="Risk assessment signals")
    current_day: int = Field(..., ge=0, description="Current simulation day")
    episode_length: int = Field(..., gt=0, description="Total episode length in days")
    net_worth: float = Field(default=0.0, description="checking + savings - total_debt")
    daily_summary: str = Field(default="", description="Human-readable events today")


# =============================================================================
# Backward-compatible aliases (used by app.py, client.py, __init__.py)
# =============================================================================

MyAction = FinancialAction
MyObservation = FinancialObservation
