# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Personal Financial Triage & Life Budget Advisor Environment.

Defines structured Pydantic models for observations (full financial state),
actions (11 financial decision types), and supporting data structures for
accounts, bills, debts, loans, emergencies, festivals, and risk signals.

New in v2 (Round 2):
  - Loan Shark Trap: formal vs informal loan actions
  - Medical Emergency Shock: sudden mandatory expenses
  - Festival Season Pressure: social spending + festive loan temptation
  - Bill Negotiation: probabilistic bill reduction
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

    # --- Original actions ---
    PAY_BILL_FULL = "pay_bill_full"
    PAY_MINIMUM = "pay_minimum"
    DEFER_BILL = "defer_bill"
    PAY_EXTRA_DEBT = "pay_extra_debt"
    TRANSFER_TO_SAVINGS = "transfer_to_savings"
    WITHDRAW_EMERGENCY = "withdraw_emergency"
    DO_NOTHING = "do_nothing"

    # --- New actions (Round 2) ---
    TAKE_FORMAL_LOAN = "take_formal_loan"
    TAKE_INFORMAL_LOAN = "take_informal_loan"
    TAKE_FESTIVE_LOAN = "take_festive_loan"
    NEGOTIATE_BILL = "negotiate_bill"


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
    apr: float = Field(..., ge=0, le=500, description="Annual percentage rate")
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
# New v2 Models — Loans, Emergencies, Festivals
# =============================================================================


class LoanOffer(BaseModel):
    """A loan offer available to the agent — may be predatory or legitimate."""

    loan_type: str = Field(..., description="'formal_bank', 'informal_lender', or 'festive'")
    label: str = Field(..., description="Human-readable label (may be misleading for informal)")
    max_amount: float = Field(..., ge=0, description="Maximum borrowable amount")
    apr: float = Field(..., ge=0, description="Annual percentage rate (can be extreme for informal)")
    processing_days: int = Field(default=0, ge=0, description="Days until funds arrive (0=instant)")
    min_credit_score: int = Field(default=0, ge=0, description="Minimum credit score required")
    available: bool = Field(default=True, description="Whether agent qualifies for this loan now")
    warning: str = Field(default="", description="Risk signal (honest for formal, misleading for informal)")


class MedicalEmergencyInfo(BaseModel):
    """An active medical emergency requiring urgent payment."""

    id: str = Field(..., description="Emergency identifier")
    amount: float = Field(..., ge=0, description="Total cost of the emergency")
    deadline_day: int = Field(..., ge=0, description="Absolute day by which payment must be made")
    description: str = Field(default="", description="Human-readable description")
    is_paid: bool = Field(default=False, description="Whether the emergency has been paid")


class FestivalInfo(BaseModel):
    """Active festival season information with social pressure signals."""

    name: str = Field(..., description="Festival name")
    days_remaining: int = Field(..., ge=0, description="Days remaining in festival window")
    daily_social_cost: float = Field(default=0.0, ge=0, description="Daily mandatory social spending")
    festive_loan_available: bool = Field(default=False, description="Whether festive loan action is available")
    festive_loan_apr: float = Field(default=0.0, ge=0, description="APR of the festive loan offer")
    pressure_message: str = Field(default="", description="Social pressure narrative")


# =============================================================================
# Action Model
# =============================================================================


class FinancialAction(Action):
    """
    Agent action for the financial triage environment.

    Each step, the agent selects one action type with optional parameters.
    Validation ensures required parameters are present for each action type.

    v2 adds: take_formal_loan, take_informal_loan, take_festive_loan, negotiate_bill
    """

    action_type: ActionType = Field(..., description="Type of financial action")
    bill_id: Optional[str] = Field(default=None, description="Target bill ID")
    debt_id: Optional[str] = Field(default=None, description="Target debt ID")
    amount: Optional[float] = Field(default=None, ge=0.01, description="Dollar amount")

    @model_validator(mode="after")
    def validate_action_params(self) -> "FinancialAction":
        """Ensure required parameters are provided for each action type."""
        at = self.action_type
        if at in (ActionType.PAY_BILL_FULL, ActionType.DEFER_BILL, ActionType.NEGOTIATE_BILL):
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
        if at in (ActionType.TAKE_FORMAL_LOAN, ActionType.TAKE_INFORMAL_LOAN, ActionType.TAKE_FESTIVE_LOAN):
            if self.amount is None or self.amount <= 0:
                raise ValueError(f"{at.value} requires a positive loan amount")
        return self


# =============================================================================
# Observation Model
# =============================================================================


class FinancialObservation(Observation):
    """
    Rich financial state observation returned every step.

    Contains full visibility into accounts, bills, debts, risk signals,
    loan offers, medical emergencies, festival pressure, and a
    human-readable daily summary for agent reasoning.
    """

    account: AccountInfo = Field(..., description="Account balances and dates")
    bills: List[BillInfo] = Field(default_factory=list, description="Active bills")
    debts: List[DebtInfo] = Field(default_factory=list, description="Outstanding debts")
    risk: RiskSignals = Field(..., description="Risk assessment signals")
    current_day: int = Field(..., ge=0, description="Current simulation day")
    episode_length: int = Field(..., gt=0, description="Total episode length in days")
    net_worth: float = Field(default=0.0, description="checking + savings - total_debt")
    daily_summary: str = Field(default="", description="Human-readable events today")

    # --- New v2 fields (all have defaults for backward compatibility) ---
    loan_offers: List[LoanOffer] = Field(default_factory=list, description="Available loan options")
    active_emergency: Optional[MedicalEmergencyInfo] = Field(
        default=None, description="Active medical emergency requiring payment"
    )
    festival: Optional[FestivalInfo] = Field(
        default=None, description="Active festival season information"
    )


# =============================================================================
# Backward-compatible aliases (used by app.py, client.py, __init__.py)
# =============================================================================

MyAction = FinancialAction
MyObservation = FinancialObservation
