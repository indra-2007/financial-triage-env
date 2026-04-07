"""
Inference Script — Personal Financial Triage & Life Budget Advisor
===================================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import sys
import json
import textwrap
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Add project root to path so we can import the environment directly
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FinancialAction, FinancialObservation, ActionType
from server.my_env_environment import MyEnvironment

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
MAX_STEPS = 100
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "do_nothing"

# ---------------------------------------------------------------------------
# Valid action types the LLM can choose from
# ---------------------------------------------------------------------------
VALID_ACTIONS = [a.value for a in ActionType]

# ---------------------------------------------------------------------------
# System prompt for the LLM financial advisor
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI financial advisor managing a person's finances day by day.
Each day you must choose EXACTLY ONE action from the following list:

ACTIONS:
- do_nothing                          → Take no financial action today
- pay_bill_full(bill_id)              → Pay a bill in full from checking
- pay_minimum(debt_id)                → Make minimum payment on a debt
- defer_bill(bill_id)                 → Explicitly defer a bill
- pay_extra_debt(debt_id, amount)     → Pay extra toward a debt
- transfer_to_savings(amount)         → Move money to savings
- withdraw_emergency(amount)          → Move money from savings to checking

RULES:
1. You can only pick ONE action per day.
2. Prioritize paying bills BEFORE their due date to avoid late fees.
3. Pay high-APR debt minimums to avoid default (3 missed = default).
4. Keep enough in checking to cover upcoming bills.
5. Build savings when possible.

RESPONSE FORMAT:
Reply with ONLY the action string on one line. Examples:
  do_nothing
  pay_bill_full(rent)
  pay_minimum(cc_visa)
  pay_extra_debt(cc_visa, 200.00)
  transfer_to_savings(100.00)
  withdraw_emergency(300.00)

Do NOT include any explanation, just the action string.
""").strip()


def observation_to_prompt(obs: FinancialObservation) -> str:
    """Convert a FinancialObservation into a readable text prompt for the LLM."""
    lines = [
        f"=== Day {obs.current_day} of {obs.episode_length} ===",
        f"Checking: ${obs.account.checking_balance:.2f}",
        f"Savings:  ${obs.account.savings_balance:.2f}",
        f"Credit Score: {obs.risk.credit_score}",
        f"Net Worth: ${obs.net_worth:.2f}",
        f"Next Salary: Day {obs.account.next_salary_day}",
        "",
    ]

    # Bills
    unpaid_bills = [b for b in obs.bills if not b.is_paid]
    paid_bills = [b for b in obs.bills if b.is_paid]
    if unpaid_bills:
        lines.append("UNPAID BILLS:")
        for b in sorted(unpaid_bills, key=lambda x: x.due_day):
            urgency = ""
            days_left = b.due_day - obs.current_day
            if days_left <= 0:
                urgency = " ⚠️ OVERDUE"
            elif days_left <= 2:
                urgency = " ⚠️ DUE SOON"
            lines.append(
                f"  - {b.id}: ${b.amount:.2f} due day {b.due_day} "
                f"({b.category}){urgency}"
            )
    if paid_bills:
        lines.append(f"PAID BILLS: {', '.join(b.id for b in paid_bills)}")

    # Debts
    if obs.debts:
        lines.append("")
        lines.append("OUTSTANDING DEBTS:")
        for d in sorted(obs.debts, key=lambda x: -x.apr):
            warning = ""
            if d.missed_payments >= 2:
                warning = " 🚨 NEAR DEFAULT"
            elif d.missed_payments >= 1:
                warning = " ⚠️ MISSED PAYMENT"
            lines.append(
                f"  - {d.id}: ${d.principal:.2f} @ {d.apr}% APR, "
                f"min ${d.minimum_due:.2f}, missed={d.missed_payments}{warning}"
            )

    # Risk
    lines.append("")
    lines.append("RISK SIGNALS:")
    lines.append(f"  Interest today: ${obs.risk.interest_today:.2f}")
    lines.append(f"  Late fee risk:  ${obs.risk.late_fee_risk:.2f}")
    if obs.risk.days_to_overdraft >= 0:
        lines.append(f"  Days to overdraft: {obs.risk.days_to_overdraft}")

    # Today's summary
    if obs.daily_summary and obs.daily_summary != "No events":
        lines.append("")
        lines.append(f"TODAY: {obs.daily_summary}")

    lines.append("")
    lines.append("What is your ONE action for today?")
    return "\n".join(lines)


def parse_action(response_text: str, obs: FinancialObservation) -> FinancialAction:
    """
    Parse an LLM response string into a FinancialAction.
    Falls back to a smart heuristic if parsing fails.
    """
    text = response_text.strip()

    # Try each line for a parseable action
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove common prefixes like "Action:" or "- "
        line = re.sub(r"^(action|next action|response)\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
        line = line.strip().strip("`").strip('"').strip("'")

        action = _try_parse_action_string(line, obs)
        if action is not None:
            return action

    # Fallback: use heuristic agent
    return _heuristic_action(obs)


def _try_parse_action_string(s: str, obs: FinancialObservation) -> Optional[FinancialAction]:
    """Attempt to parse a single action string like 'pay_bill_full(rent)'."""
    s = s.strip()

    # Simple no-arg actions
    if s in ("do_nothing", "do_nothing()"):
        return FinancialAction(action_type=ActionType.DO_NOTHING)

    # Match function-call style: action_name(arg1, arg2)
    m = re.match(r"(\w+)\s*\(\s*(.*?)\s*\)", s)
    if not m:
        # Try bare action name
        if s in VALID_ACTIONS:
            if s == "do_nothing":
                return FinancialAction(action_type=ActionType.DO_NOTHING)
        return None

    action_name = m.group(1)
    args_str = m.group(2)

    # Clean args: remove quotes
    args = [a.strip().strip("'\"") for a in args_str.split(",") if a.strip()]

    try:
        if action_name == "pay_bill_full" and len(args) >= 1:
            return FinancialAction(action_type=ActionType.PAY_BILL_FULL, bill_id=args[0])

        if action_name == "pay_minimum" and len(args) >= 1:
            return FinancialAction(action_type=ActionType.PAY_MINIMUM, debt_id=args[0])

        if action_name == "defer_bill" and len(args) >= 1:
            return FinancialAction(action_type=ActionType.DEFER_BILL, bill_id=args[0])

        if action_name == "pay_extra_debt" and len(args) >= 2:
            amount = float(args[1])
            return FinancialAction(
                action_type=ActionType.PAY_EXTRA_DEBT,
                debt_id=args[0],
                amount=amount,
            )

        if action_name == "transfer_to_savings" and len(args) >= 1:
            amount = float(args[0])
            return FinancialAction(
                action_type=ActionType.TRANSFER_TO_SAVINGS,
                amount=amount,
            )

        if action_name == "withdraw_emergency" and len(args) >= 1:
            amount = float(args[0])
            return FinancialAction(
                action_type=ActionType.WITHDRAW_EMERGENCY,
                amount=amount,
            )

        if action_name == "do_nothing":
            return FinancialAction(action_type=ActionType.DO_NOTHING)

    except (ValueError, TypeError):
        return None

    return None


def _heuristic_action(obs: FinancialObservation) -> FinancialAction:
    """
    Fallback heuristic agent — used when LLM output can't be parsed.
    Priority: pay urgent bills → pay debt mins → do nothing.
    """
    # Priority 1: Pay bills due within 2 days
    for bill in sorted(obs.bills, key=lambda b: b.due_day):
        if not bill.is_paid and bill.due_day <= obs.current_day + 2:
            if obs.account.checking_balance >= bill.amount + 100:
                return FinancialAction(
                    action_type=ActionType.PAY_BILL_FULL, bill_id=bill.id
                )

    # Priority 2: Pay minimum on debts near default
    for debt in sorted(obs.debts, key=lambda d: -d.missed_payments):
        if debt.missed_payments >= 1 and debt.principal > 0:
            if obs.account.checking_balance > debt.minimum_due + 200:
                return FinancialAction(
                    action_type=ActionType.PAY_MINIMUM, debt_id=debt.id
                )

    # Priority 3: Pay bills due within 5 days
    for bill in sorted(obs.bills, key=lambda b: b.due_day):
        if not bill.is_paid and bill.due_day <= obs.current_day + 5:
            if obs.account.checking_balance >= bill.amount + 300:
                return FinancialAction(
                    action_type=ActionType.PAY_BILL_FULL, bill_id=bill.id
                )

    # Priority 4: Pay minimums on highest-APR debts
    for debt in sorted(obs.debts, key=lambda d: -d.apr):
        if debt.principal > 0:
            if obs.account.checking_balance > debt.minimum_due + 500:
                return FinancialAction(
                    action_type=ActionType.PAY_MINIMUM, debt_id=debt.id
                )

    # Priority 5: Emergency withdrawal if checking very low
    if obs.account.checking_balance < 200 and obs.account.savings_balance > 100:
        return FinancialAction(
            action_type=ActionType.WITHDRAW_EMERGENCY,
            amount=min(300.0, obs.account.savings_balance),
        )

    return FinancialAction(action_type=ActionType.DO_NOTHING)


def run_episode(
    env: MyEnvironment,
    client: OpenAI,
    task_id: str,
    use_llm: bool = True,
) -> float:
    """Run a single episode and return the grader score."""
    obs = env.reset(seed=42, task_id=task_id)
    total_reward = 0.0
    step = 0

    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()} ({obs.episode_length} days)")
    print(f"  Checking: ${obs.account.checking_balance:.2f}")
    print(f"  Savings:  ${obs.account.savings_balance:.2f}")
    print(f"  Credit:   {obs.risk.credit_score}")
    if obs.debts:
        print(f"  Debts:    {len(obs.debts)} totaling ${sum(d.principal for d in obs.debts):.2f}")
    print(f"{'='*60}")

    while not obs.done and step < MAX_STEPS:
        action = None

        if use_llm:
            user_prompt = observation_to_prompt(obs)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_action(response_text, obs)
                action_str = f"{action.action_type.value}"
                if action.bill_id:
                    action_str += f"({action.bill_id})"
                elif action.debt_id:
                    action_str += f"({action.debt_id}"
                    if action.amount:
                        action_str += f", {action.amount}"
                    action_str += ")"
                elif action.amount:
                    action_str += f"({action.amount})"
                print(f"  Day {obs.current_day:3d}: LLM → {action_str}")
            except Exception as exc:
                print(f"  Day {obs.current_day:3d}: LLM error ({exc}), using heuristic")
                action = _heuristic_action(obs)
        else:
            action = _heuristic_action(obs)

        obs = env.step(action)
        total_reward += obs.reward
        step += 1

    score = env.get_episode_score()
    print(f"\n  --- Results ---")
    print(f"  Steps:        {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Credit Score: {obs.risk.credit_score}")
    print(f"  Checking:     ${obs.account.checking_balance:.2f}")
    print(f"  Savings:      ${obs.account.savings_balance:.2f}")
    print(f"  Net Worth:    ${obs.net_worth:.2f}")
    print(f"  GRADER SCORE: {score:.4f}")

    return score


def main() -> None:
    """Run inference on all 3 tasks and print results."""
    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Check if LLM is available
    use_llm = True
    if API_KEY in ("dummy", "", None):
        print("⚠️  No API key found. Running with heuristic agent (no LLM).")
        use_llm = False

    # Initialize environment
    env = MyEnvironment()

    # Run all 3 tasks
    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task_id in tasks:
        score = run_episode(env, client, task_id, use_llm=use_llm)
        scores[task_id] = score

    # Print final summary
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        status = "✅" if score > 0.5 else "⚠️" if score > 0.2 else "❌"
        print(f"  {task_id:8s}: {score:.4f} {status}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':8s}: {avg:.4f}")
    print(f"{'='*60}")

    # Output JSON for automated scoring
    print(f"\nscores_json={json.dumps(scores)}")


if __name__ == "__main__":
    main()
