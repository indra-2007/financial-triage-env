"""
Inference Script — Personal Financial Triage & Life Budget Advisor
===================================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using
                     from_docker_image() method

- Defaults are set only for API_BASE_URL and MODEL_name
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=financial_triage model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=easy env=financial_triage model=meta-llama/Llama-3.3-70B-Instruct
    [STEP] step=1 action=pay_bill_full(rent) reward=15.00 done=false error=null
    [STEP] step=2 action=do_nothing reward=5.00 done=false error=null
    [END] success=true steps=30 score=0.933 rewards=15.00,5.00,...
"""

import os
import re
import sys
import json
import textwrap
from typing import List, Optional

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
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "financial_triage"
MAX_STEPS = 100
TEMPERATURE = 0.2
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Valid action types the LLM can choose from
# ---------------------------------------------------------------------------
VALID_ACTIONS = [a.value for a in ActionType]

# ---------------------------------------------------------------------------
# Structured stdout logging (exact format required by OpenEnv evaluator)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str] = None
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_action(action: FinancialAction) -> str:
    """Format a FinancialAction into a clean string for logging (no spaces)."""
    action_str = action.action_type.value
    if action.bill_id:
        action_str += f"({action.bill_id})"
    elif action.debt_id:
        action_str += f"({action.debt_id}"
        if action.amount:
            action_str += f",{action.amount:.2f}"
        action_str += ")"
    elif action.amount:
        action_str += f"({action.amount:.2f})"
    return action_str


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
    rewards: List[float] = []
    step = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
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
                except Exception as exc:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)
                    action = _heuristic_action(obs)
            else:
                action = _heuristic_action(obs)

            # Format action string for logging (no spaces)
            action_str = format_action(action)

            # Execute step
            obs = env.step(action)
            step += 1

            # Log step immediately after env.step() returns
            log_step(
                step=step,
                action=action_str,
                reward=obs.reward,
                done=obs.done,
                error=None,
            )

            rewards.append(obs.reward)

        # Compute final grader score
        score = env.get_episode_score()
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Always emit [END], even on exception
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run inference on all 3 tasks and print results."""
    # Initialize OpenAI client
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key="dummy")

    # Check if LLM is available
    use_llm = True
    if not HF_TOKEN:
        print(f"[DEBUG] No API key found. Running with heuristic agent (no LLM).", flush=True)
        use_llm = False

    # Initialize environment
    env = MyEnvironment()

    # Run all 3 tasks
    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task_id in tasks:
        score = run_episode(env, client, task_id, use_llm=use_llm)
        scores[task_id] = score

    # Output JSON for automated scoring (informational)
    print(f"scores_json={json.dumps(scores)}", flush=True)


if __name__ == "__main__":
    main()
