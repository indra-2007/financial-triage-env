# 🏥💰 Financial Triage: Can AI Survive 90 Days of India's Financial Crisis?

> *"We built an RL environment trained on India's most widespread crisis — 73% of the population financially illiterate, facing salary delays and debt traps daily. Our agent learns what most Indians were never taught."*

---

## The Problem That Kills

**73% of Indian adults fail the RBI's minimum financial literacy threshold.**

India's household debt has reached 45.5% of GDP — and it's not building wealth. It's funding consumption. Personal loans, credit cards, and BNPL schemes now account for **54.9% of total household debt**, consuming 25.7% of disposable income. Credit card outstanding surged from ₹87,686 crore (2019) to **₹2,70,000 crore** (2024) — a 17% real CAGR.

The cost isn't just economic — it's mortal. The NCRB recorded 171,418 suicides in India in 2023. **66% of victims earned ₹1 lakh or less annually.** Financial insecurity, bankruptcy, and indebtedness are direct causes.

27% of India's microfinance borrowers take new loans just to repay old ones — the exact debt trap this environment simulates. India's Gen Z accounts for 43% of consumer spending while growing up without basic money management skills.

Current fintech apps — CRED, Jupiter, Fi Money — show beautiful pie charts. They gamify your credit score. They alert you when a bill is due.

**But they don't make decisions.**

When you have ₹15,000 in your account, rent of ₹12,000, a HDFC credit card minimum of ₹7,500, and a SBI EMI of ₹8,000 — a push notification saying you're "over budget" is **utterly useless**. You need to know: *which bill do I default on to minimize long-term damage to my CIBIL score?*

That is not a prediction problem. That is a **sequential decision-making problem under uncertainty over time.** The only mathematical framework designed to solve it is **Reinforcement Learning**.

---

## What This Environment Does

Financial Triage is an OpenEnv-compliant RL environment that simulates **real-world Indian personal finance survival** — calibrated against actual RBI, NSSO, and NPCI data. All amounts are in **Indian Rupees (₹)**.

The agent manages a household's finances **one day at a time**, choosing from **11 action types** across a multi-day episode:

| Action | What It Does |
|--------|-------------|
| `pay_bill_full(bill_id)` | Pay a bill in full from checking |
| `pay_minimum(debt_id)` | Make minimum payment on a debt |
| `pay_extra_debt(debt_id, amount)` | Accelerate debt payoff (avalanche strategy) |
| `transfer_to_savings(amount)` | Build emergency buffer |
| `withdraw_emergency(amount)` | Liquidate savings when cash-strapped |
| `take_formal_loan(amount)` | Apply for SBI/HDFC bank loan (14-18% APR, delayed) |
| `take_informal_loan(amount)` | ⚠️ Take instant cash from local moneylender (240-365% APR) |
| `take_festive_loan(amount)` | 🎉 Diwali loan ("low EMI" but 28% APR) |
| `negotiate_bill(bill_id)` | Attempt to negotiate a bill reduction (40% chance) |
| `defer_bill(bill_id)` | Explicitly defer a bill payment |
| `do_nothing` | Take no action today |

### Three Difficulty Levels

| Level | Duration | Income | Debts | Special Events |
|-------|----------|--------|-------|----------------|
| **Easy** | 30 days | ₹30,000/month via UPI | None | UPI micro-spends |
| **Medium** | 60 days | ₹55,000 + ₹20,000 freelance | HDFC CC (42%), SBI EMI, Flipkart BNPL | 🦈 Loan shark, 🚨 Medical ₹75K |
| **Hard** | 90 days | Job loss at day 30 → ₹28K gig income | 4 debts (₹7.3L total) | 🦈 365% APR, 🚨 2 emergencies (₹50K + ₹35K), 🎉 Diwali |

---

## 📱 UPI Micro-Transaction Simulation

India processes **14+ billion UPI transactions per month** (NPCI, 2024). The #1 silent killer of Indian household budgets isn't big expenses — it's the ₹200 Swiggy order, the ₹50 chai, the ₹150 auto-rickshaw, the ₹499 Blinkit order, multiplied by 30 days.

Our environment simulates **real UPI spending patterns**:

| Feature | What It Does |
|---------|-------------|
| **Daily micro-spends** | ₹100-₹800/day on Swiggy, Zomato, chai, auto, Blinkit (stochastic) |
| **UPI P2P pressure** | Friends/family request UPI transfers at random intervals |
| **Salary as UPI credit** | Income arrives as UPI credit on configured days |
| **UPI autopay drain** | OTT subscriptions auto-deduct via UPI mandate |

The agent must learn: *these ₹300/day micro-transactions compound to ₹9,000/month* — nearly 30% of a fresh graduate's salary. Controlling UPI drain is a survival skill.

---

## 🦈 The Loan Shark Trap

Private moneylenders across India charge **1-10% daily interest** — that's 365-3,650% APR. When the agent hits a cash crisis, two loan options appear:

| Option | Label | APR | Delivery | CIBIL Check |
|--------|-------|-----|----------|-------------|
| **SBI Loan** | "SBI Personal Loan — 14% APR" | 14% | 3 days | Requires score ≥ 650 |
| **Moneylender** | "Instant cash — no paperwork" | **365%** | Instant | None |

An untrained model reads "instant" and takes the moneylender. A **trained** agent waits 3 days for the bank loan. This maps directly to the decision 27% of Indian microfinance borrowers face when they take new loans to repay old ones.

## 🚨 Medical Emergency Shock

NSSO data shows mean out-of-pocket cost per private hospitalization is **₹1,69,504**. Even a median case drains months of salary.

The environment injects a mandatory medical bill (₹35,000–₹50,000) with a **5-day deadline**. If unpaid:
- CIBIL score drops 30 points
- 20% penalty fee
- Emergency marked as failed

The agent learns: **maintaining a savings buffer isn't optional — it's a matter of survival.**

## 🎉 Diwali Season Pressure

India's Diwali alone generates **₹85,000 crore** in consumer spending. Banks flood consumers with "0% EMI festive offers" that compound for years.

During the Diwali window, the environment:
1. Auto-deducts ₹1,500/day for social obligations (gifts, sweets, clothes)
2. Broadcasts WhatsApp-style social pressure messages
3. Makes festive loans temporarily available ("Diwali Dhamaka Loan" at 28% APR)

The agent must learn to **pre-save before Diwali** instead of borrowing during it.

---

## 🛡️ Anti-Reward-Hacking Protections

| Protection | What It Prevents |
|------------|-----------------|
| **Minimum transfer threshold** (₹500) | Farming savings_growth reward with ₹1 transfers |
| **Savings churn detection** | Withdraw-and-redeposit same day for free reward |
| **Consecutive inaction penalty** | Doing nothing 3+ days to avoid risk |
| **Predatory loan penalty** | Ongoing -8.0 per step while carrying moneylender debt |
| **Action diversity bonus** | Rewards using 3+ different action types in 7 days |
| **Emergency buffer bonus** | Rewards maintaining liquidity above upcoming obligations |

---

## 📊 Dense Reward System — 14 Components

```
Positive Signals                    Negative Signals
─────────────────                   ─────────────────
+5.0   No overdraft                 -15.0  Per late payment
+10.0  Per bill paid on time        -25.0  Overdraft
+var   High-APR debt payment        -2.0x  Interest accrued
+6.0   Savings growth (>₹500)       -50.0  Per default
+0.5x  CIBIL score improvement      -5.0   Zero savings
+3.0   Emergency buffer maintained  -8.0   Carrying moneylender debt
+1.0   Action diversity             -2.0n  Consecutive inaction
```

---

## 🚀 Quick Start

### Run Locally
```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env
pip install -e ".[dev]"
uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
```

### Run via Docker
```bash
docker build -t financial-triage .
docker run -p 7860:7860 financial-triage
```

### Use from OpenEnv Client
```python
from openenv import EnvClient

env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
print(obs.account.checking_balance)  # 25000.0 (INR)
print(obs.loan_offers)              # SBI bank vs local moneylender
```

---

## 🏗️ Architecture

```
financial-triage-env/
├── openenv.yaml                 # OpenEnv manifest (3 tasks)
├── models.py                    # Pydantic models: 11 actions, rich observations
├── tasks.py                     # Task configs (INR) + deterministic graders
├── inference.py                 # LLM agent + UPI-aware heuristic fallback
├── server/
│   ├── app.py                   # FastAPI server (port 7860)
│   └── my_env_environment.py    # Core simulation engine (~1200 lines)
├── Dockerfile                   # Production container
└── pyproject.toml               # Dependencies (openenv-core ≥0.2.3)
```

### Grading System

| Task | Criteria | Weight |
|------|----------|--------|
| **Easy** | Overdraft avoidance | 30% |
| | Bill payments on time | 35% |
| | Savings growth | 20% |
| | CIBIL maintained | 15% |
| **Medium** | +Interest minimization | 20% |
| | +Financial wisdom (avoided predatory debt) | 15% |
| **Hard** | +Crisis management (emergencies survived) | 15% |
| | +Temptation resistance (Diwali loans avoided) | 10% |

---

## 🎯 Why This Matters for India

This environment trains AI agents to do what **no existing Indian fintech app does**: make the mathematically optimal financial decision **every single day**, across a multi-step horizon of competing obligations.

An agent trained on Financial Triage learns:
- **UPI micro-leak control** — ₹300/day on food delivery compounds to ₹9,000/month
- **Avalanche debt payoff** — pay HDFC CC (42% APR) before SBI loan (14% APR)
- **Predatory lending recognition** — "instant cash" at 365% APR is never worth it
- **Medical buffer maintenance** — keep ₹25,000+ liquid for the ₹50K shock
- **Diwali resistance** — pre-save ₹10,000 before the festival instead of borrowing ₹50,000 after
- **BNPL trap avoidance** — Flipkart/Amazon Pay Later's 0% converts to 36% when overdue

> *For 51% of Indians struggling to meet their debts and liabilities — far exceeding the global average of 32% — AI-driven financial triage isn't a technological curiosity. It's a mathematical imperative for survival.*

---

## 📜 License

MIT License. Built for the [OpenEnv Hackathon](https://huggingface.co/openenv).

## 🔗 Links

- **Live Environment**: [HuggingFace Space](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
- **OpenEnv Framework**: [openenv-core](https://pypi.org/project/openenv-core/)

## 📊 Data Sources

All financial parameters are calibrated against real Indian data:
- **RBI Financial Literacy Survey** (2024) — 27% adult literacy rate
- **NSSO Health Expenditure** — ₹1,69,504 mean OOP hospitalization cost
- **TransUnion CIBIL** — Credit card delinquency at 15% (90+ DPD)
- **NPCI** — 14B+ UPI transactions/month, merchant category distributions
- **NCRB Suicide Data** (2023) — 66% of victims earned ≤₹1L/year
- **Asian Development Bank** — 24% female financial literacy rate
