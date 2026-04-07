# Personal Financial Triage & Life Budget Advisor

A production-grade OpenEnv environment simulating real-world personal financial decision-making.
An AI agent operates inside a household financial system and must make daily decisions to balance liquidity, debt, and long-term stability.

---

# 🧠 Why this matters

Financial decision-making affects every human. Poor decisions compound over time through interest, penalties, and missed obligations.

This environment models:

* real bank balances
* upcoming bills
* debt with APR
* income uncertainty
* credit score dynamics

Unlike static benchmarks, this environment captures **long-term financial consequences**, making it ideal for evaluating real-world AI agents.

---

# ⚙️ Environment Overview

* **1 step = 1 day**
* Episodes:

  * Easy → 30 days
  * Medium → 60 days
  * Hard → 90 days

The agent must continuously decide how to allocate limited resources under uncertainty.

---

# 👁 Observation Space

At every step, the agent sees:

### Account State

* checking_balance
* savings_balance
* credit_utilization
* next_salary_day

### Bills

* amount, due date, category, payment status

### Debt Stack

* principal
* APR
* minimum due
* missed payments

### Risk Signals

* days_to_overdraft
* interest_today
* late_fee_risk
* credit_score

### Time

* current_day

---

# 🎯 Action Space

The agent can take actions such as:

* pay_bill_full(bill_id)
* pay_minimum(debt_id)
* defer_bill(bill_id)
* pay_extra_debt(debt_id, amount)
* transfer_to_savings(amount)
* withdraw_emergency(amount)
* do_nothing()

All actions are validated and affect future financial states.

---

# 🔥 What makes this environment hard

### ⏳ Delayed consequences

A decision made early in the episode can have cascading effects weeks later.

### ⚖️ Competing priorities

The agent must balance:

* rent vs debt
* savings vs liquidity
* short-term survival vs long-term optimization

### 🎲 Uncertainty

* salary may be delayed or vary
* unexpected expenses occur
* financial plans cannot be deterministic

---

# 💰 Reward Design Philosophy

The environment provides **dense daily rewards**.

### Positive signals:

* avoiding overdraft
* paying bills on time
* reducing high-APR debt
* growing savings
* improving credit score

### Negative signals:

* overdraft penalties
* late payments
* interest accumulation
* missed obligations

Each step includes a detailed:

```json
reward_breakdown
```

This ensures transparency and interpretability.

---

# 🧪 Tasks

### 🟢 Easy — Stability

* Stable income
* Few bills
* No debt
* Goal: avoid overdraft

---

### 🟡 Medium — Debt Management

* Multiple debts with different APR
* Irregular income
* Goal:

  * minimize interest
  * avoid missed payments

---

### 🔴 Hard — Survival Mode

* Job loss mid-episode
* Reduced income
* Multiple creditors
* Goal:

  * avoid default
  * maintain credit score > 650
  * rebuild savings

---

# 📊 Evaluation Metrics

Agents are scored using smooth metrics:

* **Overdraft avoidance**
* **Interest minimization**
* **Credit score stability**
* **Savings growth**

Scores range from **0.0 → 1.0** and reward partial success.

---

# 🧠 Why this environment is novel

* Not a toy problem
* Not classification or QA
* Full sequential decision-making system
* Dense reward RL environment
* Direct real-world applicability

This environment can directly power real financial assistant systems.

---

# 🚀 Running the Environment

### Validate:

```bash
openenv validate
```

### Run inference:

```bash
uv run inference.py
```

---

# ⚡ Key Features

* deterministic randomness (seeded)
* realistic financial dynamics
* dense reward shaping
* interpretable reward breakdown
* fast simulation (<1s per episode)

---

# 🏁 Summary

This environment provides a realistic, high-signal benchmark for evaluating AI agents in long-horizon financial decision-making.

It combines:

* real-world utility
* technical depth
* RL-friendly design

making it suitable for both research and deployment.
