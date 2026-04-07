---
title: Financial Triage Environment
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Personal Financial Triage & Life Budget Advisor

🔗 **Live Environment**
https://huggingface.co/spaces/indra-dhanush/financial-triage-env

A production-grade OpenEnv environment simulating real-world personal financial decision-making.
An AI agent operates inside a household financial system and must make daily decisions to balance liquidity, debt, and long-term stability.

---

## 🧠 Why this matters

Financial decision-making affects every human. Poor decisions compound over time through interest, penalties, and missed obligations.

This environment models:

* real bank balances
* upcoming bills
* debt with APR
* income uncertainty
* credit score dynamics

Unlike static benchmarks, this environment captures long-term financial consequences, making it ideal for evaluating real-world AI agents.

---

## 🧠 Agent Challenge

**Can an AI survive 90 days of financial stress without going bankrupt?**

---

## ⚙️ Environment Overview

* **1 step = 1 day**

### Episodes:

* Easy → 30 days
* Medium → 60 days
* Hard → 90 days

The agent must continuously decide how to allocate limited resources under uncertainty.

---

## 👁 Observation Space

At every step, the agent sees:

### Account State

* checking_balance
* savings_balance
* credit_utilization
* next_salary_day

### Bills

* amount
* due date
* category
* payment status

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

## 🎯 Action Space

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

## 🔥 What makes this environment hard

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

## 💰 Reward Design Philosophy

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

👉 **reward_breakdown**

This ensures transparency and interpretability.

---

## 🧪 Tasks

### 🟢 Easy — Stability

* Stable income
* Few bills
* No debt
* Goal: avoid overdraft

---

### 🟡 Medium — Debt Management

* Multiple debts with different APR
* Irregular income

Goal:

* minimize interest
* avoid missed payments

---

### 🔴 Hard — Survival Mode

* Job loss mid-episode
* Reduced income
* Multiple creditors

Goal:

* avoid default
* maintain credit score > 650
* rebuild savings

---

## 📊 Evaluation Metrics

Agents are scored using smooth metrics:

* Overdraft avoidance
* Interest minimization
* Credit score stability
* Savings growth

Scores range from **0.0 → 1.0** and reward partial success.

---

## 📊 Baseline Performance

Heuristic agent scores:

* Easy: ~0.93
* Medium: ~0.56
* Hard: ~0.40

This demonstrates increasing difficulty and clear headroom for advanced agents.

---

## 🔌 API Usage (OpenEnv)

### Reset environment

POST /reset

Example:
{
"task_id": "hard"
}

---

### Step environment

POST /step

Example:
{
"action": {
"action_type": "pay_minimum",
"debt_id": "cc_platinum"
}
}

---

### Get current state

GET /state

---

## 🚀 Setup & Running the Environment

### Install dependencies

```bash
pip install -r requirements.txt
```

### Validate OpenEnv compliance

```bash
openenv validate
```

### Run inference (heuristic baseline)

```bash
python inference.py
```

### Run inference (with LLM)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-huggingface-token"
python inference.py
```

### Docker

```bash
docker build -t financial-triage-env .
docker run -p 7860:7860 financial-triage-env
```

The server will be available at `http://localhost:7860`.

---

## ⚡ Key Features

* deterministic randomness (seeded)
* realistic financial dynamics
* dense reward shaping
* interpretable reward breakdown
* fast simulation (<1s per episode)

---

## 🧠 Why this environment is novel

* Not a toy problem
* Not classification or QA
* Full sequential decision-making system
* Dense reward RL environment
* Direct real-world applicability

This environment can directly power real financial assistant systems.

---

## 🏁 Summary

This environment provides a realistic, high-signal benchmark for evaluating AI agents in long-horizon financial decision-making.

It combines:

* real-world utility
* technical depth
* RL-friendly design

making it suitable for both research and deployment.
