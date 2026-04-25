---
title: Financial Triage Environment
emoji: 💰
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏥💰 Financial Triage: Can AI Survive 90 Days of India's Financial Crisis?

> *"73% of Indians fail basic financial literacy. We built an RL environment where an AI agent learns what most Indians were never taught — which bill to pay, which to default on, and when the 'instant cash' loan is a death trap."*

**Live Environment**: [HuggingFace Space](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) · **Training Notebook**: [`financial_triage_training.ipynb`](financial_triage_training.ipynb)

---

## Why This Matters

India's household debt hit **45.5% of GDP**. Credit card outstanding surged to **₹2.7 lakh crore** (2024). 27% of microfinance borrowers take new loans to repay old ones. NCRB recorded 171,418 suicides in 2023 — 66% of victims earned ≤₹1L/year.

Fintech apps show pie charts. **We train AI to make decisions.**

---

## How It Works

![Environment Loop — reset → observe → act → step → grade](flowchart.png)

**1 step = 1 day.** The agent sees its full financial state and picks ONE action. The environment simulates salary deposits, UPI micro-spends, interest accrual, bill deadlines, medical shocks, and Diwali pressure — then returns the next day's state + a 14-component reward signal.

---

## Three Difficulty Tiers

| | Easy (30d) | Medium (60d) | Hard (90d) |
|--|-----------|-------------|-----------|
| **Income** | ₹30K/month | ₹55K + ₹20K freelance | ₹65K → job loss → ₹28K gig |
| **Debts** | None | HDFC CC, SBI EMI, BNPL | 4 debts, ₹7.3L total |
| **Events** | UPI drain | 🦈 Loan shark, 🚨 ₹45K medical | 🦈 365% APR, 🚨 2 emergencies, 🎉 Diwali |
| **Bills** | ±15% stochastic | ±15% stochastic | ±15% stochastic |

---

## Key Features

### 📱 UPI Micro-Transaction Drain
₹100–₹800/day stochastic spending (Swiggy, Zomato, chai, auto). Compounds to ₹12K/month — 40% of a fresh grad's salary.

### 🦈 Loan Shark Trap (Partial Observability)
Moneylender shows **36.5% APR** in the observation. Actual rate: **365%.** Agent must learn from consequences, not labels.

### 🚨 Medical Emergency Shock
₹35K–₹50K mandatory bills with 3–5 day deadlines. Calibrated to NSSO hospitalization data.

### 🎉 Diwali Social Pressure
7-day window: ₹1,500/day auto-deducted for gifts + social obligations. Festive loans at 28% APR.

### 🛡️ Anti-Reward-Hacking
6 protections: savings churn detection, minimum transfer thresholds, inaction penalties, predatory loan penalties, action diversity bonus, emergency buffer bonus.

---

## 📈 Results & Reward Curves

### Baseline Reward Curves (Heuristic Agent)

Easy accumulates steadily. Hard crashes at day 30 (job loss) and day 72 (second emergency).

![Heuristic baseline reward curves across Easy, Medium, and Hard tasks](reward_curves_baseline.png)

### Baseline Episode Scores

Heuristic-only scores. A trained LLM should beat these.

![Heuristic baseline scores: Easy=0.999, Medium=0.694, Hard=0.427](baseline_scores.png)

### Training Pipeline

| Phase | What | Time (T4) |
|-------|------|-----------|
| **SFT Warmup** | 180 expert examples from heuristic → teach valid actions | ~20 min |
| **GRPO** | RL against live environment → actions scored in real env | ~60 min |
| **Model** | `Qwen2.5-3B-Instruct` (4-bit). Change 1 line → 7B. | Free Colab |

---

## 🚀 Quick Start

```bash
# Run locally
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e . && uvicorn server.app:app --port 7860

# Or via Docker
docker build -t financial-triage . && docker run -p 7860:7860 financial-triage
```

```python
# Python client
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

---

## 📊 Dense Reward (14 Components)

```
POSITIVE                            NEGATIVE
+5.0   No overdraft today           -15.0  Per late payment
+10.0  Bill paid on time            -25.0  Overdraft detected
+var   High-APR debt payment        -50.0  Per default (3 missed)
+6.0   Savings grew ≥₹500          -8.0   Carrying moneylender debt
+3.0   Emergency buffer maintained  -2.0n  Consecutive inaction (n≥3)
+1.0   Action diversity bonus       -5.0   Zero savings
+0.5x  CIBIL improvement            -2.0x  Interest accrued
```

---

## 📜 License & Links

MIT License · [OpenEnv Hackathon](https://huggingface.co/openenv) · [openenv-core](https://pypi.org/project/openenv-core/)

**Data Sources**: RBI Financial Literacy Survey (2024), NSSO Health Expenditure, TransUnion CIBIL, NPCI UPI statistics, NCRB Suicide Data (2023), Asian Development Bank
