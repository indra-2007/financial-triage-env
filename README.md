---
title: Financial Triage Environment
emoji: 💰
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Financial Triage
### *Can an LLM survive 90 days of India’s household finance?*

**OpenEnv · SFT + GRPO · Qwen2.5-7B · Hugging Face Space**

[![Space](https://img.shields.io/badge/🤗%20Live%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-core%20≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

> **One line:** We train an LLM **day by day** in a realistic INR simulator—bills, UPI leaks, predatory loans, emergencies—using **dense rewards** and **heuristic → SFT → GRPO** so the model learns *policies*, not trivia.

---

## Start here (≈5 minutes for judges)

| # | Read this | You get |
|---|-----------|--------|
| 1 | [**The loop**](#the-loop--how-one-episode-runs) — flowchart + diagram | What `reset` / `step` / grader do |
| 2 | [**Themes**](#hackathon-themes) | Why this fits **Theme #2** + **#3.1** |
| 3 | [**Results**](#training--evidence) — loss + bars | Proof of **SFT + GRPO** vs **heuristic** |
| 4 | [**Quick links**](#links--materials) | Space, Colab, mini-blog, code entry points |

---

## Links & materials

| Resource | URL / file |
|----------|------------|
| **Live Space (submit this URL)** | [**`huggingface.co/spaces/indra-dhanush/financial-triage-env`**](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) |
| **Colab training** (Unsloth + TRL) | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| **Mini write-up** | [`MINI_BLOG.md`](MINI_BLOG.md) · [View on Hub](https://huggingface.co/spaces/indra-dhanush/financial-triage-env/blob/main/MINI_BLOG.md) |
| **Demo video (≤2 min)** | *Add your YouTube URL when published* — do not upload large video files to the Space. |

**Stack:** `Environment` → **verifiable step reward + episode grader** → **SFT** (heuristic traces) → **GRPO** (TRL, live env) → **Unsloth** 4-bit **Qwen2.5-7B** → **this Space**.

---

## The loop — how one episode runs

The environment is **Gym-style**: `reset(task_id)` → repeat `step(action)` until `done` → **grade in [0, 1]**. **One step = one simulated day.** The LLM picks **exactly one** financial action per day from **11** types (pay bill, pay debt, savings move, formal/informal loan, negotiate, …).

### Visual — full OpenEnv loop

<p align="center">
  <img src="flowchart.png" alt="Financial Triage: reset to observation, LLM picks one of eleven actions, environment advances one day with salary UPI interest and shocks, loop until episode ends then grader scores zero to one" width="100%" />
</p>

<p align="center"><em><strong>Caption:</strong> Each day the agent sees balances, bills, debts, risk, optional loan offers, emergencies, and a short text summary → one action → world updates (salary, UPI micro-spend, interest, fees, shocks) → step reward + next observation, or final score.</em></p>

<details>
<summary><strong>Text diagram (same loop, for accessibility / Git copy)</strong></summary>

```mermaid
flowchart LR
    A([reset(task_id)]) --> B[Observation]
    B --> C[LLM: 1 of 11 actions]
    C --> D[env.step — 1 day]
    D --> E{Episode done?}
    E -->|no| B
    E -->|yes| F[Grader: score 0.0–1.0]
```

*(Mermaid renders on GitHub; on some Hub views use the PNG above.)*

</details>

**What runs under the hood:** `server/my_env_environment.py` implements dynamics; `tasks.py` defines **easy / medium / hard** and **`grade_episode`**. See [`openenv.yaml`](openenv.yaml) for task ids.

---

## Hackathon themes

| Theme | How Financial Triage fits |
|-------|---------------------------|
| **#2 Long-horizon planning** | **30 / 60 / 90** simulated days; compounding bills, interest, and shocks—not a single chat turn. |
| **#3.1 World modeling (professional / economic)** | **Partial observability** (e.g. advertised vs realized loan cost), **stochastic** UPI, **programmatic** credit and default rules. |

Not **#1 multi-agent** or **#4 self-generated tasks**—one agent, one economic world, **objective** rewards.

---

## Judging rubric — where to look

| Weight | Criterion | In this repo |
|--------|-----------|--------------|
| 40% | **Innovation** | India-specific finance, loan-shark trap, UPI drain, 14-term reward + anti-hacking |
| 30% | **Story** | This README + [`MINI_BLOG.md`](MINI_BLOG.md) |
| 20% | **Improvement** | [`training_loss_7b.png`](#training--evidence) + [`before_after_scores_7b.png`](#training--evidence) (heuristic vs SFT vs GRPO) |
| 10% | **Reward + pipeline** | Reward table below; SFT + **GRPO on live env** in notebook |

**Checklist:** OpenEnv (`openenv-core[core] ≥ 0.2.3`) · Colab notebook · plots in repo · Space URL · write-up linked.

---

## Why this problem

Household debt and UPI-first money are real constraints in India. Dashboards show history; **we train a model to act** under **stochastic** pay, **mandatory** shocks, and **misleading** credit offers—so success is **verified in code**, not vibes.

---

## Task tiers (curriculum)

| | **Easy (30d)** | **Medium (60d)** | **Hard (90d)** |
|---|----------------|-------------------|----------------|
| **Income** | ₹30K / mo | ₹55K + freelance | Job loss → gig pay |
| **Debt** | None | CC, EMI, BNPL, traps | Large multi-debt |
| **Stress** | UPI noise | Shark + medical | Two emergencies + Diwali |
| **`task_id`** | `easy` | `medium` | `hard` |

---

## Agent: observation & actions (short)

- **Sees:** Checking/savings, bills, debts, risk signals, loan offers, emergency, festival info, **`daily_summary`** text.  
- **Acts (one per day):** e.g. `pay_bill_full`, `pay_minimum`, `defer_bill`, `pay_extra_debt`, `transfer_to_savings`, `withdraw_emergency`, `take_formal_loan`, `take_informal_loan`, `take_festive_loan`, `negotiate_bill`, `do_nothing`.  
- **Scores:** Dense **step** reward + episode **grader** in **[0, 1]**.

---

## Training & evidence

| Phase | What | Model (reported) |
|-------|------|------------------|
| **SFT** | Imitation on heuristic rollouts (`sft_dataset.jsonl`) | **Qwen2.5-7B** 4-bit (Unsloth) |
| **GRPO** | TRL + rollouts in the **live** environment | Same |

*Optional:* 3B in notebook for small GPUs; **figures below are 7B.**

### SFT loss (7B)

<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs step for Qwen2.5-7B" width="720" /></p>

*Loss vs training step—supervised phase converges before GRPO.*

### Heuristic vs SFT (7B) vs GRPO (7B)

<p align="center"><img src="before_after_scores_7b.png" alt="Bar chart: heuristic SFT and GRPO episode scores on Easy Medium Hard" width="720" /></p>

*Orange = **heuristic** (hand-coded baseline). Blue = SFT. Green = GRPO. Values printed on bars.*

### Heuristic-only context (optional)

<p align="center">
  <img src="reward_curves_baseline.png" alt="Heuristic cumulative reward vs day" width="48%" />
  <img src="baseline_scores.png" alt="Heuristic episode scores by difficulty" width="48%" />
</p>

*Reference curves for the rule-based agent—not a substitute for trained-model bars above.*

---

## Dense reward (multi-term)

| ✓ Good signals | ✗ Bad signals |
|----------------|---------------|
| On-time pay, buffer, paying predatory APR, diversity | Late pay, overdraft, default, interest drag, idle streaks, predatory carry |

**Code:** `_compute_reward` in `server/my_env_environment.py` · **`grade_*`** in `tasks.py`.

---

## Quick start

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker:** `docker build -t financial-triage .` · `docker run -p 7860:7860 financial-triage` · health: `GET /health`

**Remote client:**

```python
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

**Training** uses a **local clone** in Colab for speed; **inference** for judges can use the **Space** client above.

---

## OpenEnv manifest & engineering notes

- **Manifest:** [`openenv.yaml`](openenv.yaml) — `app: server.app:app`, port **7860**.  
- **Do not** use reserved tool names: `reset`, `step`, `state`, `close` for custom MCP/HTTP tools.  
- **QLoRA / save:** follow Unsloth merge instructions; test inference after export.

**License:** [MIT](LICENSE) · **OpenEnv:** [hub](https://huggingface.co/openenv) · **PyPI:** [openenv-core](https://pypi.org/project/openenv-core/)

*Optional dev scripts:* `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py` — not required to run the Space.

**Data citations (public stats only):** RBI, NSSO, NPCI, NCRB—narrative calibration, no private data.

---

<div align="center">

**Built for the OpenEnv hackathon — train in the real loop, show the curves, ship the Space.**

</div>
