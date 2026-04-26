---
title: Financial Triage Environment
emoji: 💰
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div style="font-size:1.08rem;line-height:1.55;">

<div align="center">

# Financial Triage
<h3 style="font-size:1.35em;font-weight:600;margin:0.4em 0 0.2em;">Long-horizon household finance in INR, under OpenEnv—numbers over slogans.</h3>
<p style="font-size:1.05em;margin:0.5em 0 1em;"><strong>OpenEnv · SFT + GRPO · Qwen2.5-7B · Hugging Face Space</strong></p>

[![Space](https://img.shields.io/badge/🤗%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/openenv--core-≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

**Financial Triage** is a Gym-style simulator: one `step` equals one day, one financial action, dense per-step returns, and a terminal grader in \([0,1]\). The point is not to *describe* a budget—it is to **allocate** under shaky income, UPI drag, and credit products that are simple to name on a slide and painful to get wrong in code. We warm-start from a hand-built heuristic, then **SFT** and **GRPO** (TRL) on **Qwen2.5-7B** in 4-bit with Unsloth, and ship the environment on this Space. Stack in one line: **environment → verifier-style step reward and episode grader → SFT on expert rollouts → GRPO on live rollouts** with **`openenv-core[core] ≥ 0.2.3`**.

## How to use this page

The order matches how you would *review* the work: what the machine does each day, how that lines up with the stated hackathon tracks, where the official rubric points in the repository, then the actual loss and bar charts, and finally how to run the Space yourself. If you are in a hurry, follow the first column in the table; if not, the sections read straight through without jumping.

| You are looking for… | Open this section |
|----------------------|-------------------|
| The control loop and diagram | [What happens each day](#what-happens-each-day-reset-step-grade) |
| Long-horizon and “world model” story | [Problem themes and scope](#problem-themes-and-scope) |
| Where innovation / results / pipeline show up in the files | [Rubric: what lives where](#rubric-what-lives-where) |
| SFT loss and heuristic vs. trained bars | [Training and the published figures](#training-and-the-published-figures) |
| Space, Colab, short write-up | [Links: demo, training, docs](#links-demo-training-docs) |

## Links: demo, training, docs

| What | Where |
|------|--------|
| Running Space | [huggingface.co/spaces/indra-dhanush/financial-triage-env](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) |
| Colab (Unsloth + TRL) | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) — notebook file: [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| Companion text | [`MINI_BLOG.md`](MINI_BLOG.md) — [same file on the Hub](https://huggingface.co/spaces/indra-dhanush/financial-triage-env/blob/main/MINI_BLOG.md) |
| Short video (optional) | Add a public URL when you have one; keep video off the repo bundle to keep the Space small. |

## What happens each day: reset, step, grade

The contract is the usual OpenEnv one: **`reset(task_id)`**, then **`step(action)`** until the episode ends, then a **single scalar** in \([0,1]\) from the grader. Each transition advances **one calendar day**; the policy picks **one** of **eleven** actions (bills, minimums, deferral, extra principal, savings in/out, three loan types, bill negotiation, or do nothing). Nothing here pretends the user is chatting with a bank—the **only** thing that matters is whether the sim’s books stay coherent under the rules you see in the code.

<p align="center">
  <img src="flowchart.png" alt="OpenEnv loop: reset, observation, one of eleven actions, one-day step, loop or grader" width="100%" style="max-width:1100px;" />
</p>

*Each turn: you observe balances, instruments, risk, and a short text summary, you act once, the world applies paydays, UPI noise, interest, fees, and shock events, you get a step reward, and you either continue or get a final grade.*

<details>
<summary><strong>Mermaid (for GitHub and viewers that support it)</strong></summary>

```mermaid
flowchart LR
    A([reset(task_id)]) --> B[Observation]
    B --> C[One of 11 actions]
    C --> D[env.step: one day]
    D --> E{Done?}
    E -->|no| B
    E -->|yes| F[Grader 0.0–1.0]
```

</details>

**Code paths:** `server/my_env_environment.py` holds the dynamics, `tasks.py` holds task definitions and `grade_episode`, [`openenv.yaml`](openenv.yaml) lists `easy` / `medium` / `hard`.

## Problem themes and scope

| Hackathon track | How this build fits it |
|------------------|------------------------|
| **#2 — Long-horizon planning** | You get **30-, 60-, and 90-day** rollouts. Bills and interest do not “reset” after one model turn; the grader is looking at a whole arc, not a single yes/no answer. |
| **#3.1 — Professional / economic world modeling** | Cash flow is **stochastic**; some loan terms are **misleading** in the observation vs. the ledger; default and overdraft are **mechanical** outcomes, not a vibe check. |

**Explicitly not in v1:** multi-agent games (**#1**) or a self-generating curriculum (**#4**). It is one agent, one economy, one scoring function you can re-run.

## Rubric: what lives where

| Weight | Criterion | Where to see it here |
|--------|-----------|----------------------|
| 40% | Innovation | INR household setting, 11 actions, UPI and informal-credit structure, 14-term-style dense reward, anti-gaming design |
| 30% | Clarity of story | This file plus [`MINI_BLOG.md`](MINI_BLOG.md) |
| 20% | Proof of training | `training_loss_7b.png` and `before_after_scores_7b.png` in the repo root (heuristic vs. SFT vs. GRPO) |
| 10% | Coherent reward + pipeline | Reward section below; notebook shows SFT then GRPO **against the live** `step` function |

**Surface area for a reviewer:** the Space (runnable), the notebook (reproducible training), the PNGs (not only a Colab session), and the short write-up linked above.

## Why a simulator, not a slide

Reading about “financial stress” is cheap; **missing a payment** in this environment costs measurable score. The public statistics we cite in prose (RBI, NSSO, NPCI, NCRB) are there to ground the *story*; every rupee *inside* the sim is still **fake, rule-driven, and inspectable**—if you need to argue with a number, you open the Python.

## Three tasks: easy, medium, hard

| | **Easy (30d)** | **Medium (60d)** | **Hard (90d)** |
|---|----------------|------------------|----------------|
| **Income** | Steady salary | Salary + side income | **Job loss, then gig** |
| **Debt** | None | Card, EMI, BNPL, predatory option | **Heavy multi-debt** |
| **Shocks** | UPI noise | Health + informality | **Stacked** crises + festival |
| **`task_id`** | `easy` | `medium` | `hard` |

## State, action, return (API-level)

**Observation** packs checking and savings, open bills, debts, risk readouts, optional loan table, any active emergency, festival state, and a **`daily_summary`**. **Action** is a single choice from the eleven templates each day. **Return** is a dense **step** signal plus, at the end, **`grade_episode`** in \([0,1]\) from `tasks.py`. That split is intentional: the model gets feedback *during* the month, and a clean scalar *after* it.

## Training and the published figures

| Stage | What runs | Model we report in the charts |
|-------|-----------|-------------------------------|
| **SFT** | Imitation on rollouts from the **hand-written** heuristic, see `sft_dataset.jsonl` | **Qwen2.5-7B** 4-bit, Unsloth |
| **GRPO** | TRL; reward comes from real **`step`** outcomes, not a relabeled static file | same backbone |

On a small GPU you can point the notebook at **3B**; the **PNGs in this commit are from the 7B run** unless you replace them.

### SFT loss (7B)
<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs step for 7B" width="720" /></p>
*Supervised loss vs. optimizer step before GRPO.*

### Heuristic · SFT (7B) · GRPO (7B)
<p align="center"><img src="before_after_scores_7b.png" alt="Heuristic, SFT, GRPO mean episode score by difficulty" width="720" /></p>
*Orange: heuristic / rule baseline. Blue: SFT. Green: GRPO. Values are printed on the chart.*

### Heuristic-only diagnostics (for context)
<p align="center">
  <img src="reward_curves_baseline.png" alt="Heuristic reward vs day" width="48%" />
  <img src="baseline_scores.png" alt="Heuristic bar scores" width="48%" />
</p>
*These are **not** the finetuned model’s learning curve; they are background on the reference policy only.*

## How step reward is shaped

| Encouraged | Discouraged |
|------------|-------------|
| Staying liquid, on-time pay, paying down predatory exposure, healthy buffers, non-trivial action mix | Delinquency, overdraft, default chains, interest bleed, inaction, carrying exploitative instruments |

**Implementation:** `_compute_reward` in `server/my_env_environment.py`, grading in `grade_*` / `grade_episode` in `tasks.py`.

## Run the Space locally or from Python

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

`docker build -t financial-triage .` then `docker run -p 7860:7860 financial-triage` — health check: `GET /health`.

**Remote client**

```python
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

Colab often trains against a **local clone** for speed; the **authoritative** remote entry point is still the Space URL in the first table. If you need a purist **client–server** line, point the client at that URL.

## Manifest, license, exports

- [`openenv.yaml`](openenv.yaml): `server.app:app`, port **7860**, task ids as above.  
- Do not register **custom** tools named `reset`, `step`, `state`, or `close` on top of OpenEnv.  
- After QLoRA / merge, follow Unsloth’s **save** recipe and re-run a short eval; bad merges are silent until inference.

[MIT](LICENSE) · [OpenEnv hub](https://huggingface.co/openenv) · [`openenv-core` on PyPI](https://pypi.org/project/openenv-core/). Public Indian macro and payment stats are cited for **narrative** only; the simulation does not ingest private account data. Optional local helpers: `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py` (not on the server path).

</div>
