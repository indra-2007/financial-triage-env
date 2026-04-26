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
<h3 style="font-size:1.35em;font-weight:600;margin:0.4em 0 0.2em;">Long-horizon household money in a simulated world—rules you can read in code, not a pitch deck.</h3>
<p style="font-size:1.05em;margin:0.5em 0 1em;"><strong>OpenEnv · Hugging Face Space</strong></p>

[![Space](https://img.shields.io/badge/🤗%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/openenv--core-≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

This project is a day-by-day money simulator. You are not here to get a pretty summary of spending—you are here to make choices with consequences: which bill gets cash first, when to move savings, and when an “easy” credit offer is a trap. Every balance change follows fixed rules, so you can **disagree with the program** by reading it, not by arguing with a chatbot. That is the point of putting the problem in an environment instead of a slide: stress shows up in the ledger, and the ledger is inspectable.

## Why a simulator, not a slide

Reading about “financial stress” is cheap. **Missing a due date in this sim costs a measurable number** in the same place every time: the state vector and the grader, not a footnote. If you cite real-world statistics, they only ground the **story in words**; every rupee **inside** the run is still fake, rule-driven, and open to audit in Python. A slide can comfort you; a simulator can still say no.

## How to use this page

The sections are ordered so you can read straight through, or use the first column to jump. The loop and diagram come first in substance; themes and the rubric map follow; then training figures, reward design, and how to run the Space.

| You are looking for… | Open this section |
|----------------------|-------------------|
| The control loop and diagram | [What happens each day](#what-happens-each-day-reset-step-grade) |
| Long-horizon and “world model” story | [Problem themes and scope](#problem-themes-and-scope) |
| Where the rubric points in the files | [Rubric: what lives where](#rubric-what-lives-where) |
| Training pipeline and the bar chart | [Training and the published figures](#training-and-the-published-figures) |
| How reward and grading work | [How step reward is shaped](#how-step-reward-is-shaped) and [Episode grade](#episode-grade) |
| Space, Colab, short write-up | [Links: demo, training, docs](#links-demo-training-docs) |
| Unsloth, training library, model size, precision | [Stack](#stack) |

## Links: demo, training, docs

| What | Where |
|------|--------|
| Running Space | [huggingface.co/spaces/indra-dhanush/financial-triage-env](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) |
| Colab (see [Stack](#stack) for tools) | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) — notebook file: [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| Companion text | [`MINI_BLOG.md`](MINI_BLOG.md) — [same file on the Hub](https://huggingface.co/spaces/indra-dhanush/financial-triage-env/blob/main/MINI_BLOG.md) |
| Short video (optional) | Add a public URL when you have one; keep video off the repo bundle to keep the Space small. |

## What happens each day: reset, step, grade

The contract is the usual OpenEnv one: **`reset(task_id)`**, then **`step(action)`** until the episode ends, then a **single scalar** from the grader: **a score between 0 and 1** (see [Episode grade](#episode-grade) for what that number is). Each transition is **one calendar day**; the policy picks **one** of **eleven** actions (bills, minimums, deferral, extra principal, savings in and out, three loan types, bill negotiation, or do nothing). The interface does not play at being your bank; the only thing that matters is whether the books stay consistent with the code.

<p align="center">
  <img src="flowchart.png" alt="OpenEnv loop: reset, observation, one of eleven actions, one-day step, loop or grader" width="100%" style="max-width:1100px;" />
</p>

*Each turn: you see balances, instruments, risk, and a short text summary, you act once, the world applies paydays, small daily spends, interest, fees, and shock events, you get a per-step return from the environment, and you either continue or receive a final grade.*

<details>
<summary><strong>Mermaid (for GitHub and viewers that support it)</strong></summary>

```mermaid
flowchart LR
    A([reset(task_id)]) --> B[Observation]
    B --> C[One of 11 actions]
    C --> D[env.step: one day]
    D --> E{Done?}
    E -->|no| B
    E -->|yes| F[Grader: score 0 to 1]
```

</details>

**Code paths:** `server/my_env_environment.py` holds the dynamics, `tasks.py` holds task definitions and `grade_episode`, [`openenv.yaml`](openenv.yaml) lists `easy` / `medium` / `hard`.

## Problem themes and scope

| Hackathon track | How this build fits it |
|------------------|------------------------|
| **#2 — Long-horizon planning** | You get **30-, 60-, and 90-day** rollouts. Bills and interest do not “reset” after one model turn; the grader is looking at a whole arc, not a single yes or no. |
| **#3.1 — Professional / economic world modeling** | Cash flow is **stochastic**; some loan terms are **misleading** in the observation compared to the ledger; default and overdraft are **mechanical** outcomes, not a vibe check. |

**Explicitly not in v1:** multi-agent games (**#1**) or a self-generating curriculum (**#4**). It is one agent, one economy, one scoring function you can re-run.

## Rubric: what lives where

| Weight | Criterion | Where to see it here |
|--------|-----------|----------------------|
| 40% | Innovation | India household setting, eleven actions, informal-credit and small-transaction structure, many additive **step** terms in one scalar plus anti-gaming, episode grader in `tasks.py` |
| 30% | Clarity of story | This file plus [`MINI_BLOG.md`](MINI_BLOG.md) |
| 20% | Proof of training | `training_loss_7b.png` and `before_after_scores_7b.png` in the repo root (heuristic vs. fine-tunes) |
| 10% | Coherent reward + pipeline | [How step reward is shaped](#how-step-reward-is-shaped), [Training and the published figures](#training-and-the-published-figures); notebook runs supervised imitation then policy optimization on the live `step` (see that section for which scalar the trainer uses) |

**Surface area for a reviewer:** the Space (runnable), the notebook (reproducible training), the PNGs (not only a Colab session), and the short write-up linked above.

## Three tasks: easy, medium, hard

| | **Easy (30d)** | **Medium (60d)** | **Hard (90d)** |
|---|----------------|------------------|----------------|
| **Income** | Steady salary | Salary + side income | **Job loss, then gig** |
| **Debt** | None | Card, EMI, buy-now-pay-later, predatory option | **Heavy multi-debt** |
| **Shocks** | Small daily noise | Health + informality | **Stacked** crises + festival |
| **`task_id`** | `easy` | `medium` | `hard` |

## State, action, return (API-level)

**Observation** packs checking and savings, open bills, debts, risk readouts, optional loan table, any active emergency, festival state, and a **`daily_summary`**. **Action** is a single choice from the eleven templates each day. **Return** has two different roles: at each step, the environment emits **one number** for that day, built from the bill of signed components in [How step reward is shaped](#how-step-reward-is-shaped). At the end, **`grade_episode`** yields **a score between 0 and 1** for the whole episode. The model is meant to get dense feedback *during* the month and a clean end scalar *after* it.

## Training and the published figures

| Stage | What runs | Model we report in the charts |
|-------|-----------|-------------------------------|
| **Supervised warm-start** | Imitation on rollouts from a **hand-written** heuristic, see `sft_dataset.jsonl` | **Qwen2.5-7B** 4-bit; see [Stack](#stack) |
| **GRPO** (via Hugging Face TRL) | Training loop that calls the **live** `env.step` with parsed actions from the model | same backbone |

**What scalar the policy-optimization step uses (important):** In `financial_triage_training.ipynb`, the `reward_func` that TRL’s trainer calls does **not** pass through the full dense per-step return from the observation, and it does **not** call **`grade_episode`**. For each generated completion, it runs **one** `env.step` with the parsed action, then maps that transition to a **small hand-written score** in **[-1, 1]**: a base value, a bonus if checking stays non-negative, a bonus if the day summary text indicates on-time pay, a penalty if the text looks like an informal loan. That keeps the online signal cheap to compute in the training loop. **By contrast,** the before / after **bar chart** is built from **full episode** runs that call **`get_episode_score()`**, which uses **`grade_episode`**: a score between 0 and 1 (see [Episode grade](#episode-grade)). So: **policy phase** = a compact proxy on **one** step; **reported bar heights** = **episode grader** after a full run.

On a small GPU you can point the notebook at a smaller public checkpoint; the **PNGs in this commit are from the 7B run** unless you replace them.

### SFT loss (7B)
<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs step for 7B" width="720" /></p>
*Supervised loss vs. optimizer step before policy optimization.*

### Heuristic · supervised · policy-optimized (7B) — **episode** scores
<p align="center"><img src="before_after_scores_7b.png" alt="Heuristic, SFT, GRPO mean episode score by difficulty" width="720" /></p>
*Orange: heuristic / rule baseline. Blue: after supervised fine-tuning. Green: after the TRL run. Values are printed on the chart. **These bars come from a single evaluation pass** (seeds and setup as in the notebook); treat them as **indicative**, not a distribution.*

### Heuristic-only diagnostics (for context)
<p align="center">
  <img src="reward_curves_baseline.png" alt="Heuristic reward vs day" width="48%" />
  <img src="baseline_scores.png" alt="Heuristic bar scores" width="48%" />
</p>
*These are **not** the fine-tuned model’s learning curve; they are background on the reference policy only.*

## How step reward is shaped

Each day, `step` returns **one** floating-point reward that is the **sum** of a fixed list of **signed components** in `_compute_reward` (solvency, on-time pay, overdraft, interest drag, predatory exposure, inaction, savings abuse, and similar). That makes the daily signal **verifiable in code**: you can read the function and see what moved the number, like a check-list rather than a single opaque score. The goal is a stable training signal, not a human rating.

| Encouraged (examples) | Discouraged (examples) |
|------------------------|-------------------------|
| Staying liquid, on-time pay, paying down predatory exposure, healthy buffers, non-trivial action mix | Delinquency, overdraft, default chains, interest bleed, inaction, carrying exploitative instruments |

**Implementation:** `_compute_reward` in `server/my_env_environment.py` sums the day’s components; episode-level grading in `grade_*` / `grade_episode` in `tasks.py`.

## Episode grade

The episode outcome is **a score between 0 and 1**. After the last step, `grade_episode` picks the grader for `easy`, `medium`, or `hard` and returns that scalar (internally, values are nudged so they are never **exactly** 0.0 or 1.0 before rounding). **Easy** mixes overdraft days, on-time bill ratio, savings, and credit change with fixed weights. **Medium** adds interest paid against a worst-case budget and a “wisdom” term (informal and festive loans, emergencies survived). **Hard** adds defaults, stronger credit and savings conditions, and festival-loan use. The mix is **deterministic** given the `history` dict the environment accumulated; the same episode always gets the same final grade. **`get_episode_score()`** in the environment is what the evaluation and bar chart code call for that end-of-episode number.

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

Colab often trains against a **local clone** for speed; the **authoritative** network entry is still the Space URL in the table above. For a **strict** client–server line, point the client at that URL.

## Stack

- **OpenEnv:** this Space depends on **`openenv-core[core] ≥ 0.2.3`**.  
- **Training notebook:** **Unsloth** for 4-bit **Qwen2.5-7B**, **Hugging Face TRL** (including the policy-optimization trainer you see in the notebook), standard Colab or similar GPU.  
- **Service:** **FastAPI** + **Uvicorn**; exact pins live in `pyproject.toml` / `requirements.txt`.  
- **Space:** this repository is the Space when pushed to `origin`.

## Manifest, license, exports

- [`openenv.yaml`](openenv.yaml): `server.app:app`, port **7860**, task ids as above.  
- Do not register **custom** tools named `reset`, `step`, `state`, or `close` on top of OpenEnv.  
- After low-rank fine-tuning and merge, follow Unsloth’s **save** recipe and re-run a short eval; a bad merge often shows up only at inference.

[MIT](LICENSE) · [OpenEnv hub](https://huggingface.co/openenv) · [`openenv-core` on PyPI](https://pypi.org/project/openenv-core/). When we mention public Indian macro and payment statistics, they are for **narrative** only; the simulation does not ingest private account data. Optional local helpers: `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py` (not on the server path).

</div>
