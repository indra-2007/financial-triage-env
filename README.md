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
<h3 style="font-size:1.35em;font-weight:600;margin:0.4em 0 0.2em;">Miss a payment here and the score drops.<br/>No explanation, no retry — same rule every time.</h3>
<p style="font-size:1.05em;margin:0.5em 0 1em;"><strong>OpenEnv · Hugging Face Space</strong></p>

[![Space](https://img.shields.io/badge/🤗%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/openenv--core-≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

A day-by-day, **INR**-denominated household-finance simulator on **OpenEnv**: the agent picks one of eleven actions per day across 30/60/90-day episodes, the env returns a dense per-step reward, and a deterministic grader returns a single **0–1** score at the close. Rules are in code you can read (`server/my_env_environment.py`, `tasks.py`) and argue with; the stochastic parts are seeded, so every number below is reproducible.

**Live demo for judges:** open the Space URL and a browser lands at **`/demo/`** — pick a task + seed → **Reset** → **Fill baseline** → **Step** → read the **Last reward** and final **Grade (0–1)**. API clients still get JSON at `/`.

## Hackathon submission, rubric, and where to look

| Weight | Criterion | Strongest evidence here |
|-------:|-----------|-------------------------|
| **40%** | Innovation | 14 additive reward terms in `_compute_reward` (overdraft, bill, debt-APR-weighted service, credit delta, liquidity-buffer bonus, 7-day action-diversity; minus late-pay, interest bleed, default, predatory-loan carry, inaction streak); mechanical anti-gaming (no savings-credit for same-day withdraw+redeposit; do-nothing streak penalty); INR stochastic UPI micro-spend + P2P pressure; an informal-loan offer whose label undersells its APR; **env ablation below** confirms mechanics bind. |
| **30%** | Story | This README (≈ 5-minute read) + [`MINI_BLOG.md`](MINI_BLOG.md) + the `/demo/` UI on the Space. |
| **20%** | Training evidence | [`training_loss_7b.png`](training_loss_7b.png), [`before_after_scores_7b.png`](before_after_scores_7b.png), plus **multi-seed** [`heuristic_scores_ci.png`](heuristic_scores_ci.png) / [`heuristic_scores.json`](heuristic_scores.json) and [`ablation_env.png`](ablation_env.png) / [`ablation_env.json`](ablation_env.json). |
| **10%** | Reward + pipeline | GRPO `reward_fn` **replays** expert `prefix_actions` for the row’s `(task_id, seed, day)`, **strictly parses** the model action, runs **one** `env.step`, and optimizes the scaled dense return (`_last_breakdown['total']`, with the end-of-episode bonus folded in) — **not** `grade_episode`. The bar chart averages `grade_episode`. |

**Themes:** **#2** long-horizon planning (30/60/90 days, compounding interest + bills) · **#3.1** economic / world modeling (stochastic paydays, partial observability between loan label and APR, mechanical default/overdraft). Multi-agent (#1) and self-spawning curricula (#4) are explicitly out of scope.

## Judge-facing materials

| What | Where |
|------|-------|
| **Live Space (submit this URL)** | <https://huggingface.co/spaces/indra-dhanush/financial-triage-env> — browsers auto-land on `/demo/` |
| **Colab training** (Unsloth + TRL, SFT → GRPO) | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| **Mini write-up** | [`MINI_BLOG.md`](MINI_BLOG.md) |
| **Multi-seed baseline eval** | [`heuristic_scores.json`](heuristic_scores.json) · [`heuristic_scores_ci.png`](heuristic_scores_ci.png) — from [`scripts/eval_heuristic.py`](scripts/eval_heuristic.py) |
| **Env ablation** | [`ablation_env.json`](ablation_env.json) · [`ablation_env.png`](ablation_env.png) — from [`scripts/ablation_env.py`](scripts/ablation_env.py) |
| **Video (≤2 min)** | *Paste YouTube URL here once uploaded; do not commit video files to the Hub repo.* |
| **Experiment tracking** | With `WANDB_API_KEY` set, notebook logs to W&B; *paste the public run URL here.* |

## What happens each day: reset, step, grade

**OpenEnv** contract: **`reset(task_id)`** → **`step(action)`** until done → one **grader scalar** in `[0, 1]`. One transition is one day; one action from eleven options (pay bill / pay minimum / defer / pay extra debt / savings in–out / formal, informal, or festive loan / negotiate / do-nothing).

<p align="center">
  <img src="flowchart.png" alt="OpenEnv loop: reset, observation, one of eleven actions, one-day step, loop or grader" width="100%" style="max-width:1100px;" />
</p>

*Each turn: observe balances, instruments, risk, `daily_summary` → act → paydays, stochastic UPI spends, interest, fees, shocks → per-step return from `_compute_reward` → continue or final grade.*

Three difficulties — **easy (30d)** fresh-grad with three bills and no debt; **medium (60d)** with three debts (42% APR credit card, SBI EMI, BNPL), a mid-episode medical emergency, and a 240% APR moneylender offer; **hard (90d)** after job-loss with four debts totalling ₹7.3 lakh, two medical shocks, Diwali week, a 365% APR moneylender and a rent hike. Task configs and the difficulty-specific graders live in [`tasks.py`](tasks.py).

## Reference-policy evaluation (multi-seed, 95% bootstrap CI)

Produced by [`scripts/eval_heuristic.py`](scripts/eval_heuristic.py) at `n=60` seeds per `(policy, task)`. Raw per-seed scores in [`heuristic_scores.json`](heuristic_scores.json). **Re-run:** `python -m scripts.eval_heuristic --seeds 60`.

<p align="center"><img src="heuristic_scores_ci.png" alt="Reference policy mean episode score with 95% CI" width="720" /></p>

| Policy | easy | medium | hard |
|--------|------|--------|------|
| **heuristic** (rule-based; SFT data source and `Fill baseline` button) | **0.999** | **0.694** | **0.423** [0.417, 0.428] |
| `random_valid` (do-nothing ∪ pay bill ∪ pay minimum, uniform) | 0.933 | 0.496 | 0.421 |
| `do_nothing` (skip every day — null baseline) | 0.578 | 0.295 | 0.316 |

Easy is nearly solved by the heuristic; medium opens a clear `heuristic ≫ random ≫ do-nothing` gap. On **hard** the heuristic only edges out `random_valid` (Δ ≈ 0.002, n=60) — that is the honest finding and the reason this env is useful for SFT → GRPO. Flat CIs on easy/medium mean the rounded grade lands on the same value after RNG-driven day-to-day details wash out; hard has real seed variance.

## Environment ablation — the mechanics actually bind

Same heuristic, same seeds (`n=40`), one configured mechanic nulled out per run. Scores rising when a mechanic is disabled = that mechanic is binding on the baseline. From [`scripts/ablation_env.py`](scripts/ablation_env.py); **re-run:** `python -m scripts.ablation_env --seeds 40`.

<p align="center"><img src="ablation_env.png" alt="Environment ablation: heuristic score when a mechanic is disabled" width="720" /></p>

| Ablation (hard task) | Mean score | Δ vs `full` |
|----------------------|------------|-------------|
| **full** (shipping env) | **0.421** | — |
| `no_medical_emergency` | 0.594 | **+0.17** |
| `no_upi_micro_spend` | 0.483 | **+0.06** |
| `no_festival` | 0.444 | +0.02 |

Medical emergencies are the largest binding constraint on the heuristic in Hard; UPI leaks are a steady drag; festive-pressure has a small but real effect. Easy / medium barely move — that is what distinct difficulties should look like.

## Training stack — what we ran and what it optimizes

**SFT.** Behavioral cloning on trajectories rolled out by the heuristic (`sft_dataset.jsonl`), so the model learns the action-string format and a sane opening repertoire. **GRPO** in TRL on Unsloth-quantized **Qwen2.5-7B-Instruct**: each row has a stored prefix of expert actions which is replayed deterministically on the same `(task_id, seed)` to reconstruct the observation at day `d`; the model’s single action string is **strictly parsed** (no heuristic fill-in if it fails), **one** `env.step` runs, and the optimizer sees the scaled dense `_last_breakdown['total']` (end-of-episode bonus folded into the last day). Checkpoints are LoRA adapters; merge with Unsloth if you need a single-file export.

<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs step for 7B" width="560" /></p>

<p align="center"><img src="before_after_scores_7b.png" alt="Heuristic, SFT, GRPO mean episode score by difficulty" width="720" /></p>

*Orange / blue / green = heuristic / SFT / GRPO.* These bars come from the **single evaluation pass** inside the notebook; the plain-text multi-seed heuristic numbers above (`heuristic_scores.json`) are what we quote.

<details><summary>Reward terms (short)</summary>

**Pushed up:** no-overdraft, on-time bill payment, debt-APR-weighted service, savings-growth (anti-churn gated), credit-score improvement, emergency-buffer bonus, 7-day action-diversity.
**Pushed down:** late payment, overdraft, interest accrued, default, zero-savings, predatory-loan carry, consecutive do-nothing streak.

14 keys in the `breakdown` dict inside `_compute_reward`; see `server/my_env_environment.py`. Episode grade (`tasks.py::grade_episode`) is a different object with difficulty-specific weights on overdraft / bills / credit / savings / interest / defaults / loan hygiene / emergency survival / temptation resistance.

</details>

## Live demo UI (same Space, same origin)

OpenEnv’s default HTTP `POST /reset` and `POST /step` create a **new** env per request, which doesn’t survive a multi-day walkthrough. `server/video_demo_server.py` adds a small `_lock`-protected session API and the static UI, mounted on the main app so:

- `/` — JSON status for API / EnvClient (browsers are 302'd to `/demo/`).
- `/demo/` — interactive pitch UI (reset, step, `Fill baseline`, live metrics, final grade).
- `/api/demo/{reset,step,heuristic,score}` — the JSON the UI talks to.

**Local:** `python -m server.video_demo_server` → `http://127.0.0.1:8088/`. Free public hosting of the standalone server is a one-click Render Blueprint ([`render.yaml`](render.yaml)); its free tier sleeps on idle so first-load is slow and in-memory episodes reset. Vercel-style serverless is **not** a fit — the session is in-memory on one process.

## Run it, and re-run every claim

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e .

# Main OpenEnv app (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860
# or: docker build -t financial-triage . && docker run -p 7860:7860 financial-triage

# Multi-seed baseline table + CI figure (~4s on a laptop)
python -m scripts.eval_heuristic --seeds 60

# Env-mechanic ablation
python -m scripts.ablation_env --seeds 40

# Same UI locally as /demo/ on the Space
python -m server.video_demo_server
```

```python
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

## Limitations and what's next

- Heuristic vs `random_valid` on **hard** is close (Δ ≈ 0.002 at n=60). The rule-based teacher is not optimal; an SFT student can match it without really solving the task. GRPO’s job is to cross that gap, which is also why the bar chart is the right artefact to compare across stages.
- Grader scores on **easy / medium** flatten across seeds in rounded form. Day-to-day details still vary; use the dense per-step return for seed-level comparisons (that is what GRPO optimizes).
- APRs, Diwali pressure magnitudes, and UPI leak rates are **calibrated** from published Indian-finance statistics cited in [`tasks.py`](tasks.py) — not fit to any real household panel. The grader is a design object, not an econometric estimate.
- **Next:** add a seed-matched `model − heuristic` plot; promote `random_valid` to greedy-by-APR; sample emergency arrival times and moneylender APRs instead of hard-coded scalars; publish LoRA adapters as a separate Model Hub repo.

## Stack, manifest, license

- **`openenv-core[core] ≥ 0.2.3`**; `Environment` in `server/my_env_environment.py`; `create_app` in `server/app.py`; `openenv.yaml` points at `server.app:app`, port **7860**, tasks `easy · medium · hard`.
- Training: **Unsloth** 4-bit **Qwen2.5-7B**, **TRL** GRPO trainer (see notebook). Serving: **FastAPI** + **Uvicorn**; pins in [`pyproject.toml`](pyproject.toml) / [`requirements.txt`](requirements.txt).
- Don't name extra MCP/HTTP tools `reset`, `step`, `state`, `close`. Smoke-test inference after any LoRA merge — bad merges hide until decode.

[MIT](LICENSE) · [OpenEnv](https://huggingface.co/openenv) · [`openenv-core`](https://pypi.org/project/openenv-core/). Cited macro statistics are **narrative only**. Helpers that aren't on the server path: `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py`.

</div>
