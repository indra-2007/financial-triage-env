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

### An Indian-household-finance simulator on OpenEnv — the agent chooses what to pay, what to defer, and which loan to refuse, one day at a time for 30 to 90 days.

<p style="font-size:1.05em;margin:0.4em 0 1em;">Rupee-denominated. Rule-based, not vibes-based. Miss a payment and the score drops.</p>

[![Space](https://img.shields.io/badge/🤗%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/openenv--core-≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

Every step is one simulated day. The observation is a rupee ledger — checking, savings, due bills, open debts, credit score, active emergencies — plus a short `daily_summary`. The action is one of eleven templates (pay a specific bill, pay debt minimum, pay extra principal, move savings in or out, take one of three loan types, negotiate, do nothing). The environment returns a dense scalar reward per step; at the end it returns one deterministic grade in `[0, 1]`.

The interesting parts — APR-weighted debt service, mechanical overdraft, UPI micro-spend, informal loans whose advertised rate undersells the APR — all live in `server/my_env_environment.py` and `tasks.py`. The stochastic bits are seeded, so every number on this page is reproducible on a laptop.

Open the Space URL and a browser lands on **`/demo/`**: pick a task, pick a seed, **Reset** → **Fill baseline** → **Step**. Last reward and final grade update live. API clients still get JSON at `/`.

## Rubric

| Weight | Criterion | Where to look |
|-------:|-----------|---------------|
| **40%** | Innovation | 14 additive reward terms in `_compute_reward`: overdraft, on-time bill pay, APR-weighted debt service, credit-score delta, savings-growth (gated against same-day withdraw-and-redeposit churn), liquidity-buffer bonus, 7-day action diversity; minus late pay, interest bleed, default, predatory-loan carry, and consecutive do-nothing streaks. UPI micro-spends and peer-pressure nudges run on top. The informal-lender quote in the observation advertises a per-day rate that hides a 240–365% APR. The [environment ablation](#environment-ablation-the-mechanics-actually-bind) shows these mechanics bind on the baseline, not just on paper. |
| **30%** | Story | This README (aimed at a five-minute read) plus [`MINI_BLOG.md`](MINI_BLOG.md) and the `/demo/` UI. |
| **20%** | Training evidence | [`training_loss_7b.png`](training_loss_7b.png), [`before_after_scores_7b.png`](before_after_scores_7b.png), plus the multi-seed artifacts [`heuristic_scores.json`](heuristic_scores.json) / [`heuristic_scores_ci.png`](heuristic_scores_ci.png) and [`ablation_env.json`](ablation_env.json) / [`ablation_env.png`](ablation_env.png). |
| **10%** | Reward and pipeline | The GRPO `reward_fn` does not call `grade_episode`. For each row it replays the stored expert `prefix_actions` on the same `(task_id, seed)` to reconstruct the exact day-`d` observation, strictly parses the model's single action (no heuristic fallback if parsing fails), runs one `env.step`, and optimizes the scaled dense return `_last_breakdown['total']` — with the end-of-episode bonus folded into the last day. `grade_episode` is reserved for the bar chart and for the Space's `/score` endpoint, so the two signals stay distinct. |

**Track fit.** This submission targets **#2 Long-horizon planning** — 30 / 60 / 90-day arcs where interest and bills compound — and **#3.1 Economic / world modeling** — stochastic paydays, partial observability between loan labels and APRs, mechanical default and overdraft. Multi-agent (#1) and self-spawning curricula (#4) are out of scope on purpose.

## Materials

| What | Where |
|------|-------|
| Space (runnable environment and `/demo/`) | <https://huggingface.co/spaces/indra-dhanush/financial-triage-env> |
| Colab notebook — SFT then GRPO on Unsloth + TRL | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| Write-up | [`MINI_BLOG.md`](MINI_BLOG.md) |
| Multi-seed baseline evaluation | [`heuristic_scores.json`](heuristic_scores.json) · [`heuristic_scores_ci.png`](heuristic_scores_ci.png) — [`scripts/eval_heuristic.py`](scripts/eval_heuristic.py) |
| Environment ablation | [`ablation_env.json`](ablation_env.json) · [`ablation_env.png`](ablation_env.png) — [`scripts/ablation_env.py`](scripts/ablation_env.py) |
| Weights & Biases | The notebook logs to W&B whenever `WANDB_API_KEY` is set. |

## The daily loop: reset, step, grade

`reset(task_id)` returns the opening ledger. Each `step(action)` advances the clock by one day: paydays land, UPI spends fire, interest accrues, bills come due, shocks roll their dice, and the reward breakdown is computed before the next observation. When the horizon closes, the grader returns a single score in `[0, 1]`.

<p align="center">
  <img src="flowchart.png" alt="OpenEnv loop: reset, observation, one of eleven actions, one-day step, continue or grade" width="100%" style="max-width:1100px;" />
</p>

Three difficulties.

- **Easy (30 days).** Fresh-graduate profile, steady salary, three recurring bills, no debt. The grader rewards on-time rent, electricity, and recharge.
- **Medium (60 days).** Salary plus a side income, three live debts (42% APR credit card, SBI personal-loan EMI, BNPL), a mid-episode medical emergency, and a 240% APR informal-loan pitch.
- **Hard (90 days).** Post-job-loss household; four debts totalling ₹7.3 lakh, two medical shocks, a Diwali week spending window, a 365% APR moneylender standing by, and a rent hike partway through.

Task configs and the per-difficulty `grade_*` functions live in [`tasks.py`](tasks.py).

## Baselines across 60 seeds

Produced by [`scripts/eval_heuristic.py`](scripts/eval_heuristic.py) with `n = 60` seeds per `(policy, task)` and a percentile bootstrap on the mean. Raw per-seed scores are in [`heuristic_scores.json`](heuristic_scores.json). Re-run with `python -m scripts.eval_heuristic --seeds 60`.

<p align="center"><img src="heuristic_scores_ci.png" alt="Reference policies: mean episode score with 95% bootstrap CI" width="720" /></p>

| Policy | easy | medium | hard (95% CI) |
|--------|------|--------|---------------|
| **heuristic** — the rule I hand-wrote; also the SFT data source and the `/demo/` Fill-baseline button | **0.999** | **0.694** | **0.423** [0.417, 0.428] |
| `random_valid` — uniform over do-nothing, pay-bill, pay-minimum | 0.933 | 0.496 | 0.421 |
| `do_nothing` — null baseline; skip every day | 0.578 | 0.295 | 0.316 |

Read: easy is close to solved by the rule, medium opens the clean ordering `heuristic ≫ random ≫ do_nothing`, and **hard** is where my rule-based teacher stops being a wall — it only edges `random_valid` by Δ ≈ 0.002 at n = 60. That is the honest finding, and it is exactly why the environment is interesting for SFT then GRPO: the student has room to actually beat the teacher. Flat CIs on easy and medium mean the rounded grade lands on the same value once the RNG-driven day-to-day details wash out; hard has real seed variance and the CI reflects it.

## Environment ablation: the mechanics actually bind

Same heuristic, same `n = 40` seeds, one configured mechanic nulled out per run. If the score rises with a mechanic disabled, that mechanic is binding on the baseline. From [`scripts/ablation_env.py`](scripts/ablation_env.py); re-run with `python -m scripts.ablation_env --seeds 40`.

<p align="center"><img src="ablation_env.png" alt="Heuristic score on hard when one mechanic is disabled" width="720" /></p>

| Ablation (hard task) | Mean score | Δ vs `full` |
|----------------------|------------|-------------|
| **full** — shipping env | **0.421** | — |
| `no_medical_emergency` | 0.594 | **+0.17** |
| `no_upi_micro_spend` | 0.483 | **+0.06** |
| `no_festival` | 0.444 | +0.02 |

Medical shocks are the single largest binding constraint on the heuristic in Hard; UPI leaks are a consistent lower-magnitude drag; festive-week pressure is smaller but real. Easy and medium barely move under the same ablation — which is what three separate difficulties should look like.

## Training: SFT, then GRPO against live `env.step`

SFT is behavioral cloning on heuristic rollouts (`sft_dataset.jsonl`). The point is not to teach a good policy — it is to teach the action grammar and a sane opening repertoire, so the GRPO parser never has to fail-safe.

GRPO runs in TRL on an Unsloth-quantized **Qwen2.5-7B-Instruct**. Each row carries a prefix of expert actions; the trainer replays that prefix deterministically on the row's `(task_id, seed)` to rebuild the day-`d` state, the model emits one action string, the parser accepts it strictly, one `env.step` runs, and the optimizer sees the scaled dense reward. Checkpoints are LoRA adapters — merge with Unsloth when you need a single-file export, and smoke-test inference afterwards because bad merges hide until decode.

<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs optimizer step for the 7B run" width="560" /></p>

<p align="center"><img src="before_after_scores_7b.png" alt="Mean episode score by difficulty: heuristic, SFT, GRPO" width="720" /></p>

Bars are one evaluation pass inside the notebook. The multi-seed heuristic numbers above (from `heuristic_scores.json`) are what I quote in prose.

<details><summary>Reward breakdown</summary>

Pushed up: no-overdraft, on-time bill payment, APR-weighted debt service, savings growth (gated against same-day churn), credit-score improvement, emergency-buffer bonus, seven-day action diversity.

Pushed down: late payment, overdraft, interest accrued, default, zero-savings, predatory-loan carry, consecutive do-nothing streak.

Fourteen keys in the `breakdown` dict inside `_compute_reward`. Episode grade (`tasks.py::grade_episode`) is a different object with difficulty-specific weights on overdraft, bills, credit, savings, interest, defaults, loan hygiene, emergency survival, and temptation resistance.

</details>

## The `/demo/` UI, and why it exists

OpenEnv's default `POST /reset` and `POST /step` create a new environment per HTTP call, which is correct for stateless RL clients but unhelpful for a live walkthrough. `server/video_demo_server.py` mounts a small lock-protected session API and a static UI on the same Space:

- `/` — JSON status for API clients and for `EnvClient`; browsers are 302'd to `/demo/`.
- `/demo/` — interactive UI with Reset, Step, Fill-baseline, live reward, and final grade.
- `/api/demo/{reset,step,heuristic,score}` — the small JSON surface the UI talks to.

Run it locally with `python -m server.video_demo_server` and hit `http://127.0.0.1:8088/`. A one-click [`render.yaml`](render.yaml) blueprint hosts the standalone server on Render's free tier; Render sleeps the dyno on idle, so first-load is slow and any in-memory episode resets after the sleep. Vercel-style serverless is not a fit — the session has to live in one process.

## Run it, and re-run every number above

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e .

uvicorn server.app:app --host 0.0.0.0 --port 7860
python -m scripts.eval_heuristic --seeds 60
python -m scripts.ablation_env --seeds 40
python -m server.video_demo_server
```

```python
# Raw HTTP against the Space — works with any client, no version dance.
import requests
BASE = "https://indra-dhanush-financial-triage-env.hf.space"
r = requests.post(f"{BASE}/reset", json={"task_id": "hard", "seed": 0}).json()
# r["observation"] is the full ledger; POST the next action to /step.
```

The Space exposes the full OpenEnv HTTP surface — `POST /reset`, `POST /step`, `GET /state`, `GET /metadata`, `GET /schema`, plus `/health` and `/mcp`. See `/docs` on the Space for the live OpenAPI. `AutoEnv.from_hub("indra-dhanush/financial-triage-env")` also works with `OPENENV_TRUST_REMOTE_CODE=1`, subject to the usual caveats about installing remote code.

Both evaluation scripts are standard-library plus `numpy` and `matplotlib`; no GPU. Outputs land at the repo root as `heuristic_scores.{json,png}`, `heuristic_scores_ci.png`, and `ablation_env.{json,png}` — the JSONs include every per-seed score, so a reviewer can recompute any mean or CI without trusting mine.

## What I would not over-claim

- On **hard**, the heuristic only edges `random_valid` by Δ ≈ 0.002 at n = 60. The rule-based teacher is not optimal, which is a feature for SFT → GRPO but also means an SFT-only student can match the teacher without really solving the task. That is exactly the gap GRPO is there to cross.
- On **easy** and **medium**, rounded grades flatten across seeds. The intra-episode story still varies; for seed-level comparisons use the dense per-step return, which is what GRPO optimizes against anyway.
- APRs, Diwali magnitudes, and UPI leak rates are **calibrated** from published Indian-finance statistics cited in [`tasks.py`](tasks.py). They are not fit to any real household panel, and the grader is a design object, not an econometric estimate.
- **Next on the list.** A seed-matched "model minus heuristic" plot, a stronger scripted baseline (greedy by APR), sampled rather than hard-coded emergency timing and informal-loan APRs, and the LoRA adapters pushed as a separate Model Hub repo.

## Stack and manifest

- **`openenv-core[core] ≥ 0.2.3`.** `Environment` in `server/my_env_environment.py`; `create_app` in `server/app.py`; `openenv.yaml` points at `server.app:app`, port **7860**, tasks `easy · medium · hard`.
- **Training:** Unsloth 4-bit Qwen2.5-7B, TRL GRPO trainer in the notebook. **Serving:** FastAPI + Uvicorn; versions pinned in [`pyproject.toml`](pyproject.toml) / [`requirements.txt`](requirements.txt).
- Do not expose extra MCP or HTTP tools named `reset`, `step`, `state`, or `close` — they collide with the OpenEnv contract. Helpers that are not on the server path: `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py`.

[MIT](LICENSE) · [OpenEnv](https://huggingface.co/openenv) · [`openenv-core`](https://pypi.org/project/openenv-core/). Any macroeconomic statistic cited in prose is narrative context, not a calibration target.

</div>
