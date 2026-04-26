---
title: Financial Triage Environment
emoji: 💰
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Financial Triage: Can an LLM Survive 90 Days of India’s Household Finance Crisis?

> *“We built an OpenEnv where an LLM learns what many were never taught—which bill to pay first, when a ‘quick cash’ loan is a trap, and how UPI micro-spend compounds across a month.”*

| What | Link |
|------|------|
| **Live Space (judges pull from here)** | [**`indra-dhanush/financial-triage-env`**](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) |
| **Training notebook (Unsloth + TRL, Colab)** | Open in Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb) in this repo |
| **Mini write-up (required materials)** | [MINI_BLOG.md](MINI_BLOG.md) — also on the Hub: [view `MINI_BLOG.md` in this Space](https://huggingface.co/spaces/indra-dhanush/financial-triage-env/blob/main/MINI_BLOG.md) |
| **Demo video (≤2 min, add when published)** | *Replace the placeholder in the “Media & judging checklist” section below with your public YouTube URL (do not check large video files into the Space).* |

**Stack (self-serve guide):** Environment (OpenEnv) → **dense, verifiable rewards** + episode grader → **SFT** (trajectory warmup) → **GRPO** (TRL) with **Unsloth** (4-bit **Qwen2.5-7B-Instruct**) → deploy on **Hugging Face Spaces**.

**`openenv-core`:** this project is built and tested on **`openenv-core[core] >= 0.2.3`** (current OpenEnv line on PyPI as of the hackathon build).

---

## OpenEnv hackathon — theme alignment

We position **Financial Triage** under the official theme rubric as follows (pick what you lead with in the pitch; both apply):

- **Theme #2 — (Super) long-horizon planning & instruction following**  
  Episodes last **30 / 60 / 90** simulated days (one action per day). The agent must track bills, paydays, debt cycles, and shocks across a long trajectory—**not** a single-turn chat.

- **Theme #3.1 — World modeling, professional / economic task**  
  A **partially observable** money world: stochastic UPI-style leakage, **misleading** informal loan rates vs realized costs, **interest and defaults**, and **CIBIL-style** dynamics—all **programmatically** verified.

This is **not** a multi-agent (Theme #1) or self-play generator (Theme #4) benchmark; the focus is a **single agent** in a **realistic economic simulator** with **objective** rewards.

---

## Judging criteria — how this README answers them

| Criterion | Where it lives in this submission |
|-----------|-----------------------------------|
| **Environment innovation** | Realistic INR household simulation, 11 action types, loan-shark partial observability, UPI micro-spend, medical and festival pressure, 14-part dense reward + anti-hacking. |
| **Storytelling** | This file + [MINI_BLOG.md](MINI_BLOG.md) — *Problem → environment → what we trained → results → why it matters*. |
| **Showing improvement in rewards (20%)** | **SFT training loss (7B)** and **Heuristic vs SFT (7B) vs GRPO (7B)** bar chart below (committed PNGs, not only Colab outputs). |
| **Reward & training pipeline (10%)** | Dense reward table + SFT on heuristic traces + **GRPO on live environment** (see notebook). |

**Minimum requirements checklist**

- [x] **OpenEnv** — `openenv.core.env_server` base classes, `openenv.yaml`, `create_app` in `server/app.py`
- [x] **Unsloth or TRL in Colab** — `financial_triage_training.ipynb`
- [x] **Loss and reward / score plots** — embedded below, files committed
- [x] **Write-up** — this README + [MINI_BLOG.md](MINI_BLOG.md) (and optional HF blog / YouTube — see *Media* section)
- [x] **Hugging Face Space** — link at top; **this repo is the Space** when pushed to `origin`

### Media and judging materials (all linked from this README)

1. **Mini-blog / write-up (in-repo, judge-readable on Hub):** [MINI_BLOG.md](MINI_BLOG.md)  
   *Optional:* duplicate the same text to a [Hugging Face blog post](https://huggingface.co/new-blog) or [HF Posts](https://huggingface.co/posts) and add your public URL in a follow-up commit.

2. **Video (optional but high-impact):** after you upload a **≤2 minute** video to YouTube, add one line under your repo’s *Quick links* or here:  
   `Demo video: https://www.youtube.com/watch?v=YOUR_VIDEO_ID`  
   (Use a URL only—**do not** commit large video files to the Space.)

3. **Plots:** saved as `training_loss_7b.png` and `before_after_scores_7b.png` in the repository root; embedded with captions in the *Results* section.

---

## Why this problem

Household and unsecured debt, UPI usage, and informal credit are central to real Indian financial stress. Fintech products visualize spending; this environment **trains a policy in the loop** with **stochastic** income and spend, **mandatory** shocks, and **deliberate** predatory-lending **mislabels** so the model cannot rely on text alone.

**Data sources (narrative calibration only, not private data):** RBI financial literacy and credit statistics, NSSO health expenditure, NPCI UPI, NCRB public reports, and similar **public** references cited in the original project research notes.

---

## How the environment works

![Environment loop: reset → observe → act → step → grade](flowchart.png)

*Caption: one daily loop. The agent receives a full financial observation, emits **one** action, the simulator advances one day, and the agent sees **dense step reward** plus a natural-language **day summary**.*

**One step = one day.** The simulator applies salary/ freelance credits, UPI micro-transactions, interest, bill and debt rules, medical and festival events, and updates a credit-score proxy and history for grading.

---

## Task tiers (curriculum)

| | Easy (30d) | Medium (60d) | Hard (90d) |
|---|------------|--------------|------------|
| **Income** | ₹30K / month | ₹55K + freelance | ₹65K then **job loss** then gig pay |
| **Debt** | None | CC, EMI, BNPL, traps | Four debts, large principal |
| **Events** | UPI noise | loan-shark + medical | two emergencies + Diwali |
| **Bills** | stochastic ±% | stochastic ±% | stochastic ±% |

`openenv.yaml` lists `task_id` values: `easy`, `medium`, `hard`.

---

## What the agent sees and does

- **Observation:** Checkings/savings, bills, debts, risk signals, optional loan offers, active emergency, festival block, and `daily_summary` text.  
- **Actions (examples):** `pay_bill_full`, `pay_minimum`, `defer_bill`, `pay_extra_debt`, `transfer_to_savings`, `withdraw_emergency`, `take_formal_loan`, `take_informal_loan`, `take_festive_loan`, `negotiate_bill`, `do_nothing`.  
- **Anti–reward-hacking (design intent):** multiple independent **penalties and bonuses** (e.g. savings churn, inaction streaks, predatory debt, diversity)—see *Reward* section.  
- **Episode score:** `grade_episode` in `tasks.py` returns a **scalar in [0,1]** per difficulty, comparable across runs.

---

## Training you can reproduce

| Phase | What | Model (reported run) |
|--------|------|------------------------|
| **SFT** | Imitation on heuristic rollouts (see `sft_dataset.jsonl` and notebook) | **Qwen2.5-7B-Instruct** (4-bit, Unsloth) |
| **GRPO** | TRL + environment rollouts, verifiable rewards | Same backbone |

*T4 / small GPU:* you may switch the notebook to **`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`** for a cheaper run; **all figures in this repo are for the 7B run** unless you regenerate and commit new PNGs.

---

## Results — evidence of training (loss + before / after)

### SFT training loss (7B)

**X-axis: training step · Y-axis: loss (cross-entropy on SFT).**

![SFT training loss for Qwen2.5-7B: loss versus training step](training_loss_7b.png)

*Caption: SFT loss falls over 60 steps and flattens at a low value—indicates a stable supervised phase before GRPO.*

### Heuristic (baseline) vs SFT (7B) vs GRPO (7B) — mean episode score by difficulty

**Y-axis: episode score in [0,1]. Orange = heuristic (hand-coded) baseline, not a pretrained LLM. Blue = SFT, Green = GRPO.**

![Heuristic vs SFT 7B vs GRPO 7B episode scores on Easy, Medium, Hard](before_after_scores_7b.png)

*Caption: three-way comparison on Easy (30d), Medium (60d), Hard (90d). Heuristic is the strong program baseline; SFT brings the model on-distribution; GRPO refines with environment feedback. Exact bar values are printed on the figure.*

### Heuristic-only baselines (context)

Cumulative **step reward** trajectories and **heuristic-only** bar snapshot (for intuition; **not** a substitute for the trained-model plots above).

![Heuristic agent: reward vs day for each difficulty](reward_curves_baseline.png)

*Caption: heuristic daily reward accrual—hard mode shows stress around job loss and second shock.*

![Heuristic-only episode scores by difficulty](baseline_scores.png)

*Caption: heuristic mean scores (Easy / Medium / Hard) used as a reference level.*

---

## Dense reward (multi-component) — specification sketch

The step reward combines **many** terms (on-time pay, overdraft, defaults, interest, buffer, diversity, etc.). A monolithic 0/1 at the end would be too sparse for learning; this design matches the self-serve guide’s “multiple independent checks” and “anti-hacking” goals.

| Direction | Examples (see code for full list) |
|-----------|-----------------------------------|
| Positive | No overdraft, bill on time, paying high-APR debt, savings buffer, CIBIL improvement, action diversity |
| Negative | Late payment, overdraft, default, interest drag, predatory carry, idle streaks, zero buffer |

**Implementation:** `server/my_env_environment.py` (`_compute_reward` and related helpers). **Episode-level** grader: `tasks.py` (`grade_episode` / `grade_*`).

---

## Quick start (local or Docker)

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Docker:** `docker build -t financial-triage .` then `docker run -p 7860:7860 financial-triage`  
(Health: `GET /health`.)

**Python client (remote):**

```python
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

**Note on client/server:** the **served** environment is the source of truth for OpenEnv. The Colab **notebook** may import the environment **locally** for fast training from a cloned copy—acceptable for a hackathon training loop. For a strict client-only story, use `EnvClient` against your Space URL.

---

## OpenEnv manifest

- **File:** [openenv.yaml](openenv.yaml)  
- **App:** `server.app:app`, port `7860`, tasks `easy` / `medium` / `hard`.

Do **not** name custom MCP or HTTP tools `reset`, `step`, `state`, or `close` (OpenEnv / MCP reserved patterns).

---

## License and references

- **License:** [MIT](LICENSE)  
- **OpenEnv:** [OpenEnv on Hugging Face](https://huggingface.co/openenv) · [`openenv-core` on PyPI](https://pypi.org/project/openenv-core/)  
- **Hackathon context:** [OpenEnv India / themes](https://huggingface.co/openenv) (see theme PDF you received)

**Optional dev utilities in repo:** `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py` — not required to run the Space; safe to ignore for evaluation.

---

## QLoRA / saving models (read before you push adapters)

If you use 4-bit QLoRA in Colab, **follow Unsloth’s documented merge / save path**; do not naïvely upcast and merge in a way that corrupts weights. **Smoke-test** your exported adapter or merged model on a few `reset` / `step` calls before the submission freeze.

---

*README version aligned with: OpenEnv Hackathon minimum requirements, self-serve training guide, and judging rubric (innovation, storytelling, reward evidence, pipeline). Push this repository to the Space `origin` before the submission deadline so judges see the same commit.*
