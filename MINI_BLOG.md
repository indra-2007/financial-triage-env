# Financial Triage: Training an LLM to Survive 90 Days of Indian Household Finance (OpenEnv)

*OpenEnv India Hackathon 2026 — short write-up (mirror of project documentation; you may repost to [Hugging Face Posts](https://huggingface.co/posts) or the HF blog and link the public URL from the Space README).*

## Problem

Most Indians never get a curriculum for **bill triage, predatory credit, and emergency shocks** under real UPI and informal lending dynamics. Fintech UIs show charts; this project asks whether an **LLM can learn a policy** that stays solvent, preserves a credit score proxy (CIBIL-style), and avoids traps—under **stochastic bills, UPI “micro-leaks,” job loss, medical shocks, and festival pressure**.

## Environment (OpenEnv)

**Financial Triage** is a Gym-style OpenEnv environment: **one step = one day**, one discrete **financial action** per step, dense **per-step reward** (multiple additive terms), and an **episodic grader** in \([0,1]\) for easy / medium / hard.

- **Observation:** Account balances, bills, debts, risk signals, optional loan offers, emergencies, and festival metadata—plus a short textual **daily summary**.
- **Actions:** pay bills, pay minimums, defer, pay down debt, move savings, formal vs informal vs festive loans, negotiate bills, or do nothing.
- **Verifiability:** State transitions and rewards are **fully programmatic**; no human-in-the-loop scoring.

## Hackathon theme alignment

- **Theme #2 — Long-horizon planning & instruction following:** Episodes run **30 / 60 / 90 days**; credit and cash-flow effects compound. The agent must track obligations over a horizon longer than a single “chat turn.”
- **Theme #3.1 — Professional / world modeling (economic simulation):** The world has **stochastic paydays**, **interest accrual**, **partial observability** (predatory rate vs advertised label), and **regulatory-sounding** constraints (defaults, overdraft, buffers).

## Training stack (what we actually ran)

1. **SFT (warm start):** Behavioral cloning on short trajectories from a **heuristic** policy to teach valid action strings and task format.  
2. **GRPO (improvement):** TRL + Unsloth, **4-bit `Qwen2.5-7B-Instruct`**, with rewards from **live** `env.step` after **replaying the expert `prefix_actions`** for that row’s `(task_id, seed, day)`, then a **strict** parse of the model’s action string; the scalar is **dense** `_last_breakdown['total']` (not `grade_episode`), normalized for TRL. Checkpoints are **LoRA adapters**; merge to full weights with Unsloth if you need a single merged file for export.

*Hardware note:* 7B training was run in **Colab** (e.g. A100 with HF credits); Unsloth reduces memory and speeds up the loop.

## What improved (evidence)

We report three bars per difficulty—**heuristic (baseline)**, **SFT (7B)**, **GRPO (7B)**—and the **SFT training loss** curve for 7B. Heuristic is a **hand-coded** reference, not a pretrained LLM. The key claim for judges: **SFT** learns the format and rough policy; **GRPO** nudges the policy using environment feedback so episode scores **track closer to the strong heuristic** on hard scenarios.

**Figures in this repository:** `training_loss_7b.png`, `before_after_scores_7b.png`, plus heuristic-only baselines for context.

## Why it matters

If LLMs are to act as **assistants in real money contexts**, they need training on **stochastic, partially observable, incentive-laden** worlds—not toy grids alone. This environment is a small step toward that, with a **reproducible** Space, **objective** rewards, and a **documented** SFT → GRPO path.

## Links

- **Live Space:** [https://huggingface.co/spaces/indra-dhanush/financial-triage-env](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)  
- **OpenEnv:** [https://huggingface.co/openenv](https://huggingface.co/openenv)  
- **Code / notebook:** this Space repository (`financial_triage_training.ipynb`).

## Data & ethics

All amounts are **fictional INR simulations** for research. Real financial advice requires licensed professionals. See `README.md` for cited public statistics and surveys used for calibration narrative only.
