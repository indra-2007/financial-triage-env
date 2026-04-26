# Financial Triage: Can an LLM Survive 90 Days of an Indian Household Budget?

*A short write-up for the OpenEnv India hackathon. Everything here has a file to point at.*

## Why this, and why now

Most Indians learn household finance by getting it wrong: a missed EMI, a recharge late enough to disconnect, a Diwali loan that looked small on the offer screen. Charts don’t teach that. I wanted an environment where an LLM has to actually **play the month** — feel the paydays, the interest, the Swiggy bleed, the 1 a.m. "instant cash" pitch — and get judged on how the books look at the end.

So I built **Financial Triage**: a day-by-day, **INR**-denominated simulator running on [OpenEnv](https://huggingface.co/openenv). One step is one day, one action out of eleven, a dense per-step reward, and a single **0–1** grade at the close. The state transitions are code you can read (`server/my_env_environment.py`), not a vibes-based prompt.

## What a run actually feels like

- **Observation** is structured: account balances, active bills, outstanding debts, risk signals (days to overdraft, interest accruing today, credit score, late-fee risk), optional loan offers, an active medical emergency if any, festival metadata if Diwali is on, and a short human-readable `daily_summary`. You don’t have to believe the label on a loan; you can see the APR.
- **Action** is one of eleven: `pay_bill_full`, `pay_minimum`, `defer_bill`, `pay_extra_debt`, `transfer_to_savings`, `withdraw_emergency`, `do_nothing`, `take_formal_loan`, `take_informal_loan`, `take_festive_loan`, `negotiate_bill`. Validators (`models.py`) reject malformed calls before the env sees them.
- **Reward** is the sum of 14 additive terms in `_compute_reward`: on-time bill payment, debt-APR-weighted service, credit-score deltas, a liquidity-buffer bonus, an action-diversity bonus — minus overdraft, late payment, interest bleed, defaults, predatory-loan carry, and an inaction-streak penalty. Mechanical anti-gaming is part of the reward, not a sidecar: we don’t credit savings growth for a same-day withdraw-plus-redeposit, and three-in-a-row `do_nothing` actions start bleeding points.
- **Grade** at the end (`tasks.py::grade_episode`) is deterministic from `history` — different weightings for easy/medium/hard, all in `[0, 1]`.

Three difficulties. **Easy** is a fresh graduate on ₹30k/month with three bills and no debt (30 days). **Medium** is a mid-career person with freelance income, three debts including a 42% APR credit card, a loan-shark offer, and a mid-episode medical emergency (60 days). **Hard** is the same person after losing their job, dropping to gig income, facing four debts totalling ₹7.3 lakh, two medical shocks, Diwali week, and a 365% APR moneylender circling (90 days).

## Hackathon theme fit

- **#2 — long-horizon planning / instruction following.** The agent isn’t judged on one clever turn; it has to remember last week’s rent, today’s BNPL, tomorrow’s EMI, and next week’s fever.
- **#3.1 — economic / world modeling.** Paydays jitter, micro-spend is stochastic, interest compounds, and the "instant cash" offer advertises a label that undersells the APR printed in the same observation.

## The training stack (and what's honest about it)

- **SFT.** Behavioral cloning on short trajectories rolled out by a **hand-coded** heuristic (the same one behind the `Fill baseline action` button in the demo UI). The model learns the action string format and a sane opening repertoire.
- **GRPO** in TRL on Unsloth-quantized **Qwen2.5-7B-Instruct**. Each row has a stored prefix of expert actions; the trainer **replays** that prefix on the same `(task_id, seed)` to deterministically rebuild the observation, then **strictly** parses the model’s single action string (no heuristic fill-in if parsing fails), runs **one** `env.step`, and uses the scaled **`_last_breakdown['total']`** as the per-row scalar. The `end-of-episode` bonus is folded into that scalar on the last day, so the model does see the grader’s tail.
- We do **not** use `grade_episode` inside GRPO — that’s the *headline* bar-chart metric, and optimizing it directly would collapse signal into a single scalar per episode. The dense return is what the optimizer actually moves.

## Evidence — the honest table

Short version, reproducible in a few seconds on a laptop (`python -m scripts.eval_heuristic --seeds 60`):

| Policy | easy | medium | hard |
|--------|------|--------|------|
| **heuristic** (rule-based; also the SFT teacher) | **0.999** | **0.694** | **0.423** |
| `random_valid` (do-nothing ∪ pay-bill ∪ pay-minimum) | 0.933 | 0.496 | 0.421 |
| `do_nothing` (skip every day) | 0.578 | 0.295 | 0.316 |

Three things jump out:

1. The heuristic solves **easy** almost perfectly and owns **medium**.
2. On **hard**, the heuristic barely beats a conservative random policy (Δ ≈ 0.002 at n=60). That's on purpose — it keeps the environment genuinely hard for the SFT → GRPO loop.
3. `do_nothing` is *badly* penalized everywhere. The env doesn’t reward drift.

The **env ablation** (`python -m scripts.ablation_env --seeds 40`) confirms the mechanics bind. On hard, disabling the medical emergency lifts the heuristic score by **+0.17**; disabling UPI micro-spend lifts it by **+0.06**; disabling Diwali festive-pressure by **+0.02**.

The headline training comparison (SFT and GRPO vs heuristic across difficulties) is `before_after_scores_7b.png` in the repo, with the SFT loss curve in `training_loss_7b.png`.

## What I’d change tomorrow

- Add a **seed-matched** comparison plot (same `(task_id, seed)` on x, model − heuristic Δ on y). The current bar chart is averaged; a matched plot would tell judges exactly *where* the learned policy wins or loses.
- Promote a stronger scripted baseline — greedy by debt-APR — so the SFT teacher isn’t the ceiling. The fact that `random_valid` is close to the heuristic on hard is partly a statement about how much room there is above it.
- Sample emergency arrival times and moneylender APRs inside the task config rather than hard-coding scalars; that stops seed-flat grader outputs on easy / medium and gives GRPO more signal to optimize against.
- Publish the LoRA adapters as a separate **Model Hub** repo (not bundled in the Space). The Space stays the runnable environment; the weights live next to a clear evaluation script.

## Ethics and data

Every rupee in this simulation is fictional. The APRs, Diwali social-cost magnitudes, informal-loan rates, and UPI category distributions are **calibrated** against published Indian-finance statistics cited in [`tasks.py`](tasks.py), not fit to any real household panel. The environment is a decision-making *study* — it doesn’t hand out financial advice, and it shouldn’t be treated as a product.

## Links

- **Live Space (runnable environment + `/demo/` UI):** <https://huggingface.co/spaces/indra-dhanush/financial-triage-env>
- **Colab training notebook:** <https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb>
- **Rubric map + re-run commands:** [`README.md`](README.md)
- **OpenEnv:** <https://huggingface.co/openenv>
