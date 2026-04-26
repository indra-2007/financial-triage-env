# Judges — one-page brief

One page. If anything here surprises you, the README has the long version and the scripts re-emit the numbers.

## The thing in one sentence

A day-by-day, **INR**-denominated household-finance simulator on **OpenEnv**: the agent picks one of eleven actions per day across 30 / 60 / 90-day episodes, the environment returns a dense per-step reward, and a deterministic grader returns a single **0–1** score at the end.

- **Live Space + API (submit this URL):** <https://huggingface.co/spaces/indra-dhanush/financial-triage-env>
- **Browsers get the interactive UI at `/demo/` automatically** (same origin, stateful session).
- **Colab (Unsloth + TRL, SFT → GRPO):** [notebook](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb)
- **Mini blog:** [`MINI_BLOG.md`](MINI_BLOG.md)

## Rubric map (your weights)

| Weight | Criterion | Strongest evidence in this repo |
|--------|-----------|---------------------------------|
| **40%** | **Innovation** | 14-term dense reward in `server/my_env_environment.py::_compute_reward`; mechanical anti-gaming (consecutive do-nothing streak penalty; no savings-growth credit on same-day withdraw + re-deposit; 7-day action-diversity bonus); stochastic UPI micro-spend and P2P pressure; a loan-shark offer whose label undersells its APR; three difficulty-specific graders in `tasks.py`; **ablation in plain text below**. |
| **30%** | **Story** | This brief + [`README.md`](README.md) + [`MINI_BLOG.md`](MINI_BLOG.md) + the `/demo/` UI on the Space (reset → step → grade live). |
| **20%** | **Training evidence** | `training_loss_7b.png` (SFT loss vs step), `before_after_scores_7b.png` (heuristic / SFT / GRPO bars from notebook). **Multi-seed supplement:** `heuristic_scores_ci.png` + `heuristic_scores.json` (n=60, 95% bootstrap CI); **ablation:** `ablation_env.png` + `ablation_env.json`. |
| **10%** | **Reward + pipeline** | README section "How step reward is shaped" names every term. GRPO **replays stored expert `prefix_actions`** to reconstruct the (task_id, seed, day) observation, **strictly parses** the model action (no heuristic fill-in), runs **one** `env.step`, and optimizes the scaled dense return — **not** `grade_episode`. Headline figures use `grade_episode`. |

## Headline numbers (n=60 seeds, 95% bootstrap CI)

Produced by `python -m scripts.eval_heuristic --seeds 60`; raw per-seed scores in `heuristic_scores.json`.

| Policy | easy | medium | hard |
|--------|------|--------|------|
| **heuristic** (also the SFT data source and `Fill baseline` button) | **0.999** | **0.694** | **0.423** ± CI [0.417, 0.428] |
| `random_valid` (do-nothing ∪ pay bill ∪ pay minimum) | 0.933 | 0.496 | 0.421 |
| `do_nothing` (null baseline) | 0.578 | 0.295 | 0.316 |

## Environment ablation — the mechanics bind

`python -m scripts.ablation_env --seeds 40`; raw in `ablation_env.json`. Hard task, heuristic policy, 95% bootstrap CI:

| Ablation | Mean score | Δ vs `full` |
|----------|------------|-------------|
| **full** (shipping env) | **0.421** | — |
| `no_medical_emergency` | 0.594 | **+0.17** |
| `no_upi_micro_spend` | 0.483 | **+0.06** |
| `no_festival` | 0.444 | +0.02 |

If scores rise when a mechanic is switched off, that mechanic is binding on the baseline.

## 90-second demo path for a reviewer

1. Open the Space URL. You land on `/demo/`.
2. Pick **hard**, **seed 7**, click **Reset**.
3. Click **Fill baseline action** → **Step** a few days. Watch the **Last reward** change and the **Daily summary** log.
4. Hit **Reset** again with **easy**, **seed 0**, step to the end, and read off **Grade (0–1)**.
5. (Optional) `POST /api/demo/reset` from curl / an OpenEnv client to confirm the stateful JSON API is real.

## How to re-run **every** claim

```bash
pip install -r requirements.txt matplotlib numpy
python -m scripts.eval_heuristic --seeds 60     # numbers + figure + json
python -m scripts.ablation_env --seeds 40       # numbers + figure + json
python -m server.video_demo_server              # same UI as /demo/ on the Space
```

## Honest limitations

- The heuristic vs `random_valid` gap on **hard** is only ~0.002 at n=60 — our rule-based teacher is not optimal there, which is exactly what makes the env a useful training target for GRPO.
- Grader scores on **easy / medium** flatten across seeds in the low-digit rounded form; day-to-day details still vary, so use the **dense per-step return** for seed-level comparisons (that is what GRPO optimizes).
- Economic magnitudes (APRs, Diwali pressure, UPI leak rates) are **calibrated** from published Indian-finance statistics but not fit to a real panel.

If anything above is unclear: the README has the full context, the scripts re-emit every number, and the `/demo/` UI lets you feel the same env a model saw during training.
