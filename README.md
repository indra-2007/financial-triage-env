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
### Long-horizon household finance in INR—under OpenEnv, with the receipts.

**OpenEnv · SFT + GRPO · Qwen2.5-7B · Hugging Face Space**

[![Space](https://img.shields.io/badge/🤗%20Space-indra--dhanush%2Ffinancial--triage--env-yellow)](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb)
[![OpenEnv](https://img.shields.io/badge/openenv--core-≥%200.2.3-blue)](https://pypi.org/project/openenv-core/)

</div>

**Financial Triage** is a Gym-style simulator: one day per `step`, one financial action, dense rewards, and a terminal grader in \([0,1]\). The model does not summarize your spending—it **allocates** under stochastic income, UPI leakage, and credit traps that are easy to **describe** in a slide and harder to **survive** in code. Training follows the usual stack: **heuristic → SFT → GRPO** (TRL) on **Qwen2.5-7B** in 4-bit (Unsloth), deployed on this **Space**.

**Pipeline:** `Environment` → per-step **verifier-style reward** + **episode grader** → **SFT** (expert rollouts) → **GRPO** (live rollouts) → **`openenv-core[core] ≥ 0.2.3`**

---

## At a glance

| Section | Contents |
|--------|----------|
| [Loop & diagram](#the-environment-loop) | `reset` / `step` / grading, with flowchart |
| [Theme fit](#theme-alignment) | Long horizon (#2) and economic world modeling (#3.1) |
| [Rubric map](#rubric-map) | Where innovation, results, and pipeline land in the repo |
| [Results](#training--results) | SFT loss and heuristic vs. SFT vs. GRPO bars (7B) |
| [Resources](#resources) | Space, Colab, write-up, optional video URL |

---

## Resources

| Item | Link |
|------|------|
| **Space** | [huggingface.co/spaces/indra-dhanush/financial-triage-env](https://huggingface.co/spaces/indra-dhanush/financial-triage-env) |
| **Training** | [Open in Colab](https://colab.research.google.com/github/indra-2007/financial-triage-env/blob/main/financial_triage_training.ipynb) · [`financial_triage_training.ipynb`](financial_triage_training.ipynb) |
| **Write-up** | [`MINI_BLOG.md`](MINI_BLOG.md) · [on Hub](https://huggingface.co/spaces/indra-dhanush/financial-triage-env/blob/main/MINI_BLOG.md) |
| **Video** | *Optional:* add a short (≤2 min) public URL; host off-repo—no large media in the Space bundle. |

---

## The environment loop

Standard **OpenEnv** contract: `reset(task_id)` → `step(action)` until termination → **scalar episode score** in \([0,1]\). **One transition = one calendar day**; the policy emits **a single** action from **11** candidates (bills, debt service, savings moves, three loan types including informal/festive, negotiation, or abstention).

### Flowchart

<p align="center">
  <img src="flowchart.png" alt="OpenEnv loop: reset, observation, one of eleven actions, one-day step with dynamics, repeat or grade" width="100%" />
</p>

*Daily loop: full observation (accounts, instruments, risk, text summary) → one action → state update (income, UPI, interest, penalties, events) → step reward; at episode end, the grader applies a difficulty-specific score.*

<details>
<summary><strong>Mermaid (GitHub / compatible viewers)</strong></summary>

```mermaid
flowchart LR
    A([reset(task_id)]) --> B[Observation]
    B --> C[Policy: one of 11 actions]
    C --> D[env.step: one day]
    D --> E{Terminal?}
    E -->|no| B
    E -->|yes| F[Grader: 0.0–1.0]
```

</details>

**Implementation:** dynamics in `server/my_env_environment.py`; task specs and `grade_episode` in `tasks.py`; manifest in [`openenv.yaml`](openenv.yaml).

---

## Theme alignment

| Track | Relevance |
|-------|-----------|
| **#2 Long-horizon planning** | 30-, 60-, and 90-day episodes; obligations and shocks compound. Not a one-shot QA benchmark. |
| **#3.1 Professional / economic world modeling** | Partial information (e.g. advertised vs. realized credit terms), **stochastic** cash flows, **objective** solvency and default logic—no “does it feel reasonable?” as the only metric. |

Out of scope for this build: **#1** multi-agent coalition games, **#4** self-generating task curricula—here, a **single** agent, **one** economy, **reproducible** scoring.

---

## Rubric map

| Criterion (weight) | Evidence in this submission |
|--------------------|------------------------------|
| **Innovation (40%)** | Domain-specific setting (INR, UPI, informal credit), 11 action types, multi-term reward, anti–reward-hacking design |
| **Story & presentation (30%)** | This README, [`MINI_BLOG.md`](MINI_BLOG.md) |
| **Observable improvement (20%)** | [`training_loss_7b.png`](#training--results) · [`before_after_scores_7b.png`](#training--results) (heuristic, SFT, GRPO) |
| **Reward & pipeline (10%)** | Reward sketch below; SFT on traces + GRPO in **live** env (see notebook) |

**Submission surface:** OpenEnv + Colab + plots committed here + `MINI_BLOG` linked; Space is the runnable artifact.

---

## Why this environment

Most people never **practice** triage under income volatility, not **read** about it. A chart cannot foreclose a predatory line item; a policy in this simulator can, and **failure** shows up in balances and the grader, not in a footnote. We calibrate the narrative to **public** Indian macro and payments statistics; everything inside the sim is **synthetic** and **executable**.

---

## Task tiers

| | **Easy (30d)** | **Medium (60d)** | **Hard (90d)** |
|---|----------------|------------------|----------------|
| **Income** | Steady salaried | Salaried + freelance | **Job loss → gig** |
| **Debt** | None | Card, EMI, BNPL, predatory offer | **Multi-tranche** stress |
| **Shocks** | UPI noise | Health + informality | **Stacked** emergencies + festival |
| **`task_id`** | `easy` | `medium` | `hard` |

---

## Observation, action, return

- **State:** checking/savings, bills, debts, risk metrics, optional loan table, active emergency, festival block, and a concise **`daily_summary`**.  
- **Action space:** 11 discrete templates (examples: full/minimum pay, defer, extra debt pay, savings transfer/withdrawal, three loan channels, `negotiate_bill`, `do_nothing`). **One** action per `step`.  
- **Returns:** dense **step** reward (multi-component) + episode **grader** in \([0,1]\) per `tasks.grade_episode`.

---

## Training & results

| Phase | Data / signal | Model (reported) |
|-------|----------------|------------------|
| **SFT** | Trajectories from a **hand-engineered** heuristic; `sft_dataset.jsonl` | **Qwen2.5-7B** 4-bit, Unsloth |
| **GRPO** | TRL; rewards from the **actual** `step` loop, not a frozen offline label set | same backbone |

*Smaller GPUs:* the notebook can swap in 3B; the **published** figures are **7B**.

### SFT loss (7B)

<p align="center"><img src="training_loss_7b.png" alt="SFT training loss vs step, Qwen2.5-7B" width="720" /></p>

*Supervised loss vs. step index—used as the warm-start phase before policy optimization.*

### Heuristic · SFT (7B) · GRPO (7B)

<p align="center"><img src="before_after_scores_7b.png" alt="Mean episode score by task: heuristic, SFT 7B, GRPO 7B" width="720" /></p>

*Heuristic: rule-based **reference policy**. SFT: imitation. GRPO: reinforcement against the environment. Bar labels are on the figure.*

### Heuristic baselines (context)

<p align="center">
  <img src="reward_curves_baseline.png" alt="Heuristic cumulative reward by day" width="48%" />
  <img src="baseline_scores.png" alt="Heuristic mean episode score by difficulty" width="48%" />
</p>

*Diagnostics for the reference agent only; they are **not** the learning curve for the finetuned model.*

---

## Dense reward (multi-term)

| Incentivized | Penalized |
|--------------|-----------|
| Solvency, on-time payment, de-risking high-APR balances, buffer maintenance, non-degenerate action patterns | Delinquency, overdraft, default spiral, interest drag, inaction, carrying exploitative credit |

`server/my_env_environment.py` — `_compute_reward` · `tasks.py` — `grade_*` / `grade_episode`.

---

## Quick start

```bash
git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env
cd financial-triage-env && pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Container:** `docker build -t financial-triage .` → `docker run -p 7860:7860 financial-triage` — liveness: `GET /health`.

**Remote client:**

```python
from openenv import EnvClient
env = EnvClient.from_hub("indra-dhanush/financial-triage-env")
obs = env.reset(task_id="hard")
```

**Note:** Colab training often runs against a **local checkout** of the environment for throughput; the Space remains the **canonical** network endpoint. For a strict **client–server** evaluation path, call the Space URL above.

---

## Manifest & operations

- [`openenv.yaml`](openenv.yaml) — `server.app:app`, **7860**, tasks `easy` · `medium` · `hard`.  
- **Reserved names** for any extra MCP/HTTP tools: do not overload `reset`, `step`, `state`, `close`.  
- **QLoRA / merge:** use Unsloth’s documented save path; re-run a short eval after export.

**License · references:** [MIT](LICENSE) · [OpenEnv](https://huggingface.co/openenv) · [`openenv-core` on PyPI](https://pypi.org/project/openenv-core/) · *Public* statistics (RBI, NSSO, NPCI, NCRB) inform **wording** only; **no** proprietary feeds.

*Auxiliary scripts (optional):* `fix_ipynb.py`, `generate_loss_plot.py`, `test_reward.py` — not on the live server path.
