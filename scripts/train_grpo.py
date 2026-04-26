"""Stage 2 — GRPO fine-tuning on top of the SFT LoRA checkpoint.

Reward function does **not** call the episode grader. For each row it:
  1. Replays the stored expert prefix on the row's (task_id, seed) to
     reconstruct the exact day-`d` observation.
  2. Strictly parses the model's single-line action (no heuristic fallback
     — malformed output gets reward -1.0).
  3. Runs one env.step and optimises the scaled dense return from
     `_last_breakdown["total"]`.

This is the exact training code that runs inside the Colab notebook's GRPO
cell, extracted as a standalone script so a judge can read the pipeline
without opening Jupyter.

Run (GPU required — tested on a single Colab T4, after train_sft.py):

  export WANDB_API_KEY=...   # optional — mirrors the run to W&B
  python -m scripts.train_grpo

Outputs:
  - outputs/grpo/                      LoRA adapters on top of SFT
  - outputs/grpo/trainer_state.json    per-step GRPO log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from server.my_env_environment import MyEnvironment
from inference import (
    SYSTEM_PROMPT,
    _heuristic_action,
    format_action,
    observation_to_prompt,
    parse_action_strict,
    replay_expert_prefix,
)


SFT_CHECKPOINT = str(_ROOT / "outputs" / "sft")
OUTPUT_DIR = str(_ROOT / "outputs" / "grpo")
TASKS = ("easy", "medium", "hard")
SEEDS_PER_TASK = 3
DAYS_PER_SEED = 20


def _build_prompts():
    prompts = []
    for task_id in TASKS:
        for seed in range(SEEDS_PER_TASK):
            env = MyEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            prefix: list[str] = []
            for _ in range(min(obs.episode_length, DAYS_PER_SEED)):
                obs_text = observation_to_prompt(obs)
                prompts.append({
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text},
                    ],
                    "task_id": task_id,
                    "seed": seed,
                    "prefix_actions": list(prefix),
                })
                act = _heuristic_action(obs)
                prefix.append(format_action(act))
                obs = env.step(act)
    return Dataset.from_list(prompts)


def _completion_to_action_text(c):
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c)


def reward_fn(prompts, completions, task_id, seed, prefix_actions, **kwargs):
    """Replay expert prefix → parse → one env.step → clipped dense return."""
    out = []
    for c, tid, s, pre in zip(completions, task_id, seed, prefix_actions, strict=True):
        try:
            action_text = _completion_to_action_text(c).strip().split("\n")[0].strip()
            env = MyEnvironment()
            obs = replay_expert_prefix(env, str(tid), int(s), list(pre or []))
            a = parse_action_strict(action_text, obs)
            if a is None:
                out.append(-1.0)
                continue
            env.step(a)
            step_reward = float(env._last_breakdown.get("total", 0.0))
            out.append(max(-1.0, min(1.0, step_reward / 30.0)))
        except Exception:
            out.append(-1.0)
    return out


def main() -> int:
    print(f"[train_grpo] loading SFT checkpoint from {SFT_CHECKPOINT}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_CHECKPOINT,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    dataset = _build_prompts()
    print(f"[train_grpo] prompt rows: {len(dataset)}")

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            logging_steps=5,
            max_completion_length=64,
            num_generations=4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        ),
    )

    print("[train_grpo] starting training…")
    trainer.train()
    trainer.save_state()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train_grpo] done — adapters saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
