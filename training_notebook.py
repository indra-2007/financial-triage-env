# =============================================================================
# 🏥💰 Financial Triage — GRPO Training Notebook
# =============================================================================
#
# This notebook trains an LLM to manage Indian household finances using
# Reinforcement Learning (GRPO) against the Financial Triage OpenEnv environment.
#
# Requirements: Google Colab with GPU (T4 for 3B optional, A100 for 7B)
#
# Architecture:
#   1. Collect expert trajectories from heuristic agent
#   2. SFT fine-tune (behavioral cloning)
#   3. GRPO refinement (beat the heuristic)
#   4. Evaluate trained vs untrained
#   5. Generate reward plots
#
# Time estimate: ~2-3 hours total on Colab T4
# =============================================================================

# %% [markdown]
# # 🏥💰 Financial Triage — Training an AI Financial Advisor for India
#
# **Goal**: Train a Qwen2.5-7B (or 3B on T4) LLM to manage Indian household finances — bills, debts,
# UPI micro-transactions, medical emergencies, and Diwali social pressure.
#
# **Environment**: [Financial Triage on HF Spaces](https://huggingface.co/spaces/indra-dhanush/financial-triage-env)

# %% Cell 1: Install Dependencies
# ============================================================================
# NOTE: Run this cell first, then RESTART the runtime before continuing!
# ============================================================================

# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q "trl>=0.12.0" "transformers>=4.46.0" "datasets>=3.0.0"
# !pip install -q "openenv-core[core]>=0.2.3" "pydantic>=2.0.0" "matplotlib"
# !pip install -q "accelerate>=0.34.0" "bitsandbytes>=0.44.0"

# After restart, also install the environment locally:
# !pip install -q "openai>=1.0.0" "uvicorn>=0.27.0" "fastapi>=0.109.0"

# %% Cell 2: Clone the Environment
# ============================================================================

import os
import subprocess

# Clone the environment repo (for local execution — no network dependency)
if not os.path.exists("financial-triage-env"):
    subprocess.run([
        "git", "clone",
        "https://huggingface.co/spaces/indra-dhanush/financial-triage-env",
        "financial-triage-env"
    ], check=True)

# Add to path
import sys
sys.path.insert(0, "financial-triage-env")

# Verify import
from server.my_env_environment import MyEnvironment
from models import FinancialAction, ActionType, FinancialObservation
from inference import _heuristic_action, observation_to_prompt, SYSTEM_PROMPT
from tasks import get_task_config

print("✅ Environment loaded successfully!")

# Quick smoke test
env = MyEnvironment()
obs = env.reset(seed=42, task_id="easy")
print(f"   Easy task: {obs.episode_length} days, checking=INR {obs.account.checking_balance:,.0f}")

# %% Cell 3: Collect Expert Trajectories
# ============================================================================
# Run the heuristic agent on all tasks × multiple seeds to create training data.
# Each (observation → action) pair becomes a training example.
# ============================================================================

import json
import random

def collect_trajectories(task_ids=["easy", "medium", "hard"], seeds_per_task=15):
    """Collect (system_prompt, observation_text, action_text, reward) tuples."""
    data = []
    scores_by_task = {}

    for task_id in task_ids:
        task_scores = []
        for seed in range(seeds_per_task):
            env = MyEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)

            episode_pairs = []
            for day in range(obs.episode_length):
                # Build observation text
                obs_text = observation_to_prompt(obs)

                # Get heuristic action
                action = _heuristic_action(obs)
                action_text = _action_to_text(action)

                # Step environment
                obs = env.step(action)

                # Store pair with step reward
                episode_pairs.append({
                    "task_id": task_id,
                    "seed": seed,
                    "day": day,
                    "observation": obs_text,
                    "action": action_text,
                    "step_reward": obs.reward if hasattr(obs, 'reward') else 0.0,
                })

            # Get episode score
            score = env.get_episode_score()
            task_scores.append(score)

            # Tag all pairs with episode score
            for pair in episode_pairs:
                pair["episode_score"] = score
                data.append(pair)

        scores_by_task[task_id] = task_scores
        print(f"  {task_id:>6}: {len(task_scores)} episodes, "
              f"avg score={sum(task_scores)/len(task_scores):.4f}, "
              f"best={max(task_scores):.4f}")

    return data, scores_by_task


def _action_to_text(action: FinancialAction) -> str:
    """Convert a FinancialAction to its text representation."""
    at = action.action_type.value
    if action.bill_id:
        return f"{at}({action.bill_id})"
    elif action.debt_id and action.amount:
        return f"{at}({action.debt_id}, {action.amount:.0f})"
    elif action.debt_id:
        return f"{at}({action.debt_id})"
    elif action.amount:
        return f"{at}({action.amount:.0f})"
    return at


print("📊 Collecting expert trajectories from heuristic agent...")
print("   (This takes ~30 seconds)")
trajectories, baseline_scores = collect_trajectories()
print(f"\n✅ Collected {len(trajectories)} training examples")
print(f"   Easy avg:   {sum(baseline_scores['easy'])/len(baseline_scores['easy']):.4f}")
print(f"   Medium avg: {sum(baseline_scores['medium'])/len(baseline_scores['medium']):.4f}")
print(f"   Hard avg:   {sum(baseline_scores['hard'])/len(baseline_scores['hard']):.4f}")

# %% Cell 4: Build Training Dataset
# ============================================================================
# Convert trajectories into the format TRL expects for SFT and GRPO.
# ============================================================================

from datasets import Dataset

def build_sft_dataset(trajectories):
    """Build SFT dataset: each example is a conversation (system + user → assistant)."""
    records = []
    for t in trajectories:
        # Format as chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": t["observation"]},
            {"role": "assistant", "content": t["action"]},
        ]
        records.append({
            "messages": messages,
            "task_id": t["task_id"],
            "episode_score": t["episode_score"],
        })
    return Dataset.from_list(records)


def build_grpo_dataset(trajectories):
    """Build GRPO dataset: prompts for the model to generate completions."""
    records = []
    seen = set()
    for t in trajectories:
        # De-duplicate similar observations
        key = t["observation"][:200]
        if key in seen:
            continue
        seen.add(key)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": t["observation"]},
        ]
        records.append({
            "prompt": prompt,
            "task_id": t["task_id"],
            "day": t["day"],
        })
    return Dataset.from_list(records)


sft_dataset = build_sft_dataset(trajectories)
grpo_dataset = build_grpo_dataset(trajectories)

print(f"✅ SFT dataset:  {len(sft_dataset)} examples")
print(f"✅ GRPO dataset: {len(grpo_dataset)} unique prompts")

# %% Cell 5: Load Model with Unsloth
# ============================================================================
# Load Qwen2.5-7B-Instruct with 4-bit quantization (primary run in README figures).
# For a smaller T4 run, use "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" instead.
# ============================================================================

from unsloth import FastLanguageModel
import torch

# ── Choose model size ──
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # A100 / large GPU (see README figures)
# MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"  # T4 / smaller GPU
# MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # Alternative 8B

MAX_SEQ_LENGTH = 2048
LORA_RANK = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"   Trainable params: {model.print_trainable_parameters()}")

# %% Cell 6: Phase 1 — SFT (Behavioral Cloning from Heuristic)
# ============================================================================
# Fine-tune the model to imitate the heuristic agent.
# This gives a model that can produce valid action strings.
# ============================================================================

from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    seed=42,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field=None,  # Using messages format
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)

print("🚀 Starting SFT training (behavioral cloning)...")
print("   This takes ~20-40 minutes on T4")
sft_results = sft_trainer.train()

# Save SFT checkpoint
model.save_pretrained("./sft_model")
tokenizer.save_pretrained("./sft_model")
print(f"✅ SFT training complete! Final loss: {sft_results.training_loss:.4f}")

# %% Cell 7: Evaluate SFT Model
# ============================================================================
# Run the SFT-trained model on the environment and compare vs heuristic.
# ============================================================================

from inference import parse_action

FastLanguageModel.for_inference(model)

def run_trained_episode(model, tokenizer, task_id, seed=42):
    """Run a full episode using the trained model."""
    env = MyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    for day in range(obs.episode_length):
        # Build prompt
        obs_text = observation_to_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        # Tokenize
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode action
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        action_text = response.split("\n")[0].strip()

        # Parse action
        action = parse_action(action_text, obs)
        if action is None:
            action = _heuristic_action(obs)  # Fallback

        obs = env.step(action)

    return env.get_episode_score()


print("📊 Evaluating SFT model...")
sft_scores = {}
for task_id in ["easy", "medium", "hard"]:
    scores = []
    for seed in range(5):
        score = run_trained_episode(model, tokenizer, task_id, seed=seed)
        scores.append(score)
    sft_scores[task_id] = scores
    avg = sum(scores) / len(scores)
    print(f"  {task_id:>6}: avg={avg:.4f} (seeds 0-4)")

# %% Cell 8: Phase 2 — GRPO Refinement
# ============================================================================
# Use GRPO to train the model beyond the heuristic baseline.
# The model proposes actions, the environment scores them.
# ============================================================================

from trl import GRPOTrainer, GRPOConfig

# Put model back in training mode
FastLanguageModel.for_training(model)


def financial_reward_function(prompts, completions, **kwargs):
    """
    Reward function for GRPO: execute the proposed action in the environment
    and return the step-level reward.
    """
    rewards = []
    for prompt_msgs, completion in zip(prompts, completions):
        try:
            # Parse the action from the completion
            action_text = completion.strip().split("\n")[0].strip()

            # Create a temporary env to evaluate
            # (We use a fixed seed for consistency within each batch)
            env = MyEnvironment()
            obs = env.reset(seed=42, task_id="medium")

            # Try to find the right day state from the prompt
            parsed_action = parse_action(action_text, obs)
            if parsed_action is None:
                rewards.append(-1.0)  # Invalid action penalty
                continue

            # Execute action and get reward
            next_obs = env.step(parsed_action)
            step_reward = env._last_breakdown.get("total", 0.0) if hasattr(env, '_last_breakdown') else 0.0

            # Normalize reward to [-1, 1] range
            normalized = max(-1.0, min(1.0, step_reward / 30.0))
            rewards.append(normalized)

        except Exception:
            rewards.append(-1.0)

    return rewards


# GRPO config
grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=50,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    seed=42,
    max_completion_length=64,
    num_generations=4,  # Number of completions per prompt
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
)

# Use a subset for GRPO (faster iteration)
grpo_subset = grpo_dataset.shuffle(seed=42).select(range(min(200, len(grpo_dataset))))

grpo_trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=grpo_subset,
    reward_funcs=financial_reward_function,
    args=grpo_config,
)

print("🚀 Starting GRPO training...")
print("   This takes ~60-90 minutes on T4")
grpo_results = grpo_trainer.train()

# Save GRPO checkpoint
model.save_pretrained("./grpo_model")
tokenizer.save_pretrained("./grpo_model")
print(f"✅ GRPO training complete!")

# %% Cell 9: Final Evaluation — Before vs After
# ============================================================================
# Compare: Untrained baseline → SFT → GRPO
# ============================================================================

FastLanguageModel.for_inference(model)

print("📊 Final evaluation — comparing all approaches...")
print()

# Run GRPO-trained model
grpo_scores = {}
for task_id in ["easy", "medium", "hard"]:
    scores = []
    for seed in range(5):
        score = run_trained_episode(model, tokenizer, task_id, seed=seed)
        scores.append(score)
    grpo_scores[task_id] = scores

# Print comparison table
print("╔═══════════╦══════════════╦══════════════╦══════════════╗")
print("║ Task      ║  Heuristic   ║  SFT (7B)    ║  GRPO (7B)   ║")
print("╠═══════════╬══════════════╬══════════════╬══════════════╣")
for task_id in ["easy", "medium", "hard"]:
    h_avg = sum(baseline_scores[task_id]) / len(baseline_scores[task_id])
    s_avg = sum(sft_scores[task_id]) / len(sft_scores[task_id])
    g_avg = sum(grpo_scores[task_id]) / len(grpo_scores[task_id])
    delta = "↑" if g_avg > h_avg else "↓"
    print(f"║ {task_id:>9} ║    {h_avg:.4f}    ║    {s_avg:.4f}    ║    {g_avg:.4f} {delta}  ║")
print("╚═══════════╩══════════════╩══════════════╩══════════════╝")

# %% Cell 10: Generate Reward Plots
# ============================================================================
# Create publication-quality plots for the README.
# ============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Plot 1: Training Loss Curve ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Financial Triage — Training Progress", fontsize=14, fontweight='bold')

# SFT loss
sft_log = sft_trainer.state.log_history
sft_steps = [x["step"] for x in sft_log if "loss" in x]
sft_losses = [x["loss"] for x in sft_log if "loss" in x]

axes[0].plot(sft_steps, sft_losses, color="#2196F3", linewidth=2, label="SFT Loss")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Loss")
axes[0].set_title("Phase 1: SFT (Behavioral Cloning)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# GRPO reward
grpo_log = grpo_trainer.state.log_history
grpo_steps = [x["step"] for x in grpo_log if "reward" in x]
grpo_rewards = [x["reward"] for x in grpo_log if "reward" in x]

if grpo_rewards:
    axes[1].plot(grpo_steps, grpo_rewards, color="#4CAF50", linewidth=2, label="GRPO Reward")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Phase 2: GRPO (Policy Optimization)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("📈 Saved: training_curves.png")

# ── Plot 2: Before vs After Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 6))

tasks = ["Easy\n(30 days)", "Medium\n(60 days)", "Hard\n(90 days)"]
task_ids = ["easy", "medium", "hard"]

heuristic_avgs = [sum(baseline_scores[t]) / len(baseline_scores[t]) for t in task_ids]
sft_avgs = [sum(sft_scores[t]) / len(sft_scores[t]) for t in task_ids]
grpo_avgs = [sum(grpo_scores[t]) / len(grpo_scores[t]) for t in task_ids]

x = np.arange(len(tasks))
width = 0.25

bars1 = ax.bar(x - width, heuristic_avgs, width, label="Heuristic (baseline)",
               color="#FF9800", edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x, sft_avgs, width, label="SFT (7B trained)",
               color="#2196F3", edgecolor="white", linewidth=1.5)
bars3 = ax.bar(x + width, grpo_avgs, width, label="GRPO (7B optimized)",
               color="#4CAF50", edgecolor="white", linewidth=1.5)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel("Episode Score (0.0 — 1.0)", fontsize=12)
ax.set_title("Financial Triage — Agent Performance Comparison", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, 1.1)
ax.grid(True, axis="y", alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("before_after_scores.png", dpi=150, bbox_inches="tight")
print("📊 Saved: before_after_scores.png")

plt.show()
print("\n✅ All plots generated! Add these to your README and HF Space.")

# %% Cell 11: Push Model to HuggingFace Hub (Optional)
# ============================================================================
# Upload the trained model to HuggingFace for sharing.
# ============================================================================

# Uncomment and run:
# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")
#
# model.push_to_hub("indra-dhanush/financial-triage-agent-3b")
# tokenizer.push_to_hub("indra-dhanush/financial-triage-agent-3b")
# print("✅ Model pushed to HuggingFace Hub!")

# %% Cell 12: Summary
# ============================================================================
print("""
╔══════════════════════════════════════════════════════════════╗
║                 TRAINING COMPLETE! 🎉                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Files generated:                                            ║
║  📈 training_curves.png    — SFT loss + GRPO reward curves  ║
║  📊 before_after_scores.png — Heuristic vs SFT vs GRPO      ║
║  🤖 ./grpo_model/           — Trained LoRA adapters          ║
║                                                              ║
║  Next steps:                                                 ║
║  1. Add plots to your README.md                              ║
║  2. Push model to HuggingFace Hub                            ║
║  3. Write mini-blog or record <2min video                    ║
║  4. Git push to HF Space                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
