"""Stage 1 — Supervised Fine-Tuning of the Qwen2.5-7B-Instruct policy on
heuristic rollouts of the Financial Triage environment.

This is the exact training code that runs inside the Colab notebook's SFT cell,
extracted as a standalone script so a judge can read the pipeline without
opening Jupyter.

Run (GPU required — tested on a single Colab T4):

  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
              "trl==0.11.*" "transformers>=4.44" "datasets" "accelerate" "bitsandbytes"
  export WANDB_API_KEY=...   # optional — mirrors the run to W&B
  python -m scripts.train_sft

Outputs:
  - outputs/sft/                       LoRA adapters + tokenizer
  - outputs/sft/trainer_state.json     per-step loss log (read by plot cell)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DATASET_PATH = str(_ROOT / "sft_dataset.jsonl")
OUTPUT_DIR = str(_ROOT / "outputs" / "sft")


def main() -> int:
    print(f"[train_sft] model={MODEL_NAME}")
    print(f"[train_sft] dataset={DATASET_PATH}")
    print(f"[train_sft] output={OUTPUT_DIR}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"[train_sft] SFT examples: {len(dataset)}")

    def to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

    gc.collect()
    torch.cuda.empty_cache()
    model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=1024,
        dataset_num_proc=1,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    print("[train_sft] starting training…")
    stats = trainer.train()
    trainer.save_state()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train_sft] done — final_loss={stats.training_loss:.4f}")
    print(f"[train_sft] per-step log saved to {OUTPUT_DIR}/trainer_state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
