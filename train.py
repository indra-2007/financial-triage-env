"""Financial Triage — 7B Training Script (A10G)"""
import os, sys
sys.path.insert(0, '.')

# --- Clone env if needed ---
if not os.path.exists('financial-triage-env'):
    os.system('git clone https://huggingface.co/spaces/indra-dhanush/financial-triage-env')
sys.path.insert(0, 'financial-triage-env')

from server.my_env_environment import MyEnvironment
from inference import _heuristic_action, observation_to_prompt, parse_action, SYSTEM_PROMPT

# --- Baseline ---
print("="*50)
print("STEP 1: Baseline scores")
print("="*50)
baseline_scores = {}
for task_id in ['easy', 'medium', 'hard']:
    scores = []
    for seed in range(5):
        env = MyEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        for _ in range(obs.episode_length):
            obs = env.step(_heuristic_action(obs))
        scores.append(env.get_episode_score())
    baseline_scores[task_id] = scores
    print(f'  {task_id:>6} baseline: {sum(scores)/len(scores):.4f}')

# --- Load model ---
print("\n" + "="*50)
print("STEP 2: Loading 7B model")
print("="*50)
from unsloth import FastLanguageModel
import torch

model_name = 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name, max_seq_length=1024, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=16, lora_dropout=0, bias='none',
)
print(f'✅ {model_name}')
model.print_trainable_parameters()

# --- SFT ---
print("\n" + "="*50)
print("STEP 3: SFT Warmup")
print("="*50)
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

raw = load_dataset('json', data_files='financial-triage-env/sft_dataset.jsonl', split='train')
print(f'SFT dataset: {len(raw)} examples')

def convert(example):
    text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {'text': text}

dataset = raw.map(convert, remove_columns=raw.column_names, num_proc=1)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=1024,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=80,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed=3407,
        output_dir='outputs/sft',
    ),
)

sft_stats = trainer.train()
model.save_pretrained('outputs/sft')
tokenizer.save_pretrained('outputs/sft')
print(f'✅ SFT done! Loss: {sft_stats.training_loss:.4f}')

# --- Evaluate SFT ---
print("\n" + "="*50)
print("STEP 4: Evaluating SFT model")
print("="*50)
FastLanguageModel.for_inference(model)

def run_trained_episode(model, tokenizer, task_id, seed=42):
    env = MyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for day in range(obs.episode_length):
        obs_text = observation_to_prompt(obs)
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': obs_text}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=64, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip().split('\n')[0]
        action = parse_action(response, obs)
        if action is None:
            action = _heuristic_action(obs)
        obs = env.step(action)
    return env.get_episode_score()

sft_scores = {}
for task_id in ['easy', 'medium', 'hard']:
    scores = [run_trained_episode(model, tokenizer, task_id, seed=s) for s in range(5)]
    sft_scores[task_id] = scores
    print(f'  {task_id:>6} SFT: {sum(scores)/len(scores):.4f}  (baseline: {sum(baseline_scores[task_id])/len(baseline_scores[task_id]):.4f})')

# --- GRPO ---
print("\n" + "="*50)
print("STEP 5: GRPO Training")
print("="*50)
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

FastLanguageModel.for_training(model)

prompts = []
for task_id in ['easy', 'medium', 'hard']:
    for seed in range(3):
        env = MyEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        for day in range(min(obs.episode_length, 20)):
            obs_text = observation_to_prompt(obs)
            prompts.append({'prompt': [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':obs_text}]})
            obs = env.step(_heuristic_action(obs))

grpo_dataset = Dataset.from_list(prompts)
print(f'GRPO dataset: {len(grpo_dataset)} prompts')

def reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            action_text = completion[-1]['content'] if completion else ''
        else:
            action_text = str(completion)
        action_text = action_text.strip().split('\n')[0].strip()
        env = MyEnvironment()
        obs = env.reset(seed=42, task_id='medium')
        action = parse_action(action_text, obs)
        if action is None:
            rewards.append(-1.0)
            continue
        next_obs = env.step(action)
        r = 0.3
        if env._checking >= 0: r += 0.3
        if 'ON TIME' in (next_obs.daily_summary or ''): r += 0.4
        if 'informal' in action_text: r -= 0.5
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards

grpo_trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=grpo_dataset,
    reward_funcs=reward_fn,
    args=GRPOConfig(
        output_dir='outputs/grpo',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=5,
        max_completion_length=64,
        num_generations=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim='adamw_8bit',
        report_to='none',
    ),
)

grpo_stats = grpo_trainer.train()
model.save_pretrained('outputs/grpo')
tokenizer.save_pretrained('outputs/grpo')
print('✅ GRPO done!')

# --- Final eval ---
print("\n" + "="*50)
print("STEP 6: Final Evaluation")
print("="*50)
FastLanguageModel.for_inference(model)
grpo_scores = {}
for task_id in ['easy', 'medium', 'hard']:
    scores = [run_trained_episode(model, tokenizer, task_id, seed=s) for s in range(5)]
    grpo_scores[task_id] = scores

print('\n' + '='*55)
print(f'{"Task":>8} | {"Heuristic":>10} | {"SFT (7B)":>10} | {"GRPO (7B)":>10}')
print('-'*55)
for t in ['easy','medium','hard']:
    h = sum(baseline_scores[t])/len(baseline_scores[t])
    s = sum(sft_scores[t])/len(sft_scores[t])
    g = sum(grpo_scores[t])/len(grpo_scores[t])
    print(f'{t:>8} | {h:>10.4f} | {s:>10.4f} | {g:>10.4f}')
print('='*55)

# --- Generate plots ---
print("\n" + "="*50)
print("STEP 7: Generating plots")
print("="*50)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sft_log = trainer.state.log_history
steps = [x['step'] for x in sft_log if 'loss' in x]
losses = [x['loss'] for x in sft_log if 'loss' in x]
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(steps, losses, color='#2196F3', linewidth=2)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('SFT Training Loss (7B)'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('training_loss.png', dpi=150)
print('📈 Saved training_loss.png')

fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(3); w = 0.25
h_avg = [sum(baseline_scores[t])/len(baseline_scores[t]) for t in ['easy','medium','hard']]
s_avg = [sum(sft_scores[t])/len(sft_scores[t]) for t in ['easy','medium','hard']]
g_avg = [sum(grpo_scores[t])/len(grpo_scores[t]) for t in ['easy','medium','hard']]
ax.bar(x-w, h_avg, w, label='Heuristic', color='#FF9800')
ax.bar(x, s_avg, w, label='SFT (7B)', color='#2196F3')
ax.bar(x+w, g_avg, w, label='GRPO (7B)', color='#4CAF50')
for i in range(3):
    for v,dx in [(h_avg[i],-w),(s_avg[i],0),(g_avg[i],w)]:
        ax.text(i+dx, v+0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Easy (30d)','Medium (60d)','Hard (90d)'])
ax.set_ylabel('Score (0-1)'); ax.set_ylim(0,1.1)
ax.set_title('Financial Triage — 7B Agent Performance'); ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('before_after_scores.png', dpi=150)
print('📊 Saved before_after_scores.png')

print("\n✅ ALL DONE! Download training_loss.png and before_after_scores.png")
