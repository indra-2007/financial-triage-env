import json

filepath = "/Users/indra/Downloads/financial_triage_training (1).ipynb"
with open(filepath, 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            line = line.replace('SFT (3B)', 'SFT (7B)')
            line = line.replace('GRPO (3B)', 'GRPO (7B)')
            
            # The plotting code
            if "ax.set_title('SFT Training Loss');" in line:
                line = line.replace("'SFT Training Loss'", "'SFT Training Loss (7B)'")
            if "label='SFT'" in line:
                line = line.replace("label='SFT'", "label='SFT (7B)'")
            if "label='GRPO'" in line:
                line = line.replace("label='GRPO'", "label='GRPO (7B)'")
            if "ax.set_title('Financial Triage — Agent Performance')" in line:
                line = line.replace("Agent Performance", "7B Agent Performance")
                
            new_source.append(line)
        cell['source'] = new_source
        
        # Now let's try to fix the plot cell for the `trainer` issue
        # Only modify the cell that has "import matplotlib.pyplot as plt"
        if len(cell.get('source', [])) > 0 and "import matplotlib.pyplot as plt\n" in cell['source']:
            source_text = "".join(cell['source'])
            if "trainer.state.log_history" in source_text and "try:" not in source_text:
                new_plot_source = """import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Plot 1: Training loss
try:
    if 'trainer' in locals():
        sft_log = trainer.state.log_history
    else:
        with open('outputs/sft/trainer_state.json', 'r') as f:
            sft_log = json.load(f)['log_history']
            
    steps = [x['step'] for x in sft_log if 'loss' in x]
    losses = [x['loss'] for x in sft_log if 'loss' in x]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(steps, losses, color='#2196F3', linewidth=2)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Loss')
    ax.set_title('SFT Training Loss (7B)'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('training_loss.png', dpi=150)
    print('📈 Saved training_loss.png')
except Exception as e:
    print(f'⚠️ Could not generate SFT training loss plot: {e}')

# Plot 2: Before vs After
fig, ax = plt.subplots(figsize=(10,5))
tasks = ['Easy (30d)', 'Medium (60d)', 'Hard (90d)']
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
        
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel('Episode Score (0-1)'); ax.set_ylim(0,1.1)
ax.set_title('Financial Triage — 7B Agent Performance'); ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('before_after_scores.png', dpi=150)
print('📊 Saved before_after_scores.png')
plt.show()
"""
                cell['source'] = [line + '\n' for line in new_plot_source.split('\n')]
                # remove the trailing newline from the last line to be clean
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1][:-1]

with open(filepath, 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
