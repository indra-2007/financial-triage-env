import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('/Users/indra/Downloads/financial_triage_training (1).ipynb', 'r') as f:
    nb = json.load(f)

html_content = ""
for cell in nb.get('cells', []):
    for output in cell.get('outputs', []):
        if 'data' in output and 'text/html' in output['data']:
            html_list = output['data']['text/html']
            html_str = "".join(html_list)
            if 'Training Loss' in html_str:
                html_content = html_str
                break
    if html_content:
        break

if not html_content:
    print("Could not find HTML table with training loss in the notebook.")
    exit(1)

# Parse table
steps = []
losses = []
rows = re.findall(r'<tr>\s*<td>(\d+)</td>\s*<td>([\d.]+)</td>\s*</tr>', html_content)
for s, l in rows:
    steps.append(int(s))
    losses.append(float(l))

if not steps:
    print("Could not extract steps and losses from HTML.")
    exit(1)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(steps, losses, color='#2196F3', linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('SFT Training Loss (7B)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss_7b.png', dpi=150)
print(f"Successfully generated training_loss_7b.png with {len(steps)} steps!")
