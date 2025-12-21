import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
path = r'results/'
rag_df = pd.read_excel(path + 'rageval_ita_results.xlsx')
bm25_df = pd.read_csv(path + 'rageval_ita_bm25_results.csv')

# Filter for k=10 only
rag_k10 = rag_df[rag_df['k'] == 10].copy()
bm25_k10 = bm25_df[bm25_df['k'] == 10].copy()

# Create data for plotting
models_data = []

# Track base model scores for calculating drops
base_scores = {}

# Add base models from RAG results
for idx, row in rag_k10.iterrows():
    model_name = row['model']
    
    # Create clean names
    if 'Qwen3-Embedding-8B' in model_name:
        clean_name = 'Qwen3-Emb-8B'
        model_type = 'base'
    elif 'multilingual-e5-small' in model_name and 'finetuned' not in model_name:
        clean_name = 'ml-e5-small (base)'
        model_type = 'base'
        base_scores['ml-e5-small'] = row['ndcg']
    elif 'multilingual-e5-small' in model_name and 'finetuned' in model_name:
        clean_name = 'ml-e5-small (finetuned)'
        model_type = 'finetuned'
    elif 'e5-small-v2' in model_name and 'finetuned' not in model_name:
        clean_name = 'e5-small-v2 (base)'
        model_type = 'base'
        base_scores['e5-small-v2'] = row['ndcg']
    elif 'e5-small-v2' in model_name and 'finetuned' in model_name:
        clean_name = 'e5-small-v2 (finetuned)'
        model_type = 'finetuned'
    else:
        continue
    
    models_data.append({
        'name': clean_name,
        'ndcg': row['ndcg'],
        'type': model_type
    })

# Add BM25
bm25_score = bm25_k10['ndcg'].values[0]
models_data.append({
    'name': 'BM25',
    'ndcg': bm25_score,
    'type': 'baseline'
})

# Convert to DataFrame for easier manipulation
plot_df = pd.DataFrame(models_data)

# Sort by NDCG descending
plot_df = plot_df.sort_values('ndcg', ascending=False).reset_index(drop=True)

# Calculate percentage drops for fine-tuned models
finetuned_ml_e5_small = plot_df[plot_df['name'] == 'ml-e5-small (finetuned)']['ndcg'].values[0]
finetuned_e5_v2 = plot_df[plot_df['name'] == 'e5-small-v2 (finetuned)']['ndcg'].values[0]

drop_ml_e5 = ((finetuned_ml_e5_small - base_scores['ml-e5-small']) / base_scores['ml-e5-small']) * 100
drop_e5_v2 = ((finetuned_e5_v2 - base_scores['e5-small-v2']) / base_scores['e5-small-v2']) * 100

# Create figure
fig, ax = plt.subplots(figsize=(13, 8))

# Define colors based on type
colors = []
edge_colors = []
for model_type in plot_df['type']:
    if model_type == 'base':
        colors.append('#4A90E2')  # Blue for base models
        edge_colors.append('#2E5C8A')
    elif model_type == 'finetuned':
        colors.append('#E74C3C')  # Red for fine-tuned
        edge_colors.append('#C0392B')
    else:
        colors.append('#95A5A6')  # Grey for baseline
        edge_colors.append('#7F8C8D')

# Create bars
bars = ax.bar(range(len(plot_df)), plot_df['ndcg'], color=colors, 
              edgecolor=edge_colors, linewidth=2.5, alpha=0.85, width=0.7)

# Add value labels on bars
for i, (idx, row) in enumerate(plot_df.iterrows()):
    ax.text(i, row['ndcg'] + 0.008, f"{row['ndcg']:.3f}", 
            ha='center', va='bottom', fontsize=10.5, fontweight='bold')

# Add percentage drop annotations for fine-tuned models
for i, (idx, row) in enumerate(plot_df.iterrows()):
    if 'finetuned' in row['name']:
        if 'ml-e5-small' in row['name']:
            # Add drop annotation
            ax.text(i, row['ndcg'] - 0.02, f'{drop_ml_e5:.1f}%', 
                    ha='center', va='top', fontsize=12, fontweight='bold', 
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#C0392B', 
                             edgecolor='white', linewidth=2))
        elif 'e5-small-v2' in row['name']:
            # Position the annotation higher above the bar
            ax.text(i, row['ndcg'] + 0.05, f'{drop_e5_v2:.1f}%', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', 
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#C0392B', 
                             edgecolor='white', linewidth=2))

# Set x-axis labels
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df['name'], rotation=25, ha='right', fontsize=11.5)

# Labels and title
ax.set_ylabel('NDCG@10', fontsize=15, fontweight='bold')
ax.set_title('Catastrophic Forgetting on Custom Italian RAG Benchmark', 
             fontsize=17, fontweight='bold', pad=20)

# Grid
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Set y-axis limits
ax.set_ylim(0, max(plot_df['ndcg']) * 1.18)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4A90E2', edgecolor='#2E5C8A', linewidth=2.5, label='Base Models'),
    Patch(facecolor='#E74C3C', edgecolor='#C0392B', linewidth=2.5, label='Fine-tuned Models'),
    Patch(facecolor='#95A5A6', edgecolor='#7F8C8D', linewidth=2.5, label='Baseline (BM25)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

# Add horizontal line at BM25 score
ax.axhline(y=bm25_score, color='#27AE60', linestyle=':', linewidth=2.5, alpha=0.6)
ax.text(len(plot_df)-0.3, bm25_score + 0.01, 'BM25 baseline', fontsize=10.5, 
        color='#27AE60', ha='right', fontweight='bold')

# Improve tick label size
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('figures/figure2_catastrophic_forgetting.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure2_catastrophic_forgetting.pdf', bbox_inches='tight')
print("Figure 2 saved as figures/figure2_catastrophic_forgetting.png and .pdf")
plt.close()