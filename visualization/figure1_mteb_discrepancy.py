import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
path = r'results/'
mteb_df = pd.read_excel(path + 'mteb_ita_subset_results.xlsx')

# Calculate Italian-only average (from Belebele and Wiki Italian monolingual)
mteb_df['italian_only_avg'] = (mteb_df['belebele_ita'] + mteb_df['wiki_ita_mono']) / 2

# Create model name mapping for cleaner labels
model_mapping = {
    'intfloat/multilingual-e5-small': 'ml-e5-small',
    'intfloat/multilingual-e5-small (finetuned)': 'ml-e5-small (finetuned)',
    'intfloat/multilingual-e5-base': 'ml-e5-base',
    'intfloat/multilingual-e5-large': 'ml-e5-large',
    'BAAI/bge-m3': 'BGE-M3',
    'Qwen/Qwen3-Embedding-8B': 'Qwen3-8B',
    'Qwen/Qwen3-Embedding-4B': 'Qwen3-4B',
    'intfloat/e5-small-v2': 'e5-small-v2',
    'intfloat/e5-small-v2 (finetuned)': 'e5-small-v2 (finetuned)'
}

mteb_df['short_name'] = mteb_df['model_name'].map(model_mapping)

# Filter to only models we want to show
key_models = ['ml-e5-small', 'ml-e5-small (finetuned)', 'ml-e5-base', 'ml-e5-large', 
              'BGE-M3', 'Qwen3-8B', 'Qwen3-4B', 'e5-small-v2', 'e5-small-v2 (finetuned)']
plot_df = mteb_df[mteb_df['short_name'].isin(key_models)].copy()

# Create figure
fig, ax = plt.subplots(figsize=(12, 9))

# Plot diagonal reference line (where overall = italian-only)
min_val = 40
max_val = 96
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1.5, label='_nolegend_')

# Define colors and sizes for different models
colors = []
sizes = []
markers = []
for idx, row in plot_df.iterrows():
    if row['short_name'] == 'ml-e5-small':
        colors.append('#FFD700')  # Gold for base ml-e5-small
        sizes.append(550)
        markers.append('o')
    elif 'finetuned' in str(row['short_name']):
        colors.append('#E74C3C')  # Red for fine-tuned
        sizes.append(300)
        markers.append('s')  # Square marker
    else:
        colors.append('#4A90E2')  # Blue for other base models
        sizes.append(280)
        markers.append('o')

# Scatter plot with different markers
for idx, row in plot_df.iterrows():
    i = plot_df.index.get_loc(idx)
    ax.scatter(row['current_mteb_score'], row['italian_only_avg'],
              c=[colors[i]], s=[sizes[i]], alpha=0.8, 
              edgecolors='black', linewidth=2.5, 
              marker=markers[i], zorder=3)

# Manual label positioning to avoid overlap - carefully tuned positions
label_positions = {
    'ml-e5-small': (10, -40),
    'ml-e5-small (finetuned)': (10, 20),
    'ml-e5-base': (-95, -5),
    'ml-e5-large': (10, 15),
    'BGE-M3': (-70, 18),
    'Qwen3-8B': (-110, 5),
    'Qwen3-4B': (10, -15),
    'e5-small-v2': (10, 10),
    'e5-small-v2 (finetuned)': (-120, -15)
}

# Add labels for each point with manual positioning
for idx, row in plot_df.iterrows():
    offset = label_positions.get(row['short_name'], (10, 10))
    
    # Determine styling
    if row['short_name'] == 'ml-e5-small':
        fontweight = 'bold'
        fontsize = 11
        bbox_color = 'white'
        bbox_edge = 'black'
        bbox_linewidth = 2
    elif 'finetuned' in str(row['short_name']):
        fontweight = 'normal'
        fontsize = 10
        bbox_color = '#FFE6E6'
        bbox_edge = '#E74C3C'
        bbox_linewidth = 1.5
    else:
        fontweight = 'normal'
        fontsize = 10
        bbox_color = 'white'
        bbox_edge = 'none'
        bbox_linewidth = 0
    
    ax.annotate(row['short_name'], 
                (row['current_mteb_score'], row['italian_only_avg']),
                xytext=offset, textcoords='offset points',
                fontsize=fontsize, fontweight=fontweight,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color, 
                         edgecolor=bbox_edge, linewidth=bbox_linewidth, alpha=0.95),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.5) if abs(offset[0]) > 50 or abs(offset[1]) > 10 else None)

# Highlight ml-e5-small gap with arrow
ml_e5_row = plot_df[plot_df['short_name'] == 'ml-e5-small'].iloc[0]
gap = ml_e5_row['italian_only_avg'] - ml_e5_row['current_mteb_score']

# Draw vertical line showing gap
ax.plot([ml_e5_row['current_mteb_score'], ml_e5_row['current_mteb_score']], 
        [ml_e5_row['current_mteb_score'], ml_e5_row['italian_only_avg']], 
        'r-', linewidth=4.5, alpha=0.9, zorder=2)

# Add gap annotation with box - positioned to the left
ax.annotate(f'{gap:.1f} pts\ngap', 
            xy=(ml_e5_row['current_mteb_score'], 
                (ml_e5_row['current_mteb_score'] + ml_e5_row['italian_only_avg'])/2),
            xytext=(-75, 0), textcoords='offset points',
            fontsize=13, fontweight='bold', color='#D62728',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#D62728', linewidth=2.5),
            arrowprops=dict(arrowstyle='->', color='#D62728', linewidth=2.5),
            zorder=4)

# Labels and title
ax.set_xlabel('Overall MTEB Score', fontsize=15, fontweight='bold')
ax.set_ylabel('Italian-Only NDCG@10', fontsize=15, fontweight='bold')
ax.set_title('MTEB Aggregation Masks Italian Performance', fontsize=17, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# Set axis limits with padding
ax.set_xlim(38, 93)
ax.set_ylim(70, 97)

# Add text annotation explaining models above diagonal
ax.text(0.03, 0.97, 'Models above line: Better on Italian\nthan overall score suggests',
        transform=ax.transAxes, fontsize=10.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1))

# Add legend for marker types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90E2', 
           markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Base models'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C', 
           markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Fine-tuned models'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700', 
           markersize=11, markeredgecolor='black', markeredgewidth=1.5, label='ml-e5-small (base)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.95)

# Improve tick label size
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('figures/figure1_mteb_discrepancy.png', dpi=300, bbox_inches='tight')
plt.savefig( 'figures/figure1_mteb_discrepancy.pdf', bbox_inches='tight')
print("Figure 1 saved as figures/figure1_mteb_discrepancy.png and .pdf")
plt.close()