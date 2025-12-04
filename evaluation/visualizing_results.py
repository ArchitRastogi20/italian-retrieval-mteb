import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================================
# GLOBAL PATH VARIABLES - UPDATE THESE
# ============================================================================
MTEB_ITA_RESULTS_PATH = r"results\mteb_ita_subset_results.xlsx"
RAGEVAL_ITA_RESULTS_PATH = r"results\rageval_ita_results.xlsx"
RAGEVAL_BM25_RESULTS_PATH = r"results\rageval_ita_bm25_results.csv"
OUTPUT_DIR = r"results\figures"  # Output directory for images

# ============================================================================
# STYLING
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'claimed': '#E63946',      # Red - what's claimed
    'actual': '#2A9D8F',       # Teal - actual performance
    'rageval': '#F4A261',      # Orange - rageval results
    'bm25': '#264653',         # Dark blue - baseline
    'difference': '#E76F51'    # Coral - gap
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_mteb_data():
    """Load MTEB Italian results"""
    df = pd.read_excel(MTEB_ITA_RESULTS_PATH)
    # Calculate actual Italian score as average of Italian metrics
    df['actual_ita_score'] = df['avg_mono']  # or you can average multiple columns
    return df

def load_rageval_data():
    """Load RAGEval results"""
    df = pd.read_excel(RAGEVAL_ITA_RESULTS_PATH)
    return df

def load_bm25_data():
    """Load BM25 baseline results"""
    df = pd.read_csv(RAGEVAL_BM25_RESULTS_PATH)
    return df

# ============================================================================
# FIGURE 1: MTEB - Claimed vs Actual Italian Performance
# ============================================================================

def plot_mteb_comparison(df_mteb, output_name='fig1_mteb_claimed_vs_actual.png'):
    """Compare actual Italian MTEB scores vs what's claimed on website"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Sort by claimed score
    df_plot = df_mteb.sort_values('current_mteb_score', ascending=False).head(15)
    
    models = df_plot['model_name'].str.split('/').str[-1]  # Get model name without org
    claimed = df_plot['current_mteb_score'].values
    actual = df_plot['actual_ita_score'].values
    
    x = np.arange(len(models))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, claimed, width, label='MTEB Website Score (All Languages)', 
                   color=COLORS['claimed'], alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, actual, width, label='Actual Italian Performance', 
                   color=COLORS['actual'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=15, fontweight='bold')
    ax.set_ylabel('Score', fontsize=15, fontweight='bold')
    ax.set_title('MTEB Reality Check: Website Claims vs Real Italian Performance', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=13, framealpha=0.95, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(claimed.max(), actual.max()) * 1.15)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 2: MTEB - The Gap (Difference)
# ============================================================================

def plot_mteb_gap(df_mteb, output_name='fig2_mteb_gap_analysis.png'):
    """Show the performance gap between claimed and actual"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    df_plot = df_mteb.sort_values('current_mteb_score', ascending=False).head(15)
    models = df_plot['model_name'].str.split('/').str[-1]
    gap = df_plot['current_mteb_score'].values - df_plot['actual_ita_score'].values
    
    # Sort by gap
    sorted_idx = np.argsort(gap)[::-1]
    models = models.iloc[sorted_idx]
    gap = gap[sorted_idx]
    
    colors = [COLORS['difference'] if g > 0 else COLORS['actual'] for g in gap]
    
    bars = ax.barh(range(len(models)), gap, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, gap)):
        x_pos = val + 0.5 if val > 0 else val - 0.5
        ax.text(x_pos, i, f'{val:.1f}',
               ha='left' if val > 0 else 'right', va='center', 
               fontsize=11, fontweight='bold')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_xlabel('Performance Gap (Claimed - Actual Italian Score)', fontsize=14, fontweight='bold')
    ax.set_title('The Marketing Gap: How Much Do Website Scores Overstate Italian Performance?', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linewidth=2.5, linestyle='-', alpha=0.7)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 3: RAGEval - Full Comparison at k=10
# ============================================================================

def plot_rageval_full_comparison(df_mteb, df_rageval, df_bm25, 
                                  output_name='fig3_rageval_full_comparison.png'):
    """Compare MTEB claimed, MTEB actual, RAGEval results, and BM25 baseline"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Filter for k=10 (good middle ground)
    df_rageval_k10 = df_rageval[df_rageval['k'] == 10].copy()
    df_bm25_k10 = df_bm25[df_bm25['k'] == 10].copy()
    
    # Sort by rageval recall
    df_rageval_k10 = df_rageval_k10.sort_values('recall', ascending=False).head(12)
    
    models = df_rageval_k10['model'].str.split('/').str[-1]
    
    # Get corresponding MTEB data
    mteb_dict = df_mteb.set_index('model_name')[['current_mteb_score', 'actual_ita_score']].to_dict('index')
    
    claimed_scores = []
    actual_scores = []
    for model_full in df_rageval_k10['model']:
        if model_full in mteb_dict:
            claimed_scores.append(mteb_dict[model_full]['current_mteb_score'])
            actual_scores.append(mteb_dict[model_full]['actual_ita_score'])
        else:
            claimed_scores.append(0)
            actual_scores.append(0)
    
    # Scale RAGEval recall to 0-100 for comparison
    rageval_scores = (df_rageval_k10['recall'].values * 100).tolist()
    
    # BM25 baseline
    bm25_baseline = df_bm25_k10['recall'].values[0] * 100
    
    x = np.arange(len(models))
    width = 0.22
    
    bars1 = ax.bar(x - 1.5*width, claimed_scores, width, label='MTEB Website (All Lang)', 
                   color=COLORS['claimed'], alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x - 0.5*width, actual_scores, width, label='MTEB Italian Only', 
                   color=COLORS['actual'], alpha=0.85, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + 0.5*width, rageval_scores, width, label='RAGEval Recall@10', 
                   color=COLORS['rageval'], alpha=0.85, edgecolor='black', linewidth=1.2)
    
    # BM25 baseline line
    ax.axhline(y=bm25_baseline, color=COLORS['bm25'], linestyle='--', 
               linewidth=3.5, label=f'BM25 Baseline ({bm25_baseline:.1f})', alpha=0.9, zorder=10)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=15, fontweight='bold')
    ax.set_ylabel('Score', fontsize=15, fontweight='bold')
    ax.set_title('The Complete Picture: MTEB Claims vs Reality vs RAGEval vs BM25 (k=10)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, framealpha=0.95, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 4: RAGEval Metrics Across k values
# ============================================================================

def plot_rageval_metrics_by_k(df_rageval, df_bm25, output_name='fig4_rageval_vs_bm25_by_k.png'):
    """Show how different models perform across k values compared to BM25"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    metrics = ['recall', 'ndcg', 'mrr', 'eir']
    titles = ['Recall@k', 'NDCG@k', 'MRR@k', 'EIR@k']
    
    # Get top 5 models by average recall
    top_models = df_rageval.groupby('model')['recall'].mean().nlargest(5).index
    
    colors_models = plt.cm.tab10(np.linspace(0, 0.9, len(top_models)))
    
    for idx, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
        # Plot BM25
        bm25_data = df_bm25.sort_values('k')
        ax.plot(bm25_data['k'], bm25_data[metric], marker='s', linewidth=3.5, 
                markersize=12, label='BM25 Baseline', color=COLORS['bm25'], 
                alpha=0.9, linestyle='--', zorder=10)
        
        # Plot top models
        for model, color in zip(top_models, colors_models):
            model_data = df_rageval[df_rageval['model'] == model].sort_values('k')
            model_short = model.split('/')[-1]
            ax.plot(model_data['k'], model_data[metric], marker='o', linewidth=2.5,
                    markersize=8, label=model_short, color=color, alpha=0.8)
        
        ax.set_xlabel('k (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}: Neural Models vs BM25 Baseline', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        ax.set_xticks([1, 5, 10, 20, 50])
        ax.set_xticklabels(['1', '5', '10', '20', '50'])
    
    plt.suptitle('RAGEval Performance Analysis: Do Neural Embeddings Beat BM25?', 
                 fontsize=19, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 5: Heatmap - All Models All Metrics
# ============================================================================

def plot_comprehensive_heatmap(df_rageval, output_name='fig5_comprehensive_heatmap.png'):
    """Heatmap showing all models across all metrics at k=10"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Filter for k=10
    df_k10 = df_rageval[df_rageval['k'] == 10].copy()
    df_k10 = df_k10.sort_values('recall', ascending=False)
    
    models = df_k10['model'].str.split('/').str[-1]
    
    # Create heatmap data
    heatmap_data = df_k10[['recall', 'ndcg', 'mrr', 'eir']].values.T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, ax=ax, 
                linewidths=0.5, linecolor='gray',
                xticklabels=models, yticklabels=['Recall', 'NDCG', 'MRR', 'EIR'],
                vmin=0, vmax=0.7)
    
    ax.set_title('RAGEval Comprehensive Metrics @ k=10: All Models, All Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 6: Model Efficiency - Performance vs Parameters
# ============================================================================

def plot_efficiency_analysis(df_rageval, output_name='fig6_efficiency_analysis.png'):
    """Plot performance vs model size"""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get k=10 data
    df_k10 = df_rageval[df_rageval['k'] == 10].copy()
    
    # Convert para to numeric (handle M, B suffixes)
    def convert_params(x):
        if pd.isna(x):
            return 0
        x = str(x).upper()
        if 'B' in x:
            return float(x.replace('B', '')) * 1000
        elif 'M' in x:
            return float(x.replace('M', ''))
        return float(x)
    
    df_k10['params_m'] = df_k10['para'].apply(convert_params)
    
    # Plot
    scatter = ax.scatter(df_k10['params_m'], df_k10['recall'] * 100, 
                        s=df_k10['dim'] / 3,  # Size by embedding dimension
                        c=df_k10['ndcg'], cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add model labels
    for _, row in df_k10.iterrows():
        model_name = row['model'].split('/')[-1]
        ax.annotate(model_name, (row['params_m'], row['recall'] * 100),
                   fontsize=9, alpha=0.8, xytext=(5, 5), 
                   textcoords='offset points')
    
    ax.set_xlabel('Model Size (Million Parameters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall@10 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Efficiency Analysis: Performance vs Model Size (bubble size = embedding dim)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax, label='NDCG@10')
    cbar.set_label('NDCG@10', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/{output_name}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all figures"""
    print("\n" + "="*80)
    print("GENERATING ITALIAN RETRIEVAL BENCHMARK VISUALIZATIONS")
    print("="*80 + "\n")
    
    print("Loading data...")
    df_mteb = load_mteb_data()
    df_rageval = load_rageval_data()
    df_bm25 = load_bm25_data()
    print(f"✓ Loaded {len(df_mteb)} MTEB models")
    print(f"✓ Loaded {len(df_rageval)} RAGEval entries")
    print(f"✓ Loaded {len(df_bm25)} BM25 entries\n")
    
    print("Generating visualizations...")
    print("-"*80)
    
    # Generate all plots
    plot_mteb_comparison(df_mteb)
    plot_mteb_gap(df_mteb)
    plot_rageval_full_comparison(df_mteb, df_rageval, df_bm25)
    plot_rageval_metrics_by_k(df_rageval, df_bm25)
    plot_comprehensive_heatmap(df_rageval)
    plot_efficiency_analysis(df_rageval)
    
    print("-"*80)
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated figures:")
    print("  1. fig1_mteb_claimed_vs_actual.png - MTEB website vs real Italian performance")
    print("  2. fig2_mteb_gap_analysis.png - The marketing gap")
    print("  3. fig3_rageval_full_comparison.png - Complete comparison with BM25")
    print("  4. fig4_rageval_vs_bm25_by_k.png - Performance across k values")
    print("  5. fig5_comprehensive_heatmap.png - All models, all metrics")
    print("  6. fig6_efficiency_analysis.png - Performance vs model size")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()