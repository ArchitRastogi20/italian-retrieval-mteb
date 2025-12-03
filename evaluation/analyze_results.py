#!/usr/bin/env python3
"""
Analyze and visualize MTEB Italian Retrieval Results
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_results(csv_path: str = "results/italian_retrieval_results.csv"):
    """Analyze the results CSV and print detailed statistics."""
    
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found!")
        print("Run the evaluation first: python3 run_ita_eval.py")
        return
    
    df = pd.read_csv(csv_path)
    
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║           MTEB Italian Retrieval - Results Analysis                  ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝\n")
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("━" * 75)
    print(f"Total models evaluated:    {len(df)}")
    print(f"Successfully completed:    {len(df[df['status'] == 'Done'])}")
    print(f"Failed evaluations:        {len(df[df['status'] == 'Failed'])}")
    
    if 'avg_mono' in df.columns:
        valid_mono = df[df['avg_mono'].notna()]
        if len(valid_mono) > 0:
            print(f"\nITALIAN MONOLINGUAL PERFORMANCE (avg_mono)")
            print("━" * 75)
            print(f"Mean:    {valid_mono['avg_mono'].mean():.2f}")
            print(f"Median:  {valid_mono['avg_mono'].median():.2f}")
            print(f"Std:     {valid_mono['avg_mono'].std():.2f}")
            print(f"Min:     {valid_mono['avg_mono'].min():.2f} ({valid_mono.loc[valid_mono['avg_mono'].idxmin(), 'model_name']})")
            print(f"Max:     {valid_mono['avg_mono'].max():.2f} ({valid_mono.loc[valid_mono['avg_mono'].idxmax(), 'model_name']})")
    
    if 'avg_cross' in df.columns:
        valid_cross = df[df['avg_cross'].notna()]
        if len(valid_cross) > 0:
            print(f"\nCROSS-LINGUAL PERFORMANCE (avg_cross)")
            print("━" * 75)
            print(f"Mean:    {valid_cross['avg_cross'].mean():.2f}")
            print(f"Median:  {valid_cross['avg_cross'].median():.2f}")
            print(f"Std:     {valid_cross['avg_cross'].std():.2f}")
            print(f"Min:     {valid_cross['avg_cross'].min():.2f} ({valid_cross.loc[valid_cross['avg_cross'].idxmin(), 'model_name']})")
            print(f"Max:     {valid_cross['avg_cross'].max():.2f} ({valid_cross.loc[valid_cross['avg_cross'].idxmax(), 'model_name']})")
    
    # Top performers
    print(f"\nTOP 10 MODELS - Italian Monolingual (avg_mono)")
    print("━" * 75)
    print(f"{'Rank':<6} {'Model':<50} {'Score':>8}")
    print("━" * 75)
    
    if 'avg_mono' in df.columns:
        top_10 = df[df['avg_mono'].notna()].nlargest(10, 'avg_mono')
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            model_short = row['model_name'].split('/')[-1][:45]
            print(f"{i:<6} {model_short:<50} {row['avg_mono']:>8.2f}")
    
    print(f"\nTOP 10 MODELS - Cross-lingual (avg_cross)")
    print("━" * 75)
    print(f"{'Rank':<6} {'Model':<50} {'Score':>8}")
    print("━" * 75)
    
    if 'avg_cross' in df.columns:
        top_10_cross = df[df['avg_cross'].notna()].nlargest(10, 'avg_cross')
        for i, (idx, row) in enumerate(top_10_cross.iterrows(), 1):
            model_short = row['model_name'].split('/')[-1][:45]
            print(f"{i:<6} {model_short:<50} {row['avg_cross']:>8.2f}")
    
    # Gap analysis
    print(f"\n📉 MTEB vs ITALIAN PERFORMANCE GAP")
    print("━" * 75)
    
    if 'current_mteb_score' in df.columns and 'avg_mono' in df.columns:
        valid_gap = df[(df['current_mteb_score'].notna()) & (df['avg_mono'].notna())].copy()
        valid_gap['gap'] = valid_gap['current_mteb_score'] - valid_gap['avg_mono']
        
        print(f"Models with complete data: {len(valid_gap)}")
        print(f"Average gap: {valid_gap['gap'].mean():.2f} (MTEB - Italian)")
        
        print(f"\nBiggest OVER-performers (Italian > MTEB):")
        over_performers = valid_gap.nsmallest(5, 'gap')
        for idx, row in over_performers.iterrows():
            print(f"  {row['model_name'][:50]:50s}  Gap: {row['gap']:+6.2f}")
        
        print(f"\nBiggest UNDER-performers (Italian < MTEB):")
        under_performers = valid_gap.nlargest(5, 'gap')
        for idx, row in under_performers.iterrows():
            print(f"  {row['model_name'][:50]:50s}  Gap: {row['gap']:+6.2f}")
    
    # Task breakdown
    print(f"\nTASK-SPECIFIC SCORES")
    print("━" * 75)
    
    task_cols = ['belebele_ita', 'wiki_ita_mono', 'wiki_ita_to_eng', 'wiki_eng_to_ita']
    task_names = ['Belebele IT', 'Wiki IT Mono', 'Wiki IT→EN', 'Wiki EN→IT']
    
    for col, name in zip(task_cols, task_names):
        if col in df.columns:
            valid = df[df[col].notna()]
            if len(valid) > 0:
                print(f"{name:15s}  Mean: {valid[col].mean():5.2f}  "
                      f"Min: {valid[col].min():5.2f}  "
                      f"Max: {valid[col].max():5.2f}")
    
    # Failed models
    failed = df[df['status'] == 'Failed']
    if len(failed) > 0:
        print(f"\nFAILED EVALUATIONS ({len(failed)})")
        print("━" * 75)
        for idx, row in failed.iterrows():
            error_msg = str(row['error'])[:50] if pd.notna(row['error']) else 'Unknown error'
            print(f"  • {row['model_name']}")
            print(f"    Error: {error_msg}")
    
    print("\n" + "━" * 75)
    print(f"Analysis complete! Full results in: {csv_path}")
    print("━" * 75)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/italian_retrieval_results.csv"
    analyze_results(csv_path)
