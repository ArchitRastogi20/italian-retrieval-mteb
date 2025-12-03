#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse JSON results from MTEB Italian evaluation and generate CSV report.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("json_to_csv")

# Model list with current MTEB scores
MTEB_SCORES = {
    "Qwen/Qwen3-Embedding-9B": 90.39,
    "Qwen/Qwen3-Embedding-4B": 86.19,
    "intfloat/multilingual-e5-large-instruct": 86.26,
    "Linq-AI-Research/inf-retriever-v1": 84.69,
    "intfloat/multilingual-e5-large": 84.15,
    "BAAI/bge-m3": 84.02,
    "Snowflake/snowflake-arctic-embed-l-v2.0": 82.26,
    "sbert/bilingual-embedding-large": 84.62,
    "antoinelouis/bge-m3-custom-fr": 82.68,
    "Salesforce/SFR-Embedding-Mistral": 80.29,
    "sbert/Ling-Embed-Mistral": 80.67,
    "google/embeddingsgemma-300m": 81.19,
    "sbert/USE6-bge-m3": 82.09,
    "jinaai/jina-embeddings-v3": 81.27,
    "intfloat/e5-mistral-7b-instruct": 79.61,
    "jinaai/jina-embeddings-v4": 81.38,
    "Alibaba-NLP/gte-multilingual-base": 80.64,
    "sbert/bilingual-embedding-base": 81.25,
    "intfloat/multilingual-e5-base": 79.74,
    "Salesforce/SFR-Embedding-2_R": 77.90,
    "nvidia/NV-Embed-v1": 77.83,
    "sbert/bilingual-embedding-small": 78.20,
    "Linq-AI-Research/inf-retriever-v1-1.5b": 77.49,
    "Qwen/Qwen3-Embedding-0.6B": 77.94,
    "intfloat/multilingual-e5-small": 77.01,
    "Snowflake/snowflake-arctic-embed-m-v2.0": 74.57,
    "jinaai/jasper_en_vision_language_v1": 72.61,
    "clagator/LaBSE-ru-turbo": 69.26,
    "DoyyingDG/KaLM-embedding-multilingual-mini-instruct-v1": 69.01,
    "infgrad/stella_en_1.5B_v5": 69.51,
    "Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka": 65.65,
    "sentence-transformers/LaBSE": 63.52,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 63.09,
    "sentence-transformers/STS-multilingual-mpnet-base-v2": 56.76,
    "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet": 56.22,
}


def extract_ndcg_from_json(json_path: Path) -> Dict[str, Optional[float]]:
    """Extract ndcg@10 scores from JSON result file."""
    scores = {
        "belebele_ita": None,
        "wiki_ita_mono": None,
        "wiki_ita_to_eng": None,
        "wiki_eng_to_ita": None,
    }
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is the output from run_ita_eval.py (nested format)
        if isinstance(data, dict) and any(key in data for key in scores.keys()):
            # Format: {task_name: {task_results: [{scores: {test: [{ndcg_at_10: ...}]}}]}}
            for task_name in scores.keys():
                if task_name in data:
                    task_data = data[task_name]
                    if isinstance(task_data, dict):
                        for task_result in task_data.get("task_results", [task_data]):
                            if isinstance(task_result, dict):
                                test_scores = task_result.get("scores", {}).get("test", [])
                                if test_scores:
                                    score_dict = test_scores[0] if isinstance(test_scores, list) else test_scores
                                    if "ndcg_at_10" in score_dict:
                                        scores[task_name] = round(score_dict["ndcg_at_10"] * 100, 2)
                                        break
        
        # Check if this is the original format (single task result)
        elif "task_results" in data:
            model_name = data.get("model_name", "")
            
            for task_result in data["task_results"]:
                task_name = task_result.get("task_name", "")
                test_scores = task_result.get("scores", {}).get("test", [])
                
                if not test_scores:
                    continue
                
                # Process each subset
                for subset_scores in test_scores:
                    hf_subset = subset_scores.get("hf_subset", "")
                    languages = subset_scores.get("languages", [])
                    ndcg_10 = subset_scores.get("ndcg_at_10")
                    
                    if ndcg_10 is None:
                        continue
                    
                    ndcg_10_percent = round(ndcg_10 * 100, 2)
                    
                    # Map to our task names
                    if "BelebeleRetrieval" in task_name:
                        if hf_subset == "ita_Latn-ita_Latn":
                            scores["belebele_ita"] = ndcg_10_percent
                        elif hf_subset == "ita_Latn-eng_Latn":
                            # For Belebele, IT->EN cross-lingual is also part of the task
                            if not scores["wiki_ita_to_eng"]:  # Use if Wiki didn't provide it
                                scores["wiki_ita_to_eng"] = ndcg_10_percent
                        elif hf_subset == "eng_Latn-ita_Latn":
                            # For Belebele, EN->IT cross-lingual is also part of the task
                            if not scores["wiki_eng_to_ita"]:  # Use if Wiki didn't provide it
                                scores["wiki_eng_to_ita"] = ndcg_10_percent
                    
                    elif "WikipediaRetrieval" in task_name:
                        if hf_subset == "it":
                            scores["wiki_ita_mono"] = ndcg_10_percent
                        elif hf_subset == "ita_Latn-eng_Latn":
                            scores["wiki_ita_to_eng"] = ndcg_10_percent
                        elif hf_subset == "eng_Latn-ita_Latn":
                            scores["wiki_eng_to_ita"] = ndcg_10_percent
        
    except Exception as e:
        logger.error(f"Error parsing {json_path}: {e}")
    
    return scores


def parse_results_directory(results_dir: Path) -> pd.DataFrame:
    """Parse all JSON files in results directory and create DataFrame."""
    all_results = []
    
    # Look for subdirectories named after models
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        json_file = model_dir / "results.json"
        if not json_file.exists():
            continue
        
        # Convert directory name back to model name
        model_name = model_dir.name.replace("_", "/", 1)  # Only replace first underscore
        
        logger.info(f"Processing {model_name}...")
        
        # Extract scores
        scores = extract_ndcg_from_json(json_file)
        
        # Calculate averages
        avg_mono = None
        if scores["belebele_ita"] and scores["wiki_ita_mono"]:
            avg_mono = round((scores["belebele_ita"] + scores["wiki_ita_mono"]) / 2, 2)
        
        avg_cross = None
        if scores["wiki_ita_to_eng"] and scores["wiki_eng_to_ita"]:
            avg_cross = round((scores["wiki_ita_to_eng"] + scores["wiki_eng_to_ita"]) / 2, 2)
        
        result = {
            "model_name": model_name,
            "current_mteb_score": MTEB_SCORES.get(model_name, None),
            "belebele_ita": scores["belebele_ita"],
            "wiki_ita_mono": scores["wiki_ita_mono"],
            "wiki_ita_to_eng": scores["wiki_ita_to_eng"],
            "wiki_eng_to_ita": scores["wiki_eng_to_ita"],
            "avg_mono": avg_mono,
            "avg_cross": avg_cross,
            "status": "Done" if any(scores.values()) else "No scores",
        }
        
        all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    # Sort by avg_mono descending
    if "avg_mono" in df.columns and not df["avg_mono"].isna().all():
        df = df.sort_values("avg_mono", ascending=False, na_position="last")
    
    return df


def main():
    results_dir = Path("results")
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    logger.info(f"Parsing results from: {results_dir}")
    
    df = parse_results_directory(results_dir)
    
    if df.empty:
        logger.warning("No results found!")
        return
    
    # Save to CSV
    output_csv = results_dir / "italian_retrieval_results.csv"
    df.to_csv(output_csv, index=False)
    
    logger.info(f"\n{'-'*80}")
    logger.info(f"Results saved to: {output_csv}")
    logger.info(f"Total models: {len(df)}")
    logger.info(f"{'-'*80}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if "avg_mono" in df.columns:
        valid_mono = df[df["avg_mono"].notna()]["avg_mono"]
        if len(valid_mono) > 0:
            print(f"\nMonolingual Average (Belebele + Wiki IT):")
            print(f"  Mean: {valid_mono.mean():.2f}")
            print(f"  Median: {valid_mono.median():.2f}")
            print(f"  Min: {valid_mono.min():.2f}")
            print(f"  Max: {valid_mono.max():.2f}")
    
    if "avg_cross" in df.columns:
        valid_cross = df[df["avg_cross"].notna()]["avg_cross"]
        if len(valid_cross) > 0:
            print(f"\nCross-lingual Average (ITâ†”EN):")
            print(f"  Mean: {valid_cross.mean():.2f}")
            print(f"  Median: {valid_cross.median():.2f}")
            print(f"  Min: {valid_cross.min():.2f}")
            print(f"  Max: {valid_cross.max():.2f}")
    
    print("\n" + "="*80)
    print("TOP 10 MODELS (by avg_mono)")
    print("="*80)
    
    if "avg_mono" in df.columns:
        top_10 = df[df["avg_mono"].notna()].head(10)
        for idx, row in top_10.iterrows():
            print(f"{row['model_name']:60s} | {row['avg_mono']:5.2f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
