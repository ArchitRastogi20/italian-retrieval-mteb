#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import gc

import torch
import mteb
from tqdm import tqdm

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("mteb_ita_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mteb_ita_eval")

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

PROGRESS_FILE = OUTPUT_DIR / "progress.json"
CSV_OUTPUT = OUTPUT_DIR / "italian_retrieval_results.csv"

# Model list with current MTEB scores
MODELS_TO_EVALUATE = [
    ("Qwen/Qwen3-Embedding-9B", 90.39),
    ("Qwen/Qwen3-Embedding-4B", 86.19),
    ("intfloat/multilingual-e5-large-instruct", 86.26),
    ("Linq-AI-Research/inf-retriever-v1", 84.69),
    ("intfloat/multilingual-e5-large", 84.15),
    ("BAAI/bge-m3", 84.02),
    ("Snowflake/snowflake-arctic-embed-l-v2.0", 82.26),
    ("sbert/bilingual-embedding-large", 84.62),
    ("antoinelouis/bge-m3-custom-fr", 82.68),
    ("Salesforce/SFR-Embedding-Mistral", 80.29),
    ("sbert/Ling-Embed-Mistral", 80.67),
    ("google/embeddingsgemma-300m", 81.19),
    ("sbert/USE6-bge-m3", 82.09),
    ("jinaai/jina-embeddings-v3", 81.27),
    ("intfloat/e5-mistral-7b-instruct", 79.61),
    ("jinaai/jina-embeddings-v4", 81.38),
    ("Alibaba-NLP/gte-multilingual-base", 80.64),
    ("sbert/bilingual-embedding-base", 81.25),
    ("intfloat/multilingual-e5-base", 79.74),
    ("Salesforce/SFR-Embedding-2_R", 77.90),
    ("nvidia/NV-Embed-v1", 77.83),
    ("sbert/bilingual-embedding-small", 78.20),
    ("Linq-AI-Research/inf-retriever-v1-1.5b", 77.49),
    ("Qwen/Qwen3-Embedding-0.6B", 77.94),
    ("intfloat/multilingual-e5-small", 77.01),
    ("Snowflake/snowflake-arctic-embed-m-v2.0", 74.57),
    ("jinaai/jasper_en_vision_language_v1", 72.61),
    ("clagator/LaBSE-ru-turbo", 69.26),
    ("DoyyingDG/KaLM-embedding-multilingual-mini-instruct-v1", 69.01),
    ("infgrad/stella_en_1.5B_v5", 69.51),
    ("Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka", 65.65),
    ("sentence-transformers/LaBSE", 63.52),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 63.09),
    ("sentence-transformers/STS-multilingual-mpnet-base-v2", 56.76),
    ("Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet", 56.22),
]

# ----------------------------
# Helper Functions
# ----------------------------
def get_batch_size_for_model(model_name: str) -> int:
    """Determine batch size based on model name heuristics."""
    model_lower = model_name.lower()
    
    # Large models (>3B parameters or specific large models)
    if any(x in model_lower for x in ["9b", "7b", "large", "mistral", "nv-embed"]):
        return 32
    # Medium models
    elif any(x in model_lower for x in ["4b", "1.5b", "1b", "base", "m3"]):
        return 64
    # Small models
    elif any(x in model_lower for x in ["small", "mini", "300m", "0.6b"]):
        return 128
    # Default
    return 64


def load_progress() -> Dict:
    """Load progress from previous runs."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_progress(progress: Dict):
    """Save progress to resume later."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def cleanup_model_cache():
    """Clean up model cache to save disk space."""
    try:
        # Clear torch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear Python garbage
        gc.collect()
        
        # Clear HuggingFace cache for this model
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            # Remove blob files older than current run
            for blob_dir in cache_dir.glob("models--*"):
                try:
                    shutil.rmtree(blob_dir / "blobs", ignore_errors=True)
                except:
                    pass
        
        logger.info("Model cache cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning cache: {e}")


def to_dict(obj):
    """Convert pydantic models to dict for JSON serialization."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    return obj


def evaluate_model(model_name: str, mteb_score: float, batch_size: int) -> Dict:
    """Evaluate a single model on Italian retrieval tasks."""
    result = {
        "model_name": model_name,
        "current_mteb_score": mteb_score,
        "status": "Done",
        "error": None,
        "belebele_ita": None,
        "wiki_ita_mono": None,
        "wiki_ita_to_eng": None,
        "wiki_eng_to_ita": None,
        "avg_mono": None,
        "avg_cross": None,
        "evaluation_time": 0,
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Initial batch size: {batch_size}")
        logger.info(f"{'='*80}")
        
        # Load model
        model = mteb.get_model(model_name)
        
        # Try FP16 for faster inference on GPU
        if torch.cuda.is_available():
            try:
                if hasattr(model, "model") and hasattr(model.model, "half"):
                    model.model.half()
                    logger.info("Using FP16 for faster encoding")
            except Exception as e:
                logger.warning(f"Could not enable FP16: {e}")
        
        # Create tasks
        tasks = []
        
        # BelebeleRetrieval - all Italian subsets (mono + cross-lingual)
        try:
            bele = mteb.get_task(
                "BelebeleRetrieval",
                hf_subsets=["ita_Latn-ita_Latn", "ita_Latn-eng_Latn", "eng_Latn-ita_Latn"],
            )
            tasks.append(("belebele", bele))
        except Exception as e:
            logger.error(f"Could not create BelebeleRetrieval task: {e}")
        
        # WikipediaRetrievalMultilingual - Italian mono
        try:
            wiki_mono = mteb.get_task(
                "WikipediaRetrievalMultilingual",
                hf_subsets=["it"],
                languages=["ita"],
            )
            tasks.append(("wiki_ita_mono", wiki_mono))
        except Exception as e:
            logger.error(f"Could not create Wiki mono task: {e}")
        
        if not tasks:
            raise ValueError("No tasks could be created")
        
        # Evaluate each task with progress bar
        all_results = {}
        task_pbar = tqdm(tasks, desc=f"Tasks for {model_name.split('/')[-1]}", leave=False)
        
        for task_name, task in task_pbar:
            task_pbar.set_description(f"Running {task_name}")
            
            current_batch = batch_size
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    encode_kwargs = {"batch_size": current_batch}
                    task_result = mteb.evaluate(model, [task], encode_kwargs=encode_kwargs)
                    all_results[task_name] = to_dict(task_result)
                    logger.info(f"  {task_name} completed with batch_size={current_batch}")
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and attempt < max_retries - 1:
                        current_batch = max(1, current_batch // 2)
                        logger.warning(f"OOM detected, reducing batch size to {current_batch}")
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise
        
        # Save raw JSON output
        model_safe_name = model_name.replace("/", "_")
        json_output_dir = OUTPUT_DIR / model_safe_name
        json_output_dir.mkdir(exist_ok=True)
        
        json_output_path = json_output_dir / "results.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved raw JSON to {json_output_path}")
        
        # Extract ndcg@10 scores
        for task_name, task_result in all_results.items():
            try:
                scores_dict = {}
                # Navigate through the nested structure
                if isinstance(task_result, dict):
                    for task_data in task_result.get("task_results", [task_result]):
                        if isinstance(task_data, dict):
                            test_scores = task_data.get("scores", {}).get("test", [])
                            if test_scores:
                                # Belebele returns multiple subsets
                                if task_name == "belebele":
                                    for subset_scores in test_scores:
                                        hf_subset = subset_scores.get("hf_subset", "")
                                        ndcg_10 = subset_scores.get("ndcg_at_10")
                                        
                                        if ndcg_10 is not None:
                                            ndcg_10_percent = round(ndcg_10 * 100, 2)
                                            
                                            if hf_subset == "ita_Latn-ita_Latn":
                                                result["belebele_ita"] = ndcg_10_percent
                                                logger.info(f"  belebele_ita: {ndcg_10_percent}")
                                            elif hf_subset == "ita_Latn-eng_Latn":
                                                result["wiki_ita_to_eng"] = ndcg_10_percent
                                                logger.info(f"  wiki_ita_to_eng (from Belebele): {ndcg_10_percent}")
                                            elif hf_subset == "eng_Latn-ita_Latn":
                                                result["wiki_eng_to_ita"] = ndcg_10_percent
                                                logger.info(f"  wiki_eng_to_ita (from Belebele): {ndcg_10_percent}")
                                else:
                                    # Wiki tasks return single subset
                                    scores = test_scores[0] if isinstance(test_scores, list) else test_scores
                                    if "ndcg_at_10" in scores:
                                        result[task_name] = round(scores["ndcg_at_10"] * 100, 2)
                                        logger.info(f"  {task_name}: {result[task_name]}")
                                break
            except Exception as e:
                logger.error(f"Error extracting score for {task_name}: {e}")
        
        # Calculate averages
        if result["belebele_ita"] and result["wiki_ita_mono"]:
            result["avg_mono"] = round((result["belebele_ita"] + result["wiki_ita_mono"]) / 2, 2)
        
        if result["wiki_ita_to_eng"] and result["wiki_eng_to_ita"]:
            result["avg_cross"] = round((result["wiki_ita_to_eng"] + result["wiki_eng_to_ita"]) / 2, 2)
        
        result["evaluation_time"] = round(time.time() - start_time, 2)
        logger.info(f"  Model evaluation completed in {result['evaluation_time']}s")
        
    except Exception as e:
        result["status"] = "Failed"
        result["error"] = str(e)
        logger.error(f" Model evaluation failed: {e}")
    
    finally:
        # Cleanup
        cleanup_model_cache()
    
    return result


def main():
    """Main evaluation loop."""
    logger.info("="*80)
    logger.info("MTEB Italian Retrieval Evaluation Pipeline")
    logger.info("="*80)
    logger.info(f"Total models to evaluate: {len(MODELS_TO_EVALUATE)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("="*80)
    
    # Load progress
    progress = load_progress()
    completed_models = set(progress["completed"])
    all_results = []
    
    # Load existing results if any
    if CSV_OUTPUT.exists():
        import pandas as pd
        try:
            existing_df = pd.read_csv(CSV_OUTPUT)
            for _, row in existing_df.iterrows():
                all_results.append(row.to_dict())
            logger.info(f"Loaded {len(all_results)} existing results from CSV")
        except:
            pass
    
    # Evaluate models
    models_to_run = [(m, s) for m, s in MODELS_TO_EVALUATE if m not in completed_models]
    
    if not models_to_run:
        logger.info("All models already evaluated!")
        return
    
    logger.info(f"Models to evaluate: {len(models_to_run)}")
    logger.info(f"Already completed: {len(completed_models)}")
    
    # Main progress bar
    main_pbar = tqdm(models_to_run, desc="Overall Progress")
    
    for model_name, mteb_score in main_pbar:
        main_pbar.set_description(f"Evaluating {model_name.split('/')[-1]}")
        
        # Determine batch size
        batch_size = get_batch_size_for_model(model_name)
        
        # Evaluate
        result = evaluate_model(model_name, mteb_score, batch_size)
        all_results.append(result)
        
        # Update progress
        if result["status"] == "Done":
            progress["completed"].append(model_name)
        else:
            progress["failed"][model_name] = result["error"]
        
        save_progress(progress)
        
        # Save intermediate CSV
        save_results_to_csv(all_results)
    
    logger.info("="*80)
    logger.info("Evaluation completed!")
    logger.info(f"Total evaluated: {len(progress['completed'])}")
    logger.info(f"Failed: {len(progress['failed'])}")
    logger.info(f"Results saved to: {CSV_OUTPUT}")
    logger.info("="*80)


def save_results_to_csv(results: List[Dict]):
    """Save results to CSV file."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        "model_name",
        "current_mteb_score",
        "belebele_ita",
        "wiki_ita_mono",
        "wiki_ita_to_eng",
        "wiki_eng_to_ita",
        "avg_mono",
        "avg_cross",
        "evaluation_time",
        "status",
        "error",
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Sort by avg_mono (descending)
    if "avg_mono" in df.columns:
        df = df.sort_values("avg_mono", ascending=False, na_position="last")
    
    df.to_csv(CSV_OUTPUT, index=False)
    logger.info(f"CSV saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("MTEB_NUM_PROC", str(os.cpu_count() or 8))
    
    main()
