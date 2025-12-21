#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging

import torch
import mteb

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("mteb_ita_eval")

BANNER = r"""
MTEB Italian Evaluation - Full (Monolingual + Cross-lingual)
"""

def parse_args():
    ap = argparse.ArgumentParser(description="MTEB Italian Retrieval (Monolingual + Cross-lingual)")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--batch-size", type=int, default=128, help="Encode batch size")
    ap.add_argument("--fp16", action="store_true", help="Try FP16 on GPU (best-effort)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings in encode()")
    ap.add_argument("--save-json", default="", help="Optional path to save results JSON")
    ap.add_argument("--cross-lingual", action="store_true", help="Include cross-lingual tasks (IT-EN, EN-IT)")
    return ap.parse_args()

def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("MTEB_NUM_PROC", str(os.cpu_count() or 8))

    logger.info(BANNER)
    logger.info(f"Args: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model via mteb.get_model on {device}: {args.model}")
    model = mteb.get_model(args.model)

    # Best-effort FP16 on the wrapped SentenceTransformer (if exposed)
    if args.fp16 and torch.cuda.is_available():
        try:
            if hasattr(model, "model") and hasattr(model.model, "half"):
                model.model.half()
                logger.info("Using FP16 (half precision) for faster encoding.")
        except Exception as e:
            logger.warning(f"Could not enable FP16: {e}")

    # -------------------------
    # Build tasks with Italian subsets
    # -------------------------
    tasks = []
    
    # BelebeleRetrieval - Italian monolingual
    try:
        bele_subsets = ["ita_Latn-ita_Latn"]
        
        # Add cross-lingual if requested
        if args.cross_lingual:
            bele_subsets.extend(["ita_Latn-eng_Latn", "eng_Latn-ita_Latn"])
        
        bele = mteb.get_task(
            "BelebeleRetrieval",
            hf_subsets=bele_subsets,
        )
        tasks.append(bele)
        logger.info(f"BelebeleRetrieval subsets: {bele_subsets}")
    except Exception as e:
        logger.warning(f"Could not create BelebeleRetrieval task: {e}")

    # WikipediaRetrievalMultilingual - Italian tasks
    try:
        wiki_subsets = ["it"]  # Italian monolingual
        
        # Add cross-lingual if requested
        if args.cross_lingual:
            wiki_subsets.extend(["it-en", "en-it"])
        
        wiki = mteb.get_task(
            "WikipediaRetrievalMultilingual",
            hf_subsets=wiki_subsets,
            languages=["ita"],  # Still Italian-focused
        )
        tasks.append(wiki)
        logger.info(f"WikipediaRetrievalMultilingual subsets: {wiki_subsets}")
    except Exception as e:
        logger.warning(f"Could not create WikipediaRetrievalMultilingual task: {e}")

    if not tasks:
        logger.error("No tasks to evaluate after configuration. Exiting.")
        return

    logger.info(f"Tasks to evaluate: {[t.metadata.name for t in tasks]}")

    # Batch size / normalization forwarded to the model's encode_* calls
    encode_kwargs = {"batch_size": args.batch_size}
    if args.normalize:
        encode_kwargs["normalize_embeddings"] = True

    logger.info("Starting MTEB evaluation (Italian subsets)...")
    results = mteb.evaluate(model, tasks, encode_kwargs=encode_kwargs)

    # ---- JSON-serializable dump ----
    def _to_dict(obj):
        if hasattr(obj, "model_dump"):  # pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):        # pydantic v1
            return obj.dict()
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(x) for x in obj]
        return obj

    results_serializable = _to_dict(results)

    print("\n===== RESULTS =====")
    print(json.dumps(results_serializable, indent=2, ensure_ascii=False))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results JSON to {args.save_json}")

    logger.info("Done.")


if __name__ == "__main__":
    main()