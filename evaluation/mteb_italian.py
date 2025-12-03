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
             
"""

def parse_args():
    ap = argparse.ArgumentParser(description="MTEB v2 Italian Retrieval (Wiki + Belebele) with logging/tqdm")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--batch-size", type=int, default=128, help="Encode batch size")
    ap.add_argument("--fp16", action="store_true", help="Try FP16 on GPU (best-effort)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings in encode()")
    ap.add_argument("--save-json", default="", help="Optional path to save results JSON")
    return ap.parse_args()

def maybe_enable_fp16(mteb_model):
    """
    Best-effort: if the MTEB wrapper exposes a SentenceTransformer under .model,
    put it in FP16 for speed on Ampere+ (e.g., RTX 3080 Ti).
    """
    if not torch.cuda.is_available():
        return False
    try:
        st = getattr(mteb_model, "model", None)
        if st is not None and hasattr(st, "half"):
            st.half()
            logger.info("Using FP16 (half precision) for faster encoding.")
            return True
    except Exception as e:
        logger.warning(f"FP16 not applied (continuing in FP32): {e}")
    return False

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
    # Build tasks with explicit Italian subsets
    # -------------------------
    tasks = []
    try:
        bele = mteb.get_task(
            "BelebeleRetrieval",
            # run ONLY the Italian subset
            hf_subsets=["ita_Latn-ita_Latn"],
        )
        tasks.append(bele)
    except Exception as e:
        logger.warning(f"Could not create BelebeleRetrieval task (ita_Latn-ita_Latn): {e}")

    try:
        wiki = mteb.get_task(
            "WikipediaRetrievalMultilingual",
            # for this task the subset id for Italian is just 'it'
            hf_subsets=["it"],
            # (optional) also signal language in ISO 639-3 form
            languages=["ita"],
        )
        tasks.append(wiki)
    except Exception as e:
        logger.warning(f"Could not create WikipediaRetrievalMultilingual task (it): {e}")

    if not tasks:
        logger.error("No tasks to evaluate after configuration. Exiting.")
        return

    logger.info(f"Tasks to evaluate: {[t.metadata.name for t in tasks]}")

    # Batch size / normalization forwarded to the modelâ€™s encode_* calls
    encode_kwargs = {"batch_size": args.batch_size}
    if args.normalize:
        encode_kwargs["normalize_embeddings"] = True

    logger.info("Starting MTEB evaluation (Italian-only subsets)...")
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
