#!/usr/bin/env python3

import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm.auto import tqdm

# Efficiency knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

logging.basicConfig(format="%(asctime)s | %(levelname)-8s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int):
        self.data = []
        logger.info(f"Loading {jsonl_path} (RAW, no prefixes)...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=max_samples, desc="Loading")):
                if i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    # NO PREFIXES - use raw text
                    self.data.append({
                        'query': item['query'].strip(),
                        'positive': item['positive'].strip(),
                        'negative': item['negative'].strip()
                    })
                except:
                    continue
        
        logger.info(f"Loaded {len(self.data):,} triplets (no formatting)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, model):
    """Smart batching collate"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch]
    
    # Tokenize
    query_features = model.tokenize(queries)
    pos_features = model.tokenize(positives)
    neg_features = model.tokenize(negatives)
    
    return [query_features, pos_features, neg_features]

def run_mteb_quick(model_path: str, step: int) -> Dict[str, float]:
    output_json = f"/workspace/checkpoints/mteb_{step}.json"
    logger.info(f"Quick MTEB eval @ step {step}")
    
    try:
        result = subprocess.run(
            ["python", "mteb_eval_ita_retrieval.py",
             "--model", model_path,
             "--batch-size", "256",
             "--fp16",
             "--save-json", output_json],
            check=True,
            capture_output=True,
            text=True,
            timeout=900
        )
        logger.info(f"MTEB stdout: {result.stdout[-200:]}")  # Log last 200 chars
        
        with open(output_json, 'r') as f:
            results = json.load(f)
        
        metrics = {}
        for task_name, task_results in results.items():
            if isinstance(task_results, dict) and 'test' in task_results:
                for split in task_results['test']:
                    subset = split.get('hf_subset', 'unknown')
                    if 'ndcg_at_10' in split:
                        metrics[f"{task_name}_{subset}_ndcg@10"] = split['ndcg_at_10']
                    if 'ndcg_at_1' in split:
                        metrics[f"{task_name}_{subset}_ndcg@1"] = split['ndcg_at_1']
        
        return metrics
    except subprocess.CalledProcessError as e:
        logger.error(f"MTEB failed with exit code {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        logger.error(f"STDOUT: {e.stdout}")
        return {}
    except Exception as e:
        logger.error(f"MTEB failed: {e}")
        return {}

def main():
    cfg = {
        "base_model": "intfloat/multilingual-e5-small",
        "train_file": "/workspace/it-retrieval-triplets-mc4/train_random_neg.jsonl",  # â† Random negs
        "max_samples": 500_000,
        "batch_size": 192,
        "grad_accum": 2,
        "epochs": 1,
        "lr": 5e-6,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "temperature": 0.05,
        "max_seq_length": 128,
        "num_workers": 12,
        "checkpoint_dir": "/workspace/checkpoints_v4_random",
        "save_steps": 500,
        "eval_steps": 500,
    }
    cfg["eff_batch"] = cfg["batch_size"] * cfg["grad_accum"]
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    
    wandb.init(
        project="italian-e5-retrieval",
        name=f"e5-it-noprefix-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=cfg
    )
    
    logger.info("="*80)
    logger.info("Italian E5 Fine-tuning [NO PREFIXES]")
    logger.info(json.dumps(cfg, indent=2))
    logger.info("="*80)
    
    # Load model
    logger.info("Loading model...")
    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]
    
    if hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
        model[0].auto_model.gradient_checkpointing_enable()
    
    logger.info(f"Device: {model.device}")
    
    # Load data WITHOUT prefixes
    dataset = TripletDataset(cfg["train_file"], cfg["max_samples"])
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, model)
    )
    
    steps_per_epoch = len(dataset) // cfg["eff_batch"]
    total_steps = steps_per_epoch * cfg["epochs"]
    
    logger.info(f"Dataset: {len(dataset):,}")
    logger.info(f"Effective batch: {cfg['eff_batch']}")
    logger.info(f"Steps/epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Warmup: {cfg['warmup_steps']} ({100*cfg['warmup_steps']/total_steps:.1f}%)")
    
    # Loss
    loss_fn = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=1.0 / cfg["temperature"]
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        eps=1e-6,
        weight_decay=cfg["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps
    )
    
    # Scaler for mixed precision
    use_amp = torch.cuda.is_bf16_supported()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training state
    model.train()
    global_step = 0
    best_wiki = 0.0
    
    logger.info(f"Mixed precision: {'BF16' if use_amp else 'FP32'}")
    logger.info("Starting training...")
    
    for epoch in range(cfg["epochs"]):
        logger.info(f"\n{'='*80}\nEpoch {epoch+1}/{cfg['epochs']}\n{'='*80}")
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, features in enumerate(pbar):
            # Move to device
            for feat in features:
                for k in feat:
                    feat[k] = feat[k].to(model.device)
            
            # Forward with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(features, None) / cfg["grad_accum"]
            
            # Backward
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            
            # Update
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                lr = scheduler.get_last_lr()[0]
                step_loss = loss.item() * cfg["grad_accum"]
                
                pbar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': global_step
                })
                
                # Log to wandb
                if global_step % 10 == 0:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + step/len(dataloader),
                        "train/step": global_step,
                    })
                
                # Save checkpoint
                if global_step % cfg["save_steps"] == 0:
                    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"checkpoint-{global_step}")
                    logger.info(f"\nðŸ’¾ Saving checkpoint: {ckpt_path}")
                    model.save(ckpt_path)
                    
                    # Cleanup old checkpoints
                    ckpts = sorted([
                        d for d in Path(cfg["checkpoint_dir"]).iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")
                    ], key=lambda x: int(x.name.split("-")[1]))
                    
                    for old_ckpt in ckpts[:-5]:
                        shutil.rmtree(old_ckpt)
                
                # Eval
                if global_step % cfg["eval_steps"] == 0:
                    logger.info(f"\n{'='*80}\nðŸ” MTEB Eval @ step {global_step}\n{'='*80}")
                    
                    # Save temp checkpoint
                    eval_path = os.path.join(cfg["checkpoint_dir"], f"eval-{global_step}")
                    model.save(eval_path)
                    
                    # Run MTEB
                    metrics = run_mteb_quick(eval_path, global_step)
                    
                    if metrics:
                        logger.info(f"Results @ step {global_step}:")
                        for k, v in metrics.items():
                            logger.info(f"  {k}: {v:.4f}")
                            wandb.log({f"eval/{k}": v, "train/step": global_step})
                        
                        # Track best
                        wiki = metrics.get("WikipediaRetrievalMultilingual_it_ndcg@10", 0)
                        bele = metrics.get("BelebeleRetrieval_ita_Latn-ita_Latn_ndcg@10", 0)
                        
                        logger.info(f"   Wiki: {wiki:.4f} (baseline: 0.8931)")
                        logger.info(f"   Bele: {bele:.4f} (baseline: 0.9238)")
                        
                        if wiki > best_wiki:
                            best_wiki = wiki
                            best_path = os.path.join(cfg["checkpoint_dir"], "best")
                            if os.path.exists(best_path):
                                shutil.rmtree(best_path)
                            shutil.copytree(eval_path, best_path)
                            logger.info(f"âœ¨ NEW BEST! Wiki nDCG@10: {wiki:.4f}")
                            wandb.log({"eval/best_wiki": best_wiki})
                    
                    model.train()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"\n Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "train/epoch_num": epoch+1})
    
    # Final eval
    logger.info("\n" + "="*80 + "\nðŸ Final Evaluation\n" + "="*80)
    
    final_path = os.path.join(cfg["checkpoint_dir"], "final")
    model.save(final_path)
    
    final_metrics = run_mteb_quick(final_path, total_steps)
    
    if final_metrics:
        logger.info("Final Results:")
        for k, v in final_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            wandb.log({f"final/{k}": v})
        
        wiki_final = final_metrics.get("WikipediaRetrievalMultilingual_it_ndcg@10", 0)
        if wiki_final > best_wiki:
            best_wiki = wiki_final
            best_path = os.path.join(cfg["checkpoint_dir"], "best")
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(final_path, best_path)
    
    logger.info("\n" + "="*80)
    logger.info("ðŸŽ‰ TRAINING COMPLETE!")
    logger.info(f"Baseline Wiki: 0.8931 | Baseline Bele: 0.9238")
    logger.info(f"Best Wiki nDCG@10: {best_wiki:.4f} ({best_wiki-0.8931:+.4f})")
    logger.info(f"Best model: {os.path.join(cfg['checkpoint_dir'], 'best')}")
    logger.info("="*80)
    
    wandb.finish()

if __name__ == "__main__":
    main()