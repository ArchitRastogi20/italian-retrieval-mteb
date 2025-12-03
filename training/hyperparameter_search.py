#!/usr/bin/env python3
"""
E5-Small-v2 Italian Fine-tuning with Optuna Hyperparameter Optimization
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses
from transformers import get_linear_schedule_with_warmup
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from tqdm.auto import tqdm
import numpy as np

# Efficiency knobs
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int):
        self.data = []
        logger.info(f"Loading {jsonl_path} (max {max_samples:,})...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=max_samples, desc="Loading")):
                if i >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                    self.data.append({
                        'query': item['query'].strip(),
                        'positive': item['positive'].strip(),
                        'negative': item['negative'].strip()
                    })
                except:
                    continue
        
        logger.info(f"Loaded {len(self.data):,} triplets")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, model):
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch]
    
    query_features = model.tokenize(queries)
    pos_features = model.tokenize(positives)
    neg_features = model.tokenize(negatives)
    
    return [query_features, pos_features, neg_features]

def run_mteb_eval(model_path: str, step: int, checkpoint_dir: str) -> Dict[str, float]:
    """Run MTEB evaluation with high batch size"""
    output_json = os.path.join(checkpoint_dir, f"mteb_{step}.json")
    logger.info(f"ðŸ” MTEB eval @ step {step}")
    
    try:
        result = subprocess.run(
            ["python", "mteb_eval_ita_retrieval.py",
             "--model", model_path,
             "--batch-size", "1024",  # High batch size for speed
             "--fp16",
             "--save-json", output_json],
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        with open(output_json, 'r') as f:
            results = json.load(f)
        
        # Parse results
        metrics = {}
        if isinstance(results, list):
            for task_result in results:
                if 'scores' in task_result and 'test' in task_result['scores']:
                    for split in task_result['scores']['test']:
                        task_name = task_result.get('task_name', 'unknown')
                        subset = split.get('hf_subset', 'unknown')
                        
                        if 'ndcg_at_10' in split:
                            metrics[f"{task_name}_{subset}_ndcg@10"] = split['ndcg_at_10']
                        if 'ndcg_at_1' in split:
                            metrics[f"{task_name}_{subset}_ndcg@1"] = split['ndcg_at_1']
        else:
            # Try alternate format
            for task_name, task_results in results.items():
                if isinstance(task_results, dict) and 'test' in task_results:
                    for split in task_results['test']:
                        subset = split.get('hf_subset', 'unknown')
                        if 'ndcg_at_10' in split:
                            metrics[f"{task_name}_{subset}_ndcg@10"] = split['ndcg_at_10']
                        if 'ndcg_at_1' in split:
                            metrics[f"{task_name}_{subset}_ndcg@1"] = split['ndcg_at_1']
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"MTEB eval timeout @ step {step}")
        return {}
    except Exception as e:
        logger.error(f"MTEB failed: {e}")
        return {}

class EarlyStoppingCallback:
    """Early stopping based on evaluation metric"""
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.warning(f"  ï¸ Early stopping triggered! No improvement for {self.patience} evals")
                return True
        return False

def train_single_run(
    model: SentenceTransformer,
    dataloader: DataLoader,
    cfg: Dict,
    trial: Optional[optuna.Trial] = None,
    is_trial: bool = False
) -> Dict[str, float]:
    """Single training run with optional trial reporting"""
    
    steps_per_epoch = len(dataloader.dataset) // cfg["eff_batch"]
    total_steps = steps_per_epoch * cfg["epochs"]
    
    logger.info(f"{'='*80}")
    logger.info(f"Training: {len(dataloader.dataset):,} samples, {total_steps} steps")
    logger.info(f"LR={cfg['lr']:.2e}, Batch={cfg['eff_batch']}, Temp={cfg['temperature']}")
    logger.info(f"{'='*80}")
    
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
    
    # Mixed precision
    use_amp = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Early stopping
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.002)
    
    # Training state
    model.train()
    global_step = 0
    best_wiki = 0.0
    best_step = 0
    eval_history = []
    
    for epoch in range(cfg["epochs"]):
        logger.info(f"\n{'='*80}\nEpoch {epoch+1}/{cfg['epochs']}\n{'='*80}")
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=is_trial)
        
        for step, features in enumerate(pbar):
            # Move to device
            for feat in features:
                for k in feat:
                    feat[k] = feat[k].to(model.device)
            
            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(features, None) / cfg["grad_accum"]
            
            # Backward
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            
            # Update
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                lr = scheduler.get_last_lr()[0]
                step_loss = loss.item() * cfg["grad_accum"]
                
                if not is_trial:
                    pbar.set_postfix({
                        'loss': f'{step_loss:.4f}',
                        'grad': f'{grad_norm:.2f}',
                        'lr': f'{lr:.2e}',
                        'step': global_step
                    })
                
                # Logging
                if global_step % 10 == 0 and not is_trial:
                    wandb.log({
                        "train/loss": step_loss,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/lr": lr,
                        "train/epoch": epoch + step/len(dataloader),
                        "train/step": global_step,
                    })
                
                # Save checkpoint (only in final run)
                if not is_trial and global_step % cfg["save_steps"] == 0:
                    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"checkpoint-{global_step}")
                    logger.info(f"\nðŸ’¾ Saving: {ckpt_path}")
                    model.save(ckpt_path)
                    
                    # Cleanup old
                    ckpts = sorted([
                        d for d in Path(cfg["checkpoint_dir"]).iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")
                    ], key=lambda x: int(x.name.split("-")[1]))
                    for old_ckpt in ckpts[:-3]:
                        shutil.rmtree(old_ckpt)
                
                # Evaluation
                if global_step % cfg["eval_steps"] == 0:
                    logger.info(f"\n{'='*80}\nðŸ” Eval @ step {global_step}\n{'='*80}")
                    
                    # Save temp checkpoint
                    eval_path = os.path.join(cfg["checkpoint_dir"], f"eval-{global_step}")
                    model.save(eval_path)
                    
                    # Run MTEB
                    metrics = run_mteb_eval(eval_path, global_step, cfg["checkpoint_dir"])
                    
                    if metrics:
                        wiki = metrics.get("WikipediaRetrievalMultilingual_it_ndcg@10", 0)
                        bele = metrics.get("BelebeleRetrieval_ita_Latn-ita_Latn_ndcg@10", 0)
                        
                        logger.info(f" Results @ step {global_step}:")
                        logger.info(f"   Wiki nDCG@10: {wiki:.4f} (baseline: 0.7704)")
                        logger.info(f"   Bele nDCG@10: {bele:.4f} (baseline: 0.6898)")
                        
                        eval_history.append({
                            'step': global_step,
                            'wiki': wiki,
                            'bele': bele
                        })
                        
                        # Report to trial
                        if trial is not None:
                            trial.report(wiki, global_step)
                            if trial.should_prune():
                                logger.warning("  ï¸ Trial pruned by Optuna")
                                raise optuna.TrialPruned()
                        
                        # Log to wandb
                        if not is_trial:
                            for k, v in metrics.items():
                                wandb.log({f"eval/{k}": v, "train/step": global_step})
                        
                        # Track best
                        if wiki > best_wiki:
                            best_wiki = wiki
                            best_step = global_step
                            
                            if not is_trial:
                                best_path = os.path.join(cfg["checkpoint_dir"], "best")
                                if os.path.exists(best_path):
                                    shutil.rmtree(best_path)
                                shutil.copytree(eval_path, best_path)
                                logger.info(f"âœ¨ NEW BEST! Wiki: {wiki:.4f} (+{wiki-0.7704:.4f})")
                                wandb.log({"eval/best_wiki": best_wiki})
                        
                        # Early stopping
                        if early_stopping(wiki):
                            logger.info(f"ðŸ›‘ Early stopping at step {global_step}")
                            return {
                                'best_wiki': best_wiki,
                                'best_step': best_step,
                                'final_wiki': wiki,
                                'eval_history': eval_history
                            }
                    
                    model.train()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"\n Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")
        if not is_trial:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch_num": epoch+1})
    
    return {
        'best_wiki': best_wiki,
        'best_step': best_step,
        'final_wiki': eval_history[-1]['wiki'] if eval_history else 0,
        'eval_history': eval_history
    }

def objective(trial: optuna.Trial, base_cfg: Dict) -> float:
    """Optuna objective function"""
    
    # Suggest hyperparameters
    cfg = base_cfg.copy()
    cfg.update({
        "lr": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96, 128]),
        "temperature": trial.suggest_float("temperature", 0.05, 0.10),
        "warmup_steps": trial.suggest_int("warmup_steps", 50, 200),
        "max_samples": 100_000,  # Use fewer samples for trials
        "checkpoint_dir": f"/workspace/optuna_trials/trial_{trial.number}",
    })
    cfg["eff_batch"] = cfg["batch_size"] * cfg["grad_accum"]
    
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Trial {trial.number}")
    logger.info(f"Params: {trial.params}")
    logger.info(f"{'='*80}")
    
    # Load model
    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]
    
    if hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
        model[0].auto_model.gradient_checkpointing_enable()
    
    # Load small dataset for trial
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
    
    # Train
    try:
        result = train_single_run(model, dataloader, cfg, trial=trial, is_trial=True)
        return result['best_wiki']
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return 0.0

def main():
    """Main training pipeline"""
    
    # Base config
    base_cfg = {
        "base_model": "intfloat/e5-small-v2",
        "train_file": "/workspace/it-retrieval-triplets-mc4/train.jsonl",
        "grad_accum": 2,
        "epochs": 1,
        "weight_decay": 0.01,
        "max_seq_length": 128,
        "num_workers": 12,
        "save_steps": 300,
        "eval_steps": 300,
    }
    
    # Phase 1: Hyperparameter search
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    study = optuna.create_study(
        study_name=f"e5_italian_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    )
    
    # Run optimization
    n_trials = 8  # Adjust based on time
    logger.info(f"Running {n_trials} trials...")
    
    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials=n_trials,
        timeout=3600 * 2,  # 2 hour max for search
        show_progress_bar=True
    )
    
    # Best params
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Wiki nDCG@10: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    logger.info("="*80)
    
    # Save study
    study_path = "/workspace/optuna_study.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"Study saved to {study_path}")
    
    # Phase 2: Final training with best params
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FINAL TRAINING WITH BEST HYPERPARAMETERS")
    logger.info("="*80)
    
    final_cfg = base_cfg.copy()
    final_cfg.update({
        "lr": study.best_params["lr"],
        "batch_size": study.best_params["batch_size"],
        "temperature": study.best_params["temperature"],
        "warmup_steps": study.best_params["warmup_steps"],
        "max_samples": 300_000,  # Use more data for final run
        "checkpoint_dir": "/workspace/checkpoints_final_optimized",
    })
    final_cfg["eff_batch"] = final_cfg["batch_size"] * final_cfg["grad_accum"]
    
    os.makedirs(final_cfg["checkpoint_dir"], exist_ok=True)
    
    # Initialize wandb for final run
    wandb.init(
        project="italian-e5-small-v2-optimized",
        name=f"final-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=final_cfg
    )
    
    # Load model
    model = SentenceTransformer(final_cfg["base_model"])
    model.max_seq_length = final_cfg["max_seq_length"]
    
    if hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
        model[0].auto_model.gradient_checkpointing_enable()
    
    # Load full dataset
    dataset = TripletDataset(final_cfg["train_file"], final_cfg["max_samples"])
    dataloader = DataLoader(
        dataset,
        batch_size=final_cfg["batch_size"],
        shuffle=True,
        num_workers=final_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=lambda batch: collate_fn(batch, model)
    )
    
    # Train
    result = train_single_run(model, dataloader, final_cfg, is_trial=False)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("ðŸŽ‰ TRAINING COMPLETE!")
    logger.info(f"Baseline: Wiki=0.7704, Bele=0.6898")
    logger.info(f"Best: Wiki={result['best_wiki']:.4f} (+{result['best_wiki']-0.7704:.4f}) @ step {result['best_step']}")
    logger.info(f"Final: Wiki={result['final_wiki']:.4f}")
    logger.info(f"Best model: {os.path.join(final_cfg['checkpoint_dir'], 'best')}")
    logger.info("="*80)
    
    wandb.finish()

if __name__ == "__main__":
    main()