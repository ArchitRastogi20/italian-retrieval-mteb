"""
BLAZING FAST 15M Triplet Generator
- NO duplicate checking
- Uses 100GB C4 data already downloaded
- 95% CPU and RAM usage
- Saves every 500K triplets
- Clean logging
- Seed 53
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import psutil
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

CONFIG = {
    # Use downloaded 100GB data
    "raw_data_file": "./c4_100gb/c4_italian_100gb.jsonl",
    "output_file": "./train_15M_seed73.jsonl",
    "target_triplets": 15_000_000,
    
    # Document parameters
    "min_doc_length": 100,
    "max_doc_length": 512,
    "min_query_length": 10,
    "max_query_length": 128,
    
    # Processing - MAX POWER
    "num_workers": 120,  # 95% of 128 CPUs
    "batch_size": 30_000,
    "save_frequency": 500_000,
    
    "seed": 73,
}

random.seed(CONFIG["seed"])


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_valid_document(text: str, min_len: int, max_len: int) -> bool:
    length = len(text)
    return min_len <= length <= max_len and text.count(' ') > 10


def extract_query(document: str, min_len: int, max_len: int) -> Optional[str]:
    sentences = document.split('.')
    if sentences and len(sentences[0]) >= min_len:
        query = sentences[0][:max_len].strip()
        if len(query) >= min_len:
            return query + '.'
    return None


def get_length_bucket(text: str) -> int:
    length = len(text)
    if length < 200:
        return 0
    elif length < 350:
        return 1
    else:
        return 2


def process_doc_chunk(docs_config_tuple):
    """Worker function - NO shared state"""
    docs, min_doc, max_doc, min_q, max_q = docs_config_tuple
    
    results = []
    
    for text in docs:
        text = clean_text(text)
        
        if not is_valid_document(text, min_doc, max_doc):
            continue
        
        query = extract_query(text, min_q, max_q)
        if query is None:
            continue
        
        results.append({
            "query": query,
            "positive": text,
            "bucket": get_length_bucket(text)
        })
    
    return results


class BlazingFastGenerator:
    """Super fast 15M generator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_file = Path(config["output_file"])
        
        # Clear output file if exists
        if self.output_file.exists():
            self.output_file.unlink()
        
        logger.info("="*80)
        logger.info("ðŸš€ BLAZING FAST 15M Triplet Generator")
        logger.info(f"Workers: {config['num_workers']}")
        logger.info(f"Target: {config['target_triplets']:,} triplets")
        logger.info("="*80)
        
        # Load all documents to RAM
        logger.info("ðŸ“‚ Loading documents to RAM...")
        load_start = time.time()
        
        self.all_documents = []
        self.docs_by_bucket = {0: [], 1: [], 2: []}
        
        with open(Path(config["raw_data_file"]), 'r', encoding='utf-8', buffering=100*1024*1024) as f:
            for line in tqdm(f, desc="Loading", unit=" docs"):
                # Stop if RAM > 95%
                if len(self.all_documents) % 50_000 == 0:
                    if psutil.virtual_memory().percent > 95:
                        logger.warning("  ï¸  RAM limit reached")
                        break
                
                try:
                    data = json.loads(line)
                    text = clean_text(data['text'])
                    
                    if is_valid_document(text, config["min_doc_length"], 
                                        config["max_doc_length"]):
                        self.all_documents.append(text)
                        bucket = get_length_bucket(text)
                        self.docs_by_bucket[bucket].append(text)
                except:
                    continue
        
        load_time = time.time() - load_start
        
        logger.info(f" Loaded {len(self.all_documents):,} documents in {load_time/60:.1f} min")
        logger.info(f"   Buckets: 0={len(self.docs_by_bucket[0]):,}, "
                   f"1={len(self.docs_by_bucket[1]):,}, "
                   f"2={len(self.docs_by_bucket[2]):,}")
        logger.info(f"   RAM: {psutil.virtual_memory().percent:.1f}%")
        logger.info("")
        
        self.triplet_count = 0
    
    def generate(self):
        """Generate 15M triplets"""
        logger.info("ðŸŽ¯ Starting generation...")
        logger.info("")
        
        start_time = time.time()
        last_save_time = start_time
        last_save_count = 0
        
        total_docs = len(self.all_documents)
        batch_size = self.config["batch_size"]
        batch_count = 0
        
        # Estimate for 1M triplets (run first batch to measure)
        estimation_done = False
        
        with open(self.output_file, 'w', encoding='utf-8', buffering=100*1024*1024) as out_f:
            
            with tqdm(total=self.config["target_triplets"], 
                     desc="Progress", unit=" triplets", smoothing=0.01) as pbar:
                
                while self.triplet_count < self.config["target_triplets"]:
                    
                    # Get batch (cycle through documents)
                    start_idx = (batch_count * batch_size) % total_docs
                    end_idx = start_idx + batch_size
                    
                    if end_idx > total_docs:
                        batch = self.all_documents[start_idx:] + \
                                self.all_documents[:end_idx - total_docs]
                    else:
                        batch = self.all_documents[start_idx:end_idx]
                    
                    # Process batch
                    triplets = self._process_batch(batch)
                    
                    # Write triplets
                    for triplet in triplets:
                        if self.triplet_count >= self.config["target_triplets"]:
                            break
                        
                        json.dump(triplet, out_f, ensure_ascii=False)
                        out_f.write('\n')
                        self.triplet_count += 1
                    
                    pbar.update(len(triplets))
                    
                    # Estimate after first meaningful batch
                    if not estimation_done and self.triplet_count > 1000:
                        elapsed = time.time() - start_time
                        speed = self.triplet_count / elapsed
                        time_for_1m = (1_000_000 / speed) / 60  # minutes
                        
                        logger.info(f"â±ï¸  Estimated time for 1M triplets: {time_for_1m:.1f} minutes")
                        logger.info(f"   Current speed: {speed:.0f} triplets/sec")
                        logger.info("")
                        
                        estimation_done = True
                    
                    # Save checkpoint every 500K
                    if self.triplet_count >= last_save_count + self.config["save_frequency"]:
                        out_f.flush()
                        
                        # Calculate stats for this 500K
                        current_time = time.time()
                        batch_elapsed = current_time - last_save_time
                        batch_speed = self.config["save_frequency"] / batch_elapsed
                        
                        total_elapsed = current_time - start_time
                        overall_speed = self.triplet_count / total_elapsed
                        
                        remaining = self.config["target_triplets"] - self.triplet_count
                        eta_min = (remaining / overall_speed) / 60
                        
                        logger.info(f"ðŸ’¾ Checkpoint: {self.triplet_count:,} triplets")
                        logger.info(f"   Last 500K speed: {batch_speed:.0f} t/s")
                        logger.info(f"   Overall speed: {overall_speed:.0f} t/s")
                        logger.info(f"   ETA: {eta_min:.1f} minutes")
                        logger.info(f"   RAM: {psutil.virtual_memory().percent:.1f}%")
                        logger.info(f"   CPU: {psutil.cpu_percent(interval=0.1):.1f}%")
                        logger.info("")
                        
                        last_save_time = current_time
                        last_save_count = self.triplet_count
                    
                    batch_count += 1
        
        # Final stats
        total_elapsed = time.time() - start_time
        
        logger.info("="*80)
        logger.info(" COMPLETE!")
        logger.info(f"Generated: {self.triplet_count:,} triplets")
        logger.info(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} hrs)")
        logger.info(f"Average speed: {self.triplet_count / total_elapsed:.0f} triplets/sec")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"File size: {self.output_file.stat().st_size / 1024**3:.2f} GB")
        logger.info("="*80)
    
    def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process batch with max parallelism"""
        
        workers = self.config["num_workers"]
        chunk_size = max(1, len(batch) // workers)
        
        # Prepare chunks with config (no shared objects!)
        chunks = []
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]
            chunks.append((
                chunk,
                self.config["min_doc_length"],
                self.config["max_doc_length"],
                self.config["min_query_length"],
                self.config["max_query_length"]
            ))
        
        # Process in parallel
        all_processed = []
        
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_doc_chunk, chunk) for chunk in chunks]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        all_processed.extend(result)
                    except:
                        continue
        except:
            return []
        
        if not all_processed:
            return []
        
        # Create triplets with random negatives
        triplets = []
        
        for item in all_processed:
            if self.triplet_count + len(triplets) >= self.config["target_triplets"]:
                break
            
            # Sample negative from same bucket
            bucket = item["bucket"]
            candidates = self.docs_by_bucket[bucket]
            
            if len(candidates) < 2:
                candidates = self.all_documents
            
            if not candidates:
                continue
            
            # Random negative
            negative = random.choice(candidates)
            
            triplet = {
                "query": item["query"],
                "positive": item["positive"],
                "negative": negative
            }
            
            triplets.append(triplet)
        
        return triplets


def signal_handler(sig, frame):
    logger.info("\n  ï¸  Interrupted! Progress saved.")
    sys.exit(0)


def main():
    """Main entry"""
    
    signal.signal(signal.SIGINT, signal_handler)
    
    generator = BlazingFastGenerator(CONFIG)
    generator.generate()


if __name__ == "__main__":
    main()