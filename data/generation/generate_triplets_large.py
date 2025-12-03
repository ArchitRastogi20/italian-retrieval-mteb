"""
Massive Triplet Generator - 10M triplets
- Downloads NEXT 50GB of mC4 Italian (after first 70GB)
- NO duplicate checking (we'll dedupe later)
- Adaptive CPU/RAM management (no OOM, no crashes)
- Smart resource monitoring
- Seed 53
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import time
import psutil
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_gen.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG = {
    # Download - NEXT 50GB after first 70GB
    "dataset_name": "allenai/c4",
    "language": "it",
    "download_offset_gb": 70,  # Skip first 70GB
    "download_size_gb": 50,    # Download next 50GB
    "raw_data_file": "./c4_raw_data_massive/c4_italian_massive.jsonl",
    
    # Output
    "output_file": "./train_random_neg_10M_seed53.jsonl",
    "target_triplets": 10_000_000,
    
    # Document parameters
    "min_doc_length": 100,
    "max_doc_length": 512,
    "min_query_length": 10,
    "max_query_length": 128,
    
    # Adaptive resource management
    "target_ram_percent": 95,
    "safe_ram_percent": 90,
    "emergency_ram_percent": 97,
    "initial_workers": 100,
    "min_workers": 20,
    "max_workers": 120,
    
    # Processing
    "batch_size": 50_000,
    "save_frequency": 100_000,
    "check_resources_every": 10_000,
    
    "seed": 53,
}

random.seed(CONFIG["seed"])


class ResourceMonitor:
    """Adaptive resource monitoring to prevent OOM and crashes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_workers = config["initial_workers"]
        self.cpu_count = psutil.cpu_count()
        self.total_ram_gb = psutil.virtual_memory().total / 1024**3
        
        logger.info(f"ðŸ’» System: {self.cpu_count} CPUs, {self.total_ram_gb:.1f}GB RAM")
        logger.info(f"ðŸŽ¯ Target RAM usage: {config['target_ram_percent']}%")
    
    def get_ram_percent(self) -> float:
        """Get current RAM usage %"""
        return psutil.virtual_memory().percent
    
    def get_cpu_percent(self) -> float:
        """Get current CPU usage %"""
        return psutil.cpu_percent(interval=0.1)
    
    def should_reduce_workers(self) -> bool:
        """Check if we need to reduce workers"""
        ram = self.get_ram_percent()
        
        if ram > self.config["emergency_ram_percent"]:
            logger.warning(f"  ï¸  RAM at {ram:.1f}% - EMERGENCY reduction!")
            return True
        
        if ram > self.config["target_ram_percent"]:
            logger.warning(f"  ï¸  RAM at {ram:.1f}% - reducing workers")
            return True
        
        return False
    
    def should_increase_workers(self) -> bool:
        """Check if we can increase workers"""
        ram = self.get_ram_percent()
        cpu = self.get_cpu_percent()
        
        # Only increase if we have headroom
        if ram < self.config["safe_ram_percent"] and cpu < 80:
            return True
        
        return False
    
    def adjust_workers(self) -> int:
        """Adjust worker count based on resources"""
        old_workers = self.current_workers
        
        if self.should_reduce_workers():
            # Reduce by 20%
            self.current_workers = max(
                self.config["min_workers"],
                int(self.current_workers * 0.8)
            )
        elif self.should_increase_workers():
            # Increase by 10%
            self.current_workers = min(
                self.config["max_workers"],
                int(self.current_workers * 1.1)
            )
        
        if self.current_workers != old_workers:
            logger.info(f"ðŸ”§ Workers: {old_workers} â†’ {self.current_workers}")
        
        return self.current_workers
    
    def log_status(self):
        """Log current resource usage"""
        ram = self.get_ram_percent()
        cpu = self.get_cpu_percent()
        logger.info(f" RAM: {ram:.1f}% | CPU: {cpu:.1f}% | Workers: {self.current_workers}")


def hash_text(text: str) -> str:
    """Fast hash"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_valid_document(text: str, min_len: int, max_len: int) -> bool:
    length = len(text)
    return min_len <= length <= max_len and text.count(' ') > 10


def extract_query(document: str, min_len: int, max_len: int) -> Optional[str]:
    """Extract query"""
    sentences = document.split('.')
    if sentences and len(sentences[0]) >= min_len:
        query = sentences[0][:max_len].strip()
        if len(query) >= min_len:
            return query + '.'
    return None


def get_length_bucket(text: str) -> int:
    """Length bucket"""
    length = len(text)
    if length < 200:
        return 0
    elif length < 350:
        return 1
    else:
        return 2


def process_doc_chunk(args):
    """Worker function - NO duplicate checking"""
    docs, config = args
    
    results = []
    
    for text in docs:
        text = clean_text(text)
        
        if not is_valid_document(text, config["min_doc_length"], config["max_doc_length"]):
            continue
        
        query = extract_query(text, config["min_query_length"], config["max_query_length"])
        if query is None:
            continue
        
        results.append({
            "query": query,
            "positive": text,
            "bucket": get_length_bucket(text)
        })
    
    return results


class C4Downloader:
    """Download next 50GB of C4"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_data_file = Path(config["raw_data_file"])
        self.raw_data_file.parent.mkdir(parents=True, exist_ok=True)
    
    def download(self):
        """Download with FAST skipping"""
        
        if self.raw_data_file.exists():
            current_size_gb = self.raw_data_file.stat().st_size / 1024**3
            logger.info(f" Found existing file: {current_size_gb:.2f} GB")
            
            if current_size_gb >= self.config["download_size_gb"] * 0.95:
                logger.info(" Already have enough data")
                return self.raw_data_file
        
        target_bytes = self.config["download_size_gb"] * 1024**3
        offset_bytes = self.config["download_offset_gb"] * 1024**3
        
        logger.info("="*80)
        logger.info(f" Downloading C4 Italian")
        logger.info(f"   Skip: {self.config['download_offset_gb']}GB")
        logger.info(f"   Download: {self.config['download_size_gb']}GB")
        logger.info("="*80)
        
        dataset = load_dataset(
            self.config["dataset_name"],
            self.config["language"],
            split="train",
            streaming=True
        )
        
        # FAST skip - check byte size instead of counting
        docs_to_skip = int(offset_bytes / 2000)
        logger.info(f"â­ï¸  Fast skipping ~{docs_to_skip:,} documents...")
        logger.info("   (This will take ~5-10 minutes, be patient...)")
        
        iterator = iter(dataset)
        
        # Fast skip without progress bar (much faster!)
        skip_start = time.time()
        for i in range(docs_to_skip):
            try:
                next(iterator)
                # Only log every 1M docs
                if i > 0 and i % 1_000_000 == 0:
                    elapsed = time.time() - skip_start
                    rate = i / elapsed
                    remaining_docs = docs_to_skip - i
                    eta_min = (remaining_docs / rate) / 60
                    logger.info(f"   Skipped {i:,}/{docs_to_skip:,} docs "
                               f"({rate:.0f} docs/s, ETA: {eta_min:.1f} min)")
            except StopIteration:
                logger.error("Reached end!")
                return None
        
        skip_time = time.time() - skip_start
        logger.info(f" Skip complete in {skip_time/60:.1f} minutes")
        logger.info(" Starting download...")
        
        # Download
        bytes_written = 0
        doc_count = 0
        
        with open(self.raw_data_file, 'w', encoding='utf-8', buffering=100*1024*1024) as f:
            with tqdm(total=target_bytes, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for example in iterator:
                    if bytes_written >= target_bytes:
                        break
                    
                    text = example.get('text', '').strip()
                    if text:
                        line = json.dumps({"text": text}, ensure_ascii=False) + '\n'
                        line_bytes = line.encode('utf-8')
                        f.write(line)
                        
                        bytes_written += len(line_bytes)
                        doc_count += 1
                        pbar.update(len(line_bytes))
                        
                        if doc_count % 100_000 == 0:
                            f.flush()
        
        final_gb = bytes_written / 1024**3
        logger.info(f" Downloaded: {doc_count:,} docs, {final_gb:.2f}GB")
        return self.raw_data_file


class MassiveTripletGenerator:
    """Generate 10M triplets with adaptive resource management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_file = Path(config["output_file"])
        self.monitor = ResourceMonitor(config)
        
        # Download data
        downloader = C4Downloader(config)
        self.raw_data_file = downloader.download()
        
        if self.raw_data_file is None:
            raise RuntimeError("Download failed!")
        
        self.triplet_count = 0
        self.processed_docs = 0
        
        # Load all documents to RAM (with monitoring)
        self.all_documents = []
        self.docs_by_bucket = {0: [], 1: [], 2: []}
        self._load_documents_adaptive()
    
    def _load_documents_adaptive(self):
        """Load documents with RAM monitoring"""
        logger.info("ðŸ“‚ Loading documents to RAM (adaptive)...")
        
        with open(self.raw_data_file, 'r', encoding='utf-8', buffering=100*1024*1024) as f:
            
            for line in tqdm(f, desc="Loading"):
                # Check RAM every 10K docs
                if len(self.all_documents) % 10_000 == 0:
                    ram = self.monitor.get_ram_percent()
                    
                    if ram > self.config["emergency_ram_percent"]:
                        logger.warning(f"  ï¸  RAM at {ram:.1f}% - stopping load")
                        break
                
                try:
                    data = json.loads(line)
                    text = clean_text(data['text'])
                    
                    if is_valid_document(text, 
                                        self.config["min_doc_length"],
                                        self.config["max_doc_length"]):
                        self.all_documents.append(text)
                        
                        bucket = get_length_bucket(text)
                        self.docs_by_bucket[bucket].append(text)
                except:
                    continue
        
        logger.info(f" Loaded {len(self.all_documents):,} documents")
        logger.info(f"   Buckets: 0={len(self.docs_by_bucket[0]):,}, "
                   f"1={len(self.docs_by_bucket[1]):,}, "
                   f"2={len(self.docs_by_bucket[2]):,}")
        self.monitor.log_status()
    
    def generate_triplets(self):
        """Generate with adaptive processing"""
        logger.info("="*80)
        logger.info("ðŸŽ¯ Generating 10M Triplets - Adaptive Mode")
        logger.info(f"Source: {len(self.all_documents):,} documents")
        logger.info("="*80)
        
        start_time = time.time()
        batch_size = self.config["batch_size"]
        total_docs = len(self.all_documents)
        
        with open(self.output_file, 'w', encoding='utf-8', buffering=100*1024*1024) as out_f:
            
            with tqdm(total=self.config["target_triplets"], desc="Generating", smoothing=0.01) as pbar:
                
                batch_count = 0
                
                while self.triplet_count < self.config["target_triplets"]:
                    
                    # Get batch (cycle through documents)
                    start_idx = (batch_count * batch_size) % total_docs
                    end_idx = min(start_idx + batch_size, total_docs)
                    
                    if end_idx <= start_idx:
                        # Wrap around
                        batch = self.all_documents[start_idx:] + self.all_documents[:batch_size - (total_docs - start_idx)]
                    else:
                        batch = self.all_documents[start_idx:end_idx]
                    
                    # Process batch with current worker count
                    triplets = self._process_batch_adaptive(batch)
                    
                    # Write
                    for triplet in triplets:
                        if self.triplet_count >= self.config["target_triplets"]:
                            break
                        
                        json.dump(triplet, out_f, ensure_ascii=False)
                        out_f.write('\n')
                        self.triplet_count += 1
                    
                    pbar.update(len(triplets))
                    
                    # Stats
                    elapsed = time.time() - start_time
                    speed = self.triplet_count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'speed': f'{speed:.0f}/s',
                        'workers': self.monitor.current_workers,
                        'ram': f'{self.monitor.get_ram_percent():.0f}%'
                    })
                    
                    # Resource management
                    if self.triplet_count % self.config["check_resources_every"] == 0:
                        self.monitor.adjust_workers()
                    
                    # Flush
                    if self.triplet_count % self.config["save_frequency"] == 0:
                        out_f.flush()
                        self.monitor.log_status()
                    
                    batch_count += 1
        
        elapsed = time.time() - start_time
        
        logger.info("="*80)
        logger.info(f" COMPLETE!")
        logger.info(f"Generated: {self.triplet_count:,} triplets")
        logger.info(f"Time: {elapsed / 60:.1f} min ({elapsed / 3600:.2f} hrs)")
        logger.info(f"Speed: {self.triplet_count / elapsed:.0f} triplets/sec")
        logger.info(f"Output: {self.output_file}")
        logger.info("="*80)
    
    def _process_batch_adaptive(self, batch: List[str]) -> List[Dict]:
        """Process batch with adaptive worker count"""
        
        # Split into worker chunks
        workers = self.monitor.current_workers
        chunk_size = max(1, len(batch) // workers)
        
        chunks = []
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i + chunk_size]
            chunks.append((chunk, self.config))
        
        # Process in parallel
        all_processed = []
        
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_doc_chunk, chunk) for chunk in chunks]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        all_processed.extend(result)
                    except Exception as e:
                        continue
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Reduce workers for next batch
            self.monitor.current_workers = max(
                self.config["min_workers"],
                int(self.monitor.current_workers * 0.7)
            )
            return []
        
        if not all_processed:
            return []
        
        # Create triplets
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
            
            # Fast random sample
            pos_hash = hash_text(item["positive"])
            
            for _ in range(5):
                negative = random.choice(candidates)
                if hash_text(negative) != pos_hash:
                    break
            
            triplet = {
                "query": item["query"],
                "positive": item["positive"],
                "negative": negative
            }
            
            triplets.append(triplet)
        
        return triplets


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n  ï¸  Interrupted! Saving progress...")
    sys.exit(0)


def main():
    """Main entry"""
    
    signal.signal(signal.SIGINT, signal_handler)
    
    mem = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    logger.info("="*80)
    logger.info("ðŸš€ MASSIVE Triplet Generator - 10M Triplets")
    logger.info(f"System: {cpu_count} CPUs, {mem.total/1024**3:.1f}GB RAM")
    logger.info(f"Seed: {CONFIG['seed']}")
    logger.info("="*80)
    logger.info("")
    
    generator = MassiveTripletGenerator(CONFIG)
    generator.generate_triplets()
    
    logger.info(" All done!")


if __name__ == "__main__":
    main()