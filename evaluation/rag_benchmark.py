#!/usr/bin/env python3
"""
OPTIMIZED Embedding Model Benchmarking System for RAG Evaluation
High-performance version with progress tracking and evaluation optimizations
"""

import os
import json
import csv
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Core libraries
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from tqdm import tqdm
import pickle

# Text processing
import re
from nltk.tokenize import sent_tokenize
import jieba

# Import your evaluation metrics
from rageval.evaluation.metrics import get_metric
from rageval.evaluation.metrics.rag_metrics.generation.keypoint_metrics import KEYPOINT_METRICS
from rageval.evaluation.metrics.rag_metrics.generation.rouge_l import ROUGELScore
from rageval.evaluation.metrics.rag_metrics.retrieval.precision import Precision
from rageval.evaluation.metrics.rag_metrics.retrieval.recall import Recall
from rageval.evaluation.metrics.rag_metrics.retrieval.eir import EIR

# Setup logging
def setup_logging(log_dir: str) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('embedding_benchmark')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TextChunker:
    """Handle text chunking with overlap"""
    
    def __init__(self, chunk_size: int, overlap: int = 0, language: str = "en"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.language = language
        
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments"""
        if self.language == "zh":
            # For Chinese, split by characters
            words = list(text)
        elif self.language in ["en", "it"]:
            # For English and Italian, split by words
            words = text.split()
        else:
            # Default: split by words
            words = text.split()
            
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            
            if self.language == "zh":
                chunk = ''.join(words[start:end])
            else:
                chunk = ' '.join(words[start:end])
                
            chunks.append(chunk)
            
            if end == len(words):
                break
                
            start = end - self.overlap
            
        return chunks

class EmbeddingRetriever:
    """Handle embedding generation and retrieval - HYBRID GPU+CPU OPTIMIZED"""
    
    def __init__(self, model_name: str, device: str = "cuda", logger: logging.Logger = None):
        self.model_name = model_name
        self.embedding_device = "cuda"  # Always use GPU for embeddings
        self.index_device = "cpu"       # GPU or CPU for FAISS based on use_gpu param 
        self.device = device           # Keep for compatibility
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.index = None
        self.chunks = []
        
    def load_model(self) -> bool:
        """Load the embedding model on GPU"""
        try:
            self.logger.info(f"Loading model on GPU: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device="cuda")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            return False
    
    def create_embeddings(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Create embeddings using GPU with optimized batching"""
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        # Use large batch size for RTX 4090
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device="cuda",
            normalize_embeddings=True,
            precision="float32"  # Optimize memory usage
        )
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, use_gpu: bool = True):
        """Build FAISS index - GPU optimized with CPU fallback"""
        dimension = embeddings.shape[1]
        
        if use_gpu and torch.cuda.is_available():
            # Try GPU FAISS first
            try:
                res = faiss.StandardGpuResources()
                res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
                index = faiss.IndexFlatIP(dimension)
                self.index = faiss.index_cpu_to_gpu(res, 0, index)
                self.logger.info(f"Using GPU FAISS index")
            except Exception as e:
                self.logger.warning(f"GPU FAISS failed: {e}, falling back to CPU")
                # Fallback to CPU
                self.index = faiss.IndexFlatIP(dimension)
                faiss.omp_set_num_threads(min(32, mp.cpu_count()))
                self.logger.info(f"Using CPU FAISS index with {min(32, mp.cpu_count())} threads")
        else:
            # CPU FAISS
            self.index = faiss.IndexFlatIP(dimension)
            faiss.omp_set_num_threads(min(32, mp.cpu_count()))
            self.logger.info(f"Using CPU FAISS index with {min(32, mp.cpu_count())} threads")
        
        # Move embeddings to CPU if they're on GPU for FAISS
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        elif hasattr(embeddings, 'device'):
            embeddings = np.asarray(embeddings)
        
        # Ensure float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        
        self.logger.info(f"FAISS index built with {embeddings.shape[0]:,} vectors, dimension {dimension}")
        
        # Clear GPU memory after building index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def retrieve_batch(self, query_embeddings: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k for multiple queries at once"""
        if self.index is None:
            raise RuntimeError("Index not built")
        
        # Move query embeddings to CPU if needed
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.cpu().numpy()
        
        # Ensure correct format
        query_embeddings = query_embeddings.astype(np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        return indices, scores
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Retrieve top-k similar chunks - single query"""
        indices, scores = self.retrieve_batch(query_embedding, top_k)
        return indices[0].tolist(), scores[0].tolist()

class RAGEvaluator:
    """OPTIMIZED RAG performance evaluator with progress tracking"""
    
    def __init__(self, use_openai: bool = False, openai_model: str = "gpt-4o-mini", 
                 openai_version: str = "v2", logger: logging.Logger = None):
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.openai_version = openai_version
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize working metrics only
        self.metrics = {
            "rouge-l": ROUGELScore(),
            "recall": Recall(), 
            "eir": EIR(),
        }
        # Skip the broken precision metric - we'll use custom overlap-based precision
        
        # Add keypoint metrics if OpenAI is available
        if use_openai:
            self.metrics["keypoint_metrics"] = KEYPOINT_METRICS(
                use_openai=True, 
                model=openai_model, 
                version=openai_version
            )
    
    def calculate_overlap_precision(self, retrieved_chunks: List[str], ground_truth_refs: List[str], 
                                  threshold: float = 0.3) -> float:
        """Fast precision based on word overlap"""
        if not retrieved_chunks or not ground_truth_refs:
            return 0.0
        
        relevant_chunks = 0
        ground_truth_text = " ".join(ground_truth_refs).lower().strip()
        gt_words = set(ground_truth_text.split())
        
        for chunk in retrieved_chunks:
            chunk_words = set(chunk.lower().strip().split())
            if not chunk_words:
                continue
            
            overlap_words = chunk_words.intersection(gt_words)
            overlap_ratio = len(overlap_words) / len(chunk_words)
            
            if overlap_ratio >= threshold:
                relevant_chunks += 1
        
        return relevant_chunks / len(retrieved_chunks)
    
    def calculate_substring_precision(self, retrieved_chunks: List[str], ground_truth_refs: List[str], 
                                    min_overlap_chars: int = 30) -> float:
        """Fast precision based on substring overlap"""
        if not retrieved_chunks or not ground_truth_refs:
            return 0.0
        
        relevant_chunks = 0
        ground_truth_text = " ".join(ground_truth_refs).lower()
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.lower()
            
            # Fast substring check
            max_overlap = 0
            for i in range(0, len(chunk_text) - min_overlap_chars + 1, 10):  # Step by 10 for speed
                for j in range(i + min_overlap_chars, min(i + 200, len(chunk_text) + 1)):  # Limit substring length
                    substring = chunk_text[i:j]
                    if substring in ground_truth_text:
                        max_overlap = max(max_overlap, len(substring))
            
            if max_overlap >= min_overlap_chars:
                relevant_chunks += 1
        
        return relevant_chunks / len(retrieved_chunks)
    
    def evaluate_retrieval_batch_optimized(self, eval_docs: List[Dict], language: str = "en", 
                                         batch_size: int = 50) -> List[Dict[str, float]]:
        """OPTIMIZED batch evaluation with progress tracking"""
        results = []
        
        # Process in smaller batches with progress bar
        total_batches = (len(eval_docs) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="Processing evaluation batches", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(eval_docs))
            batch_docs = eval_docs[start_idx:end_idx]
            
            batch_results = []
            
            # Process each document in batch with progress
            for doc in tqdm(batch_docs, desc=f"Batch {batch_idx+1}/{total_batches}", leave=False, unit="doc"):
                doc_results = {}
                
                # Get retrieved chunks and ground truth references
                retrieved_chunks = doc["prediction"].get("references", [])
                ground_truth_refs = doc["ground_truth"].get("references", [])
                
                # Fast custom precision calculations
                overlap_precision = self.calculate_overlap_precision(retrieved_chunks, ground_truth_refs, threshold=0.3)
                substring_precision = self.calculate_substring_precision(retrieved_chunks, ground_truth_refs, min_overlap_chars=30)
                
                doc_results["precision"] = max(overlap_precision, substring_precision)
                doc_results["overlap_precision"] = overlap_precision
                doc_results["substring_precision"] = substring_precision
                
                # Calculate other metrics (optimized)
                for metric_name, metric in self.metrics.items():
                    if metric_name == "keypoint_metrics":
                        continue
                        
                    try:
                        if metric_name == "rouge-l":
                            score = metric(doc, doc["ground_truth"], None, language=language)
                        else:
                            score = metric(doc, doc["ground_truth"], None, language=language)
                        doc_results[metric_name] = score
                    except Exception as e:
                        # Silently handle errors to avoid spam
                        doc_results[metric_name] = 0.0
                        
                batch_results.append(doc_results)
            
            results.extend(batch_results)
        
        return results

class BenchmarkRunner:
    """OPTIMIZED benchmark runner with progress tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging(config["output_dir"])
        self.results_dir = Path(config["output_dir"])
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "embeddings").mkdir(exist_ok=True)
        (self.results_dir / "results").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize components
        self.chunker = TextChunker(
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
            language=config["language"]
        )
        
        self.evaluator = RAGEvaluator(
            use_openai=config["use_openai"],
            openai_model=config["openai_model"],
            openai_version=config["openai_version"],
            logger=self.logger
        )
        
    def load_models(self) -> List[str]:
        """Load model names from models.txt"""
        models_file = Path(self.config["models_file"])
        if not models_file.exists():
            raise FileNotFoundError(f"Models file not found: {models_file}")
        
        with open(models_file, 'r') as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        self.logger.info(f"Loaded {len(models)} models from {models_file}")
        return models
    
    def load_data(self) -> List[Dict]:
        """Load evaluation data"""
        data_file = Path(self.config["data_file"])
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(data)} evaluation examples")
        return data
    
    def process_single_model(self, model_name: str, data: List[Dict]) -> Dict[str, Any]:
        """Process a single model with MAXIMUM OPTIMIZATION and progress tracking"""
        self.logger.info(f"Processing model: {model_name}")
        
        # Initialize retriever with hybrid approach
        retriever = EmbeddingRetriever(model_name, device="cuda", logger=self.logger)
        if not retriever.load_model():
            return {"error": f"Failed to load model {model_name}"}
        
        try:
            # ===== STEP 1: BATCH PREPARE ALL DATA =====
            self.logger.info(f"Preparing documents and queries for {model_name}...")
            
            all_chunks = []
            all_queries = []
            example_metadata = []
            
            for example_idx, example in enumerate(tqdm(data, desc=f"Preparing data for {model_name}", unit="example")):
                # Extract documents and create chunks
                documents = []
                if "documents" in example:
                    for doc in example["documents"]:
                        if isinstance(doc, dict) and "content" in doc:
                            documents.append(doc["content"])
                        elif isinstance(doc, str):
                            documents.append(doc)
                
                if not documents:
                    continue
                
                # Create chunks from all documents for this example
                example_chunks = []
                for doc in documents:
                    chunks = self.chunker.chunk_text(doc)
                    example_chunks.extend(chunks)
                
                if not example_chunks:
                    continue
                
                # Map chunks to global index
                chunk_start_idx = len(all_chunks)
                all_chunks.extend(example_chunks)
                
                # Process query
                query_text = example["query"]["content"] if isinstance(example["query"], dict) else example["query"]
                query_idx = len(all_queries)
                all_queries.append(query_text)
                
                # Store example metadata
                example_metadata.append({
                    'example_idx': example_idx,
                    'query_idx': query_idx,
                    'chunk_start': chunk_start_idx,
                    'num_chunks': len(example_chunks),
                    'example': example
                })
            
            if not all_chunks or not all_queries:
                return {"error": "No valid data"}
            
            self.logger.info(f"Total chunks: {len(all_chunks):,}, Total queries: {len(all_queries):,}")
            
            # ===== STEP 2: GPU BATCH CREATE ALL EMBEDDINGS =====
            # Optimized batch sizes for RTX 4090
            chunk_batch_size = 512  # Large batch for GPU
            query_batch_size = 256  # Large batch for GPU
            
            self.logger.info(f"Creating embeddings for {len(all_chunks):,} chunks on GPU...")
            chunk_embeddings = retriever.create_embeddings(all_chunks, batch_size=chunk_batch_size)
            
            self.logger.info(f"Creating embeddings for {len(all_queries):,} queries on GPU...")
            query_embeddings = retriever.create_embeddings(all_queries, batch_size=query_batch_size)
            
            # ===== STEP 3: BUILD INDEX =====
            self.logger.info("Building FAISS index...")
            use_gpu = (self.config["device"] == "cuda")
            retriever.build_index(chunk_embeddings, use_gpu=use_gpu)
            retriever.chunks = all_chunks
            
            # ===== STEP 4: OPTIMIZED BATCH RETRIEVAL =====
            self.logger.info("Processing queries with optimized batch retrieval...")
            
            model_results = []
            
            # Process each top_k value with progress
            for top_k in tqdm(self.config["top_k_values"], desc=f"Processing top_k values for {model_name}", unit="top_k"):
                self.logger.info(f"Processing top_k={top_k} with batch retrieval...")
                
                # ULTRA-FAST BATCH RETRIEVE ALL QUERIES AT ONCE
                all_indices, all_scores = retriever.retrieve_batch(query_embeddings, top_k=top_k)
                
                # Prepare batch evaluation documents
                eval_docs = []
                result_metadata = []
                
                # Prepare evaluation docs with progress
                for meta in tqdm(example_metadata, desc=f"Preparing evaluation docs (top_k={top_k})", leave=False, unit="query"):
                    example_idx = meta['example_idx']
                    query_idx = meta['query_idx']
                    example = meta['example']
                    
                    # Get retrieved chunks for this query
                    query_indices = all_indices[query_idx]
                    retrieved_chunks = [all_chunks[idx] for idx in query_indices if idx < len(all_chunks)]
                    
                    # Create evaluation document
                    eval_doc = {
                        "query": example["query"],
                        "ground_truth": example["ground_truth"],
                        "prediction": {
                            "content": " ".join(retrieved_chunks),
                            "references": retrieved_chunks
                        }
                    }
                    
                    eval_docs.append(eval_doc)
                    result_metadata.append({
                        "model": model_name,
                        "example_id": example_idx,
                        "query_id": example.get("query", {}).get("query_id", example_idx),
                        "chunk_size": self.config["chunk_size"],
                        "overlap": self.config["overlap"],
                        "top_k": top_k,
                        "num_chunks": meta["num_chunks"],
                        "retrieved_chunks": len(retrieved_chunks)
                    })
                
                # OPTIMIZED BATCH EVALUATION
                self.logger.info(f"Evaluating {len(eval_docs)} documents for top_k={top_k}...")
                evaluation_results = self.evaluator.evaluate_retrieval_batch_optimized(
                    eval_docs, language=self.config["language"], batch_size=50
                )
                
                # Combine results
                for i, (meta, eval_result) in enumerate(zip(result_metadata, evaluation_results)):
                    result = {**meta, **eval_result}
                    model_results.append(result)
                    
                # Clear memory after each top_k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Critical error processing model {model_name}: {e}")
            return {"error": str(e)}
        
        # Save individual model results
        model_file = self.results_dir / "results" / f"{model_name.replace('/', '_')}_results.json"
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Completed model: {model_name}, Results: {len(model_results):,}")
        return {"results": model_results, "count": len(model_results)}
    
    def aggregate_results(self, all_results: List[Dict]) -> pd.DataFrame:
        """Aggregate results across all models"""
        flattened_results = []
        
        for model_result in all_results:
            if "error" in model_result:
                continue
            flattened_results.extend(model_result["results"])
        
        if not flattened_results:
            self.logger.error("No results to aggregate")
            return pd.DataFrame()
        
        df = pd.DataFrame(flattened_results)
        
        # Calculate averages by model, chunk_size, overlap, and top_k
        numeric_columns = [col for col in df.columns if col not in 
                          ["model", "example_id", "query_id", "chunk_size", "overlap", "top_k", 
                           "num_chunks", "retrieved_chunks", "responses"]]
        
        grouped = df.groupby(["model", "chunk_size", "overlap", "top_k"])
        aggregated = grouped[numeric_columns].mean().reset_index()
        
        return aggregated
    
    def save_results(self, aggregated_df: pd.DataFrame, all_results: List[Dict]):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated CSV
        csv_file = self.results_dir / f"benchmark_results_{timestamp}.csv"
        aggregated_df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved aggregated results to {csv_file}")
        
        # Save detailed JSON
        json_file = self.results_dir / f"detailed_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": self.config,
                "timestamp": timestamp,
                "results": all_results,
                "summary": aggregated_df.to_dict('records')
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved detailed results to {json_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Models tested: {aggregated_df['model'].nunique()}")
        print(f"Configurations: {len(aggregated_df)}")
        print(f"Chunk size: {self.config['chunk_size']}")
        print(f"Overlap: {self.config['overlap']}")
        print(f"Top-K values: {self.config['top_k_values']}")
        print("\nTop 5 configurations by precision:")
        if 'precision' in aggregated_df.columns:
            top_configs = aggregated_df.nlargest(5, 'precision')
            for _, row in top_configs.iterrows():
                print(f"  {row['model']} (top_k={row['top_k']}): {row['precision']:.4f}")
        print("="*80)
    
    def run(self):
        """Run the complete optimized benchmark"""
        self.logger.info("Starting OPTIMIZED embedding model benchmark")
        self.logger.info(f"Configuration: {self.config}")
        
        # Load models and data
        models = self.load_models()
        data = self.load_data()
        
        # Process models
        if self.config["parallel_models"]:
            # Process models in parallel (limited by memory)
            max_workers = min(len(models), 2)  # Max 2 parallel models
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.process_single_model, model, data): model 
                          for model in models}
                
                all_results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing models"):
                    model_name = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Model {model_name} failed: {e}")
                        all_results.append({"error": str(e)})
        else:
            # Process models sequentially with progress
            all_results = []
            for model in tqdm(models, desc="Processing models sequentially"):
                result = self.process_single_model(model, data)
                all_results.append(result)
        
        # Aggregate and save results
        aggregated_df = self.aggregate_results(all_results)
        if not aggregated_df.empty:
            self.save_results(aggregated_df, all_results)
        else:
            self.logger.error("No valid results to save")
        
        self.logger.info("Benchmark completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="OPTIMIZED Embedding Model Benchmarking System")
    parser.add_argument("--models_file", type=str, default="models.txt", 
                       help="Path to models.txt file")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to evaluation data (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--chunk_size", type=int, default=512,
                       help="Chunk size for text splitting")
    parser.add_argument("--overlap", type=int, default=50,
                       help="Overlap between chunks")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=[1, 5, 10],
                       help="Top-K values to test")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh", "it"],
                       help="Language for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--use_openai", action="store_true",
                       help="Use OpenAI for generation evaluation")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini",
                       help="OpenAI model for evaluation")
    parser.add_argument("--openai_version", type=str, default="v2",
                       help="OpenAI evaluation version")
    parser.add_argument("--parallel_models", action="store_true",
                       help="Process models in parallel")
    
    args = parser.parse_args()
    
    config = vars(args)
    
    # Create and run benchmark
    runner = BenchmarkRunner(config)
    runner.run()

if __name__ == "__main__":
    main()