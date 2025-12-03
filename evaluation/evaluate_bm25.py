#!/usr/bin/env python3
"""
BM25 Retrieval Evaluation with NLTK Stopwords Toggle
Evaluates BM25 on Italian dataset with/without stopwords at different k values
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import csv
import sys

# Download NLTK stopwords
try:
    stopwords.words('italian')
except LookupError:
    print(" Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Import metrics
sys.path.insert(0, str(Path(__file__).parent / 'rageval' / 'evaluation'))
from metrics import get_metric


def tokenize(text: str, remove_stopwords: bool = False, language: str = 'italian') -> List[str]:
    """Tokenize text with optional stopword removal"""
    # Simple tokenization
    tokens = text.lower().split()
    
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words(language))
            tokens = [t for t in tokens if t not in stop_words]
        except:
            print(f"  ï¸  Warning: No stopwords available for {language}")
    
    return tokens


def chunk_documents(documents: List[dict], chunk_size: int = 512, chunk_stride: int = 256) -> Tuple[List[str], List[int]]:
    """Chunk documents and track doc_ids"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_stride,
        separators=["\n\n", "\n", " ", ""],
    )
    
    all_chunks = []
    chunk_to_doc_id = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        content = doc.get('content', '')
        if not content:
            continue
        
        chunks = splitter.split_text(content)
        all_chunks.extend(chunks)
        chunk_to_doc_id.extend([doc['doc_id']] * len(chunks))
    
    return all_chunks, chunk_to_doc_id


def build_bm25_index(chunks: List[str], remove_stopwords: bool = False) -> BM25Okapi:
    """Build BM25 index from chunks"""
    print(f"ðŸ—ï¸  Building BM25 index (stopwords={'removed' if remove_stopwords else 'kept'})...")
    
    tokenized_corpus = [tokenize(chunk, remove_stopwords=remove_stopwords) for chunk in tqdm(chunks, desc="Tokenizing")]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def retrieve_bm25(query: str, bm25: BM25Okapi, chunks: List[str], k: int, remove_stopwords: bool = False) -> List[str]:
    """Retrieve top-k chunks using BM25"""
    query_tokens = tokenize(query, remove_stopwords=remove_stopwords)
    
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    return [chunks[i] for i in top_indices]


def calculate_metrics(predictions_file: Path, language: str = 'it') -> dict:
    """Calculate all metrics from predictions file"""
    metrics_dict = {}
    
    for metric_name in ['recall', 'eir', 'ndcg', 'mrr']:
        metric_class = get_metric(metric_name)
        metric_instance = metric_class()
        
        scores = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                ground_truth = item["ground_truth"]
                
                score = metric_instance(item, ground_truth, None, language=language)
                scores.append(score)
        
        metrics_dict[metric_name] = sum(scores) / len(scores) if scores else 0.0
    
    return metrics_dict


def load_evaluation_data(path: Path) -> Tuple[List[dict], List[dict]]:
    """Load queries and documents from evaluation file"""
    queries = []
    all_docs = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            queries.append(item)
            
            for doc in item.get('documents', []):
                doc_id = doc.get('doc_id')
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
    
    docs = list(all_docs.values())
    return queries, docs


def main():
    parser = argparse.ArgumentParser(description='BM25 Retrieval Evaluation')
    parser.add_argument('--query_file', type=Path, default='data/evaluation_data_answer_focused.jsonl')
    parser.add_argument('--output_csv', type=Path, default='results/bm25_results.csv')
    parser.add_argument('--topk_values', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--chunk_stride', type=int, default=256)
    parser.add_argument('--language', type=str, default='italian')
    args = parser.parse_args()
    
    print("="*80)
    print("ðŸ” BM25 Retrieval Evaluation")
    print("="*80)
    print(f"Query file: {args.query_file}")
    print(f"TopK values: {args.topk_values}")
    print(f"Language: {args.language}")
    print(f"Stopwords: Testing both WITH and WITHOUT")
    print("="*80)
    
    # Load data
    print("\n Loading evaluation data...")
    queries, docs = load_evaluation_data(args.query_file)
    print(f" Loaded {len(queries)} queries and {len(docs)} documents")
    
    # Chunk documents
    print(f"\nâœ‚ï¸  Chunking documents (size={args.chunk_size}, stride={args.chunk_stride})...")
    chunks, chunk_doc_ids = chunk_documents(docs, args.chunk_size, args.chunk_stride)
    print(f" Created {len(chunks)} chunks")
    
    # Prepare output
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results = []
    
    # Test with and without stopwords
    for use_stopwords in [False, True]:
        stopwords_label = "without_stopwords" if use_stopwords else "with_stopwords"
        print(f"\n{'='*80}")
        print(f"Testing: {stopwords_label}")
        print(f"{'='*80}")
        
        # Build BM25 index
        bm25 = build_bm25_index(chunks, remove_stopwords=use_stopwords)
        
        # Retrieve for each k value
        for k in args.topk_values:
            print(f"\n Retrieving top-{k} ({stopwords_label})...")
            
            # Create predictions file
            temp_pred_file = Path(f"temp_bm25_{stopwords_label}_k{k}.jsonl")
            
            with open(temp_pred_file, 'w', encoding='utf-8') as f:
                for query in tqdm(queries, desc=f"Retrieving k={k}"):
                    query_text = query['query']['content']
                    
                    # Retrieve
                    retrieved = retrieve_bm25(query_text, bm25, chunks, k, remove_stopwords=use_stopwords)
                    
                    # Save prediction
                    query['prediction'] = {'references': retrieved}
                    f.write(json.dumps(query, ensure_ascii=False) + '\n')
            
            # Calculate metrics
            print(f"  Calculating metrics for k={k}...")
            metrics = calculate_metrics(temp_pred_file, language='it')
            
            # Store results
            result = {
                'model': f'BM25_{stopwords_label}',
                'k': k,
                'recall': metrics['recall'],
                'eir': metrics['eir'],
                'ndcg': metrics['ndcg'],
                'mrr': metrics['mrr']
            }
            results.append(result)
            
            print(f"   k={k:3d}: Recall={metrics['recall']:.4f} EIR={metrics['eir']:.4f} NDCG={metrics['ndcg']:.4f} MRR={metrics['mrr']:.4f}")
            
            # Clean up temp file
            temp_pred_file.unlink()
    
    # Save results to CSV
    print(f"\nðŸ’¾ Saving results to {args.output_csv}...")
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'k', 'recall', 'eir', 'ndcg', 'mrr'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f" Results saved!")
    
    # Print summary table
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print(f"{'Model':<40} {'k':>5} {'Recall':>8} {'EIR':>8} {'NDCG':>8} {'MRR':>8}")
    print("-"*80)
    for result in results:
        print(f"{result['model']:<40} {result['k']:>5} {result['recall']:>8.4f} {result['eir']:>8.4f} {result['ndcg']:>8.4f} {result['mrr']:>8.4f}")
    print("="*80)
    
    print(f"\n Complete! Results: {args.output_csv}")


if __name__ == '__main__':
    main()
