import argparse
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc
import time

def load_evaluation_data(path: Path):
    queries = []
    all_docs = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                queries.append(item)
                
                for doc in item.get('documents', []):
                    doc_id = doc.get('doc_id')
                    if doc_id not in all_docs:
                        all_docs[doc_id] = doc
                        
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {idx}: {e}")
                continue
    
    docs = list(all_docs.values())
    return queries, docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--topk_values", nargs='+', type=int, default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--use_gpu_faiss", action='store_true')
    args = parser.parse_args()

    print(f"📥 Loading evaluation data...")
    queries, docs = load_evaluation_data(args.query_file)
    
    print(f"✅ Loaded {len(queries)} queries and {len(docs)} unique documents")
    
    print(f"✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_stride,
        separators=["\n\n", "\n", " ", ""],
    )
    
    doc_chunks = []
    for doc in tqdm(docs, desc="Chunking"):
        text = doc.get('content', '')
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        doc_chunks.extend(chunks)
    
    print(f"✅ Created {len(doc_chunks)} chunks")
    
    if len(doc_chunks) == 0:
        print("❌ No chunks created!")
        return

    print(f"🔽 Loading model: {args.model_name}")
    start = time.time()
    model = SentenceTransformer(args.model_name, device=args.device)
    print(f"✅ Model loaded in {time.time()-start:.2f}s")

    print(f"🔢 Encoding {len(doc_chunks)} chunks...")
    start = time.time()
    chunk_embeddings = model.encode(
        doc_chunks,
        batch_size=args.batch_size,
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"✅ Encoding done in {time.time()-start:.2f}s")
    
    print(f"🏗️  Building FAISS index...")
    dimension = chunk_embeddings.shape[1]
    
    if args.use_gpu_faiss and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dimension)
    else:
        index = faiss.IndexFlatIP(dimension)
    
    index.add(chunk_embeddings)
    print(f"✅ FAISS index built with {index.ntotal} vectors")

    max_k = max(args.topk_values)
    print(f"🔍 Retrieving top-{max_k}...")
    
    query_texts = [q['query']['content'] for q in queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    print(f"🔎 Searching...")
    start = time.time()
    scores, indices = index.search(query_embeddings, max_k)
    print(f"✅ Search done in {time.time()-start:.2f}s")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for k in args.topk_values:
        output_file = args.output_dir / f"k{k}_predictions.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as fout:
            for i, query in enumerate(queries):
                topk_indices = indices[i, :k]
                retrieved = [doc_chunks[idx] for idx in topk_indices]
                
                query['prediction'] = {'references': retrieved}
                fout.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved k={k} to {output_file}")
    
    del model, chunk_embeddings, index, query_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
