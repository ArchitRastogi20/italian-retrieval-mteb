import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def chunk_text(text, chunk_size=512, chunk_stride=256):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_stride,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def find_chunks_containing_text(search_text, document_chunks, threshold=0.5):
    search_text_lower = search_text.lower().strip()
    search_words = set(search_text_lower.split())
    
    matching_chunks = []
    
    for chunk in document_chunks:
        chunk_lower = chunk.lower()
        
        if search_text_lower in chunk_lower:
            matching_chunks.append(chunk)
            continue
        
        chunk_words = set(chunk_lower.split())
        overlap = len(search_words & chunk_words) / len(search_words) if search_words else 0
        
        if overlap >= threshold:
            matching_chunks.append(chunk)
    
    return matching_chunks

with open('data/evaluation_data.jsonl') as f:
    items = [json.loads(line) for line in f if line.strip()]

fixed_items = []
stats = {'total': 0, 'success': 0, 'no_refs': 0, 'no_match': 0}

for item in tqdm(items, desc="Creating GT"):
    stats['total'] += 1
    
    original_refs = item['ground_truth'].get('references', [])
    relevant_doc_ids = set(item['ground_truth'].get('relevant_doc_ids', []))
    
    if not original_refs or not relevant_doc_ids:
        stats['no_refs'] += 1
        continue
    
    all_docs = item.get('documents', [])
    relevant_docs = [doc for doc in all_docs if doc['doc_id'] in relevant_doc_ids]
    
    if not relevant_docs:
        stats['no_match'] += 1
        continue
    
    all_chunks = []
    for doc in relevant_docs:
        content = doc.get('content', '')
        if content:
            chunks = chunk_text(content)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        stats['no_match'] += 1
        continue
    
    gt_chunks = []
    for ref in original_refs:
        matching_chunks = find_chunks_containing_text(ref, all_chunks, threshold=0.5)
        gt_chunks.extend(matching_chunks)
    
    seen = set()
    unique_gt_chunks = []
    for chunk in gt_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_gt_chunks.append(chunk)
    
    if not unique_gt_chunks:
        stats['no_match'] += 1
        continue
    
    item['ground_truth']['references'] = unique_gt_chunks[:5]
    fixed_items.append(item)
    stats['success'] += 1

with open('data/evaluation_data_answer_focused.jsonl', 'w') as f:
    for item in fixed_items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n Successfully processed {stats['success']}/{stats['total']} items")
print(f"   No references: {stats['no_refs']}")
print(f"   No matches: {stats['no_match']}")
