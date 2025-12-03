#!/usr/bin/env python3
"""
Fixed merge script - properly matches queries with their relevant documents
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_documents(documents_file: str) -> Dict[int, Dict]:
    """Load documents and create doc_id -> document mapping"""
    logger = logging.getLogger(__name__)
    
    documents = {}
    with open(documents_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc['doc_id']
                documents[doc_id] = doc
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def merge_files_fixed(queries_file: str, documents_file: str, output_file: str):
    """FIXED: Properly merge queries with their specific relevant documents"""
    logger = setup_logging()
    
    # Load documents
    documents = load_documents(documents_file)
    
    # Process queries and create evaluation data
    evaluation_data = []
    skipped_queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                query_data = json.loads(line)
                
                # Extract relevant document IDs from ground truth
                relevant_doc_ids = query_data['ground_truth'].get('doc_ids', [])
                
                if not relevant_doc_ids:
                    logger.warning(f"Query {query_data['query']['query_id']} has no doc_ids, skipping")
                    skipped_queries.append(query_data['query']['query_id'])
                    continue
                
                # Get the actual relevant documents
                query_documents = []
                missing_docs = []
                
                for doc_id in relevant_doc_ids:
                    if doc_id in documents:
                        query_documents.append({
                            "doc_id": doc_id,
                            "content": documents[doc_id]['content']
                        })
                    else:
                        missing_docs.append(doc_id)
                
                if missing_docs:
                    logger.warning(f"Query {query_data['query']['query_id']} references missing doc_ids: {missing_docs}")
                
                if not query_documents:
                    logger.warning(f"Query {query_data['query']['query_id']} has no valid documents, skipping")
                    skipped_queries.append(query_data['query']['query_id'])
                    continue
                
                # Add some additional candidate documents for retrieval challenge
                # This creates a more realistic scenario where the relevant docs are mixed with irrelevant ones
                candidate_doc_ids = list(documents.keys())
                additional_candidates = [
                    doc_id for doc_id in candidate_doc_ids[:100]  # First 100 as candidates
                    if doc_id not in relevant_doc_ids  # Don't duplicate relevant docs
                ][:20]  # Add up to 20 additional candidates
                
                for doc_id in additional_candidates:
                    query_documents.append({
                        "doc_id": doc_id,
                        "content": documents[doc_id]['content']
                    })
                
                # Verify ground truth references exist in the documents
                ground_truth_refs = query_data['ground_truth'].get('references', [])
                references_found = []
                
                for ref in ground_truth_refs:
                    ref_found = False
                    for doc in query_documents[:len(relevant_doc_ids)]:  # Only check relevant docs
                        if ref.lower() in doc['content'].lower():
                            ref_found = True
                            break
                    if ref_found:
                        references_found.append(ref)
                    else:
                        logger.warning(f"Reference not found in documents: {ref[:50]}...")
                
                if not references_found and ground_truth_refs:
                    logger.error(f"Query {query_data['query']['query_id']}: NO ground truth references found in documents!")
                    # Still include it but log the issue
                
                # Create evaluation item in expected format
                eval_item = {
                    "query": {
                        "query_id": query_data['query']['query_id'],
                        "content": query_data['query']['content']
                    },
                    "documents": query_documents,
                    "ground_truth": {
                        "content": query_data['ground_truth']['content'],
                        "references": query_data['ground_truth']['references'],
                        "keypoints": query_data['ground_truth'].get('keypoints', []),
                        "relevant_doc_ids": relevant_doc_ids  # Keep track of which docs are actually relevant
                    },
                    "language": query_data['language'],
                    "domain": query_data.get('domain', 'Unknown'),
                    "query_type": query_data['query'].get('query_type', 'Unknown')
                }
                
                evaluation_data.append(eval_item)
                
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                continue
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in evaluation_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f" Created evaluation file with {len(evaluation_data)} queries: {output_file}")
    if skipped_queries:
        logger.info(f"  ï¸ Skipped {len(skipped_queries)} queries due to missing documents")
    
    # Verification step
    logger.info("ðŸ” Verification: Checking first few queries...")
    for i, item in enumerate(evaluation_data[:3]):
        query_id = item['query']['query_id']
        num_docs = len(item['documents'])
        num_refs = len(item['ground_truth']['references'])
        relevant_docs = len(item['ground_truth']['relevant_doc_ids'])
        logger.info(f"Query {query_id}: {num_docs} total docs ({relevant_docs} relevant), {num_refs} ground truth refs")
    
    return len(evaluation_data)

if __name__ == "__main__":
    # File paths
    queries_file = "data/queries.jsonl"
    documents_file = "data/documents.jsonl" 
    output_file = "data/evaluation_data.jsonl"
    
    # Check if input files exist
    if not Path(queries_file).exists():
        print(f" Error: {queries_file} not found")
        exit(1)
        
    if not Path(documents_file).exists():
        print(f" Error: {documents_file} not found")
        exit(1)
    
    # Merge files with the fixed logic
    print("ðŸ”„ Merging queries with their SPECIFIC relevant documents...")
    count = merge_files_fixed(queries_file, documents_file, output_file)
    
    print(f" Successfully created FIXED {output_file} with {count} evaluation items")
    print(f"ðŸ“ Now ground truth references should be found in the document corpus!")
    print(f"ðŸš€ Ready to run: ./run_ir_benchmark.sh")
    
    # Quick verification
    print("\nðŸ” Quick verification...")
    import json
    with open(output_file, 'r') as f:
        first_item = json.loads(f.readline())
        print(f"First query: {first_item['query']['content'][:80]}...")
        print(f"Documents provided: {len(first_item['documents'])}")
        print(f"Relevant doc IDs: {first_item['ground_truth']['relevant_doc_ids']}")
        print(f"Ground truth refs: {len(first_item['ground_truth']['references'])}")