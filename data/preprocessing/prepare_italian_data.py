import json
import random

# Load queries
with open('RAGEval-italian-selected/dragonball_queries_italian_gold.jsonl') as f:
    queries = [json.loads(line) for line in f]

# Load documents
with open('RAGEval-italian-selected/translated_docs_gold.jsonl') as f:
    docs_list = [json.loads(line) for line in f]

# Create doc_id lookup
docs_by_id = {doc['doc_id']: doc for doc in docs_list}

# Combine into evaluation format
combined = []
for query in queries:
    # Get all documents for this query (all docs from dataset)
    item = {
        'query': {
            'content': query['query']['content'],
            'query_id': query['query']['query_id'],
            'query_type': query['query'].get('query_type', '')
        },
        'ground_truth': {
            'references': query['ground_truth'].get('references', []),
            'relevant_doc_ids': query['ground_truth'].get('doc_ids', []),
            'keypoints': query['ground_truth'].get('keypoints', [])
        },
        'documents': docs_list,  # All docs available for retrieval
        'domain': query.get('domain', ''),
        'language': query.get('language', 'it')
    }
    combined.append(item)

# Shuffle and sample 1000
random.seed(42)
random.shuffle(combined)
sampled = combined[:1000]

print(f" Total queries: {len(queries)}")
print(f" Sampled: {len(sampled)}")
print(f" Total documents in corpus: {len(docs_list)}")

# Save
with open('data/evaluation_data.jsonl', 'w') as f:
    for item in sampled:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f" Saved to: data/evaluation_data.jsonl")
