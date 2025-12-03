import math

def _normalize_text(text):
    return ' '.join(text.lower().split())

class NDCG:
    name: str = "NDCG"
    
    def __call__(self, doc, ground_truth, results, language="it"):
        retrieves = doc["prediction"].get("references", [])
        gt_refs = doc['ground_truth'].get('references', [])
        
        if not retrieves or not gt_refs:
            return 0.0
        
        # Calculate relevance scores (1 if match, 0 otherwise)
        relevance = []
        gt_norms = [_normalize_text(gt) for gt in gt_refs]
        
        for ret in retrieves:
            ret_norm = _normalize_text(ret)
            is_relevant = any(
                ret_norm == gt or ret_norm in gt or gt in ret_norm
                for gt in gt_norms
            )
            relevance.append(1 if is_relevant else 0)
        
        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        
        # IDCG (ideal)
        ideal = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        
        return dcg / idcg if idcg > 0 else 0.0
