def _normalize_text(text):
    return ' '.join(text.lower().split())

class MRR:
    name: str = "MRR"
    
    def __call__(self, doc, ground_truth, results, language="it"):
        retrieves = doc["prediction"].get("references", [])
        gt_refs = doc['ground_truth'].get('references', [])
        
        if not retrieves or not gt_refs:
            return 0.0
        
        gt_norms = [_normalize_text(gt) for gt in gt_refs]
        
        for i, ret in enumerate(retrieves, 1):
            ret_norm = _normalize_text(ret)
            if any(ret_norm == gt or ret_norm in gt or gt in ret_norm for gt in gt_norms):
                return 1.0 / i
        
        return 0.0
