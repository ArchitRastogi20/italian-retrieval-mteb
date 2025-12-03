from .utils import exist_match

class EIR:
    name: str = "EIR"
    
    def __call__(self, doc, ground_truth, results, language="it"):
        retrieves = doc["prediction"].get("references", [])
        gt_refs = doc['ground_truth'].get('references', [])
        
        if not retrieves or not gt_refs:
            return 0.0
        
        total_words = sum(len(r.split()) for r in retrieves)
        matched_words = 0
        
        for gt in gt_refs:
            for ret in retrieves:
                if exist_match(gt, [ret], language=language):
                    matched_words += len(gt.split())
                    break
        
        return matched_words / total_words if total_words > 0 else 0.0
