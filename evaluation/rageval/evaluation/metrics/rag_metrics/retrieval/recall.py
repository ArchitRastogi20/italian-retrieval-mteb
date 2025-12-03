def _normalize_text(text):
    return ' '.join(text.lower().split())

def chunk_match(query_chunk, retrieved_chunks, threshold=0.8):
    query_norm = _normalize_text(query_chunk)
    
    for retrieved_chunk in retrieved_chunks:
        retrieved_norm = _normalize_text(retrieved_chunk)
        
        if query_norm == retrieved_norm or query_norm in retrieved_norm or retrieved_norm in query_norm:
            return True
        
        query_words = set(query_norm.split())
        retrieved_words = set(retrieved_norm.split())
        
        if len(query_words) > 0:
            overlap = len(query_words & retrieved_words) / len(query_words)
            if overlap >= threshold:
                return True
    
    return False

class Recall:
    name: str = "Recall"
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def calculate_recall(self, retrieves, ground_truths, language=None):
        if not ground_truths or not retrieves:
            return 0.0
        
        match_count = sum(
            1 for gt_chunk in ground_truths
            if chunk_match(gt_chunk, retrieves, self.threshold)
        )
        
        return match_count / len(ground_truths)

    def __call__(self, doc, ground_truth, results, language=None):
        retrieves = [r for r in doc["prediction"].get("references", [])]
        ground_truths = doc['ground_truth'].get('references', [])
        return self.calculate_recall(retrieves, ground_truths, language=language)
