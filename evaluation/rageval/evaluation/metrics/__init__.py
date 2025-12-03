from .rag_metrics.retrieval.recall import Recall
from .rag_metrics.retrieval.eir import EIR
from .rag_metrics.retrieval.ndcg import NDCG
from .rag_metrics.retrieval.mrr import MRR

METRICS = {
    'recall': Recall,
    'eir': EIR,
    'ndcg': NDCG,
    'mrr': MRR
}

def get_metric(name):
    return METRICS.get(name.lower())
