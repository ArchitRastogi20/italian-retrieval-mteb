from typing import List, Union
import re
try:
    from pysbd import Segmenter
    segmenter_it = Segmenter(language="it", clean=True)
except:
    segmenter_it = None

def split_sentences(text: str, language: str) -> List[str]:
    if not isinstance(text, str):
        if isinstance(text, dict) and "content" in text:
            text = text["content"]
        else:
            text = str(text)
    
    if language == 'it' and segmenter_it:
        return [s.strip() for s in segmenter_it.segment(text) if s.strip()]
    else:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def _normalize_space(s: str) -> str:
    return " ".join(s.split())

def _flatten_query(q: Union[List[str], List[dict], str]) -> str:
    if isinstance(q, list):
        parts = []
        for x in q:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and isinstance(x.get("content"), str):
                parts.append(x["content"])
        return " ".join(parts)
    return q

def exist_match(query_text, reference_texts, language="it", case_sensitive=False, use_sentence_split=True):
    q_str = _flatten_query(query_text)
    if not isinstance(q_str, str):
        return 0
    q_str = _normalize_space(q_str)
    ref_str = _normalize_space(" ".join(r for r in reference_texts if isinstance(r, str)))
    
    if not case_sensitive:
        q_str = q_str.lower()
        ref_str = ref_str.lower()
    
    if use_sentence_split:
        try:
            sentences = split_sentences(q_str, language)
        except:
            sentences = [q_str]
    else:
        sentences = [q_str]
    
    for s in sentences:
        s_cmp = _normalize_space(s)
        if s_cmp and s_cmp not in ref_str:
            return 0
    return 1
