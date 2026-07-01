import os
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

def compute_similarity_scores(job_desc: str, candidates: List[Dict[str, Any]]) -> List[float]:
    """
    Computes vector similarity scores using scikit-learn TF-IDF Vectorizer.
    If TF-IDF fitting fails, falls back to Jaccard overlap.
    """
    if not candidates:
        return []

    # Construct representative candidate text representation
    candidate_texts = []
    for c in candidates:
        profile = c.get("profile") or {}
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        career_history = c.get("career_history") or []
        if not isinstance(career_history, list):
            career_history = []
            
        history_desc_parts = []
        for h in career_history:
            if isinstance(h, dict) and h.get("description"):
                history_desc_parts.append(h.get("description"))
        history_desc = " ".join(history_desc_parts)
        candidate_texts.append(f"{headline} {summary} {history_desc}".strip())

    try:
        import re
        def normalize_text(t):
            t = re.sub(r'\b(faiss|qdrant|weaviate|pinecone|milvus|opensearch|elasticsearch)\b', '__VECTOR_DB__', t, flags=re.IGNORECASE)
            t = re.sub(r'\b(ndcg|mrr|map)\b', '__RANKING_METRIC__', t, flags=re.IGNORECASE)
            return t
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        all_texts = [normalize_text(job_desc)] + [normalize_text(t) for t in candidate_texts]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Cosine similarity between job description and candidate profiles
        jd_vector = tfidf_matrix[0]
        cand_vectors = tfidf_matrix[1:]
        
        # Multiply cand_vectors by jd_vector transpose to get cosine similarity
        similarities = (cand_vectors * jd_vector.T).toarray().flatten()
        return [max(0.0, min(1.0, float(score))) for score in similarities]
    except Exception:
        pass

    # Jaccard overlap fallback
    scores = []
    jd_words = set(re_tokenize(job_desc.lower()))
    for text in candidate_texts:
        text_words = set(re_tokenize(text.lower()))
        if not jd_words or not text_words:
            scores.append(0.0)
            continue
        intersection = jd_words.intersection(text_words)
        union = jd_words.union(text_words)
        scores.append(float(len(intersection) / len(union)))
    return scores

def re_tokenize(text: str) -> List[str]:
    import re
    return re.findall(r'\w+', text)
