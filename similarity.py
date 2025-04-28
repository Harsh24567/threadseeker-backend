import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def load_threads(file_path: str = None) -> List[Dict]:
    """
    Loads the local thread_data.json dataset
    """
    if file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "thread_data.json")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_similar_threads(query: str, top_k: int = 10) -> List[Dict]:
    """
    Perform semantic search over thread_data.json using SBERT
    Returns top_k most similar threads with relevance_score
    """
    threads = load_threads()
    if not threads:
        return []

    corpus = [t["title"] + " " + t.get("content", "") for t in threads]

    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    for i, thread in enumerate(threads):
        thread["relevance_score"] = float(similarities[i])

    sorted_threads = sorted(threads, key=lambda x: x["relevance_score"], reverse=True)
    return sorted_threads[:top_k]
