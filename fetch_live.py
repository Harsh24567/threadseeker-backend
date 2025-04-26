import os
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import praw

model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_reddit_threads(topic: str) -> List[Dict]:
    try:
        reddit = praw.Reddit(
            client_id="4NM1Zo0T45RXrtoLTzYQUg",
            client_secret="CHXlPubty1EGgv2Grrva6XSEcpxsaQ",
            user_agent="ThreadSeekerAI/0.1 by Instinct23"
        )
        posts = reddit.subreddit("all").search(topic, limit=10)
        return [
            {
                "title": post.title,
                "content": post.selftext or post.url,
                "tags": ["reddit", topic],
                "source": "Reddit",
                "timestamp": f"{post.created_utc:.0f}"
            } for post in posts
        ]
    except:
        return []


def score_and_sort_threads(query: str, threads: list) -> list:
    if not threads:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    thread_texts = [t["title"] + " " + t["content"] for t in threads]
    thread_embeddings = model.encode(thread_texts, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, thread_embeddings)[0]

    for i, thread in enumerate(threads):
        thread["relevance_score"] = float(similarities[i])

    return sorted(threads, key=lambda x: x["relevance_score"], reverse=True)

def fetch_live_threads(topic):
    raw_threads = fetch_reddit_threads(topic)
    return score_and_sort_threads(topic, raw_threads)