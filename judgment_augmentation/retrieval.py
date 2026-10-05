"""BM25 first-stage retrieval: the cheap candidate generator that both the
baseline and augmented rankers rerank on top of. Rerankers in production
search systems sit on top of a retriever like this one, not on the whole
corpus; keeping that structure here mirrors the real deployment shape
instead of evaluating against the entire document collection.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, docs: dict[str, str]):
        self.doc_ids = list(docs.keys())
        tokenized = [docs[d].lower().split() for d in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: -x[1])
        return ranked[:k]
