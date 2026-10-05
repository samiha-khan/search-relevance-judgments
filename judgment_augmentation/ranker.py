"""A small supervised reranker: takes (query, document, BM25 score) triples
labeled with a relevance grade and learns to predict relevance from cheap
lexical features. This is the thing whose training data varies between the
two experimental conditions (human judgments only vs. human + LLM-generated
judgments) -- the ranker itself is identical in both.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


def extract_features(query: str, document: str, bm25_score: float) -> list[float]:
    q_terms = set(query.lower().split())
    d_terms = document.lower().split()
    d_term_set = set(d_terms)
    overlap = len(q_terms & d_term_set) / max(len(q_terms), 1)
    return [
        bm25_score,
        overlap,
        len(d_terms),
        len(q_terms),
    ]


class RelevanceRanker:
    def __init__(self):
        # min_child_samples defaults to 1: with only a few labeled examples
        # per query (the whole point of the tail-query regime this project
        # studies), a higher default silently refuses to split and predicts
        # the dataset mean for every input -- a flat, useless ranker that
        # looks like it trained successfully.
        #
        # n_jobs=1: LightGBM's default multi-threaded OpenMP pool segfaults
        # in this process once torch/transformers has already loaded its
        # own OpenMP runtime (a known conflict between the two libraries'
        # bundled OpenMP on macOS). The dataset here is small enough that
        # single-threaded training costs nothing.
        self.model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=1,
            verbosity=-1, n_jobs=1,
        )

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        self.model.fit(np.array(features), np.array(labels))

    def predict(self, features: list[list[float]]) -> np.ndarray:
        return self.model.predict(np.array(features))
