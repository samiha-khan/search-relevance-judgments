"""NDCG for ranking quality, and agreement metrics for judging whether the
LLM-generated relevance judgments are trustworthy against real human ones.
"""
from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_relevances: list[int], ideal_relevances: list[int], k: int = 10) -> float:
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


def cohen_kappa(human: list[int], llm: list[int]) -> float:
    """Cohen's kappa for ordinal agreement between two labelers over the
    same items, treating each distinct relevance grade as its own class."""
    assert len(human) == len(llm)
    n = len(human)
    labels = sorted(set(human) | set(llm))
    label_idx = {l: i for i, l in enumerate(labels)}

    confusion = [[0] * len(labels) for _ in labels]
    for h, m in zip(human, llm):
        confusion[label_idx[h]][label_idx[m]] += 1

    observed_agreement = sum(confusion[i][i] for i in range(len(labels))) / n

    row_marginals = [sum(row) / n for row in confusion]
    col_marginals = [sum(confusion[i][j] for i in range(len(labels))) / n for j in range(len(labels))]
    expected_agreement = sum(r * c for r, c in zip(row_marginals, col_marginals))

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def exact_agreement_rate(human: list[int], llm: list[int]) -> float:
    assert len(human) == len(llm)
    matches = sum(1 for h, m in zip(human, llm) if h == m)
    return matches / len(human)
