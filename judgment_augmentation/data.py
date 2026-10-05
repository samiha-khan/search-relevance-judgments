"""Loads NFCorpus (Boteva et al., 2016), a real medical/nutrition IR
benchmark with graded human relevance judgments (0/1/2), via ir_datasets.

NFCorpus's judgments are naturally sparse and uneven across queries: some
queries have many judged documents, most have only a handful. That unevenness
stands in for the paper's "tail queries lack behavioral signal" problem
without needing to synthetically delete anything.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import ir_datasets


@dataclass
class Corpus:
    queries: dict[str, str]          # query_id -> text
    docs: dict[str, str]             # doc_id -> text (title + body)
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}


def load_nfcorpus(split: str = "test") -> Corpus:
    docs_ds = ir_datasets.load("beir/nfcorpus")
    split_ds = ir_datasets.load(f"beir/nfcorpus/{split}")

    docs = {d.doc_id: f"{d.title} {d.text}".strip() for d in docs_ds.docs_iter()}
    queries = {q.query_id: q.text for q in split_ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in split_ds.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    return Corpus(queries=queries, docs=docs, qrels=dict(qrels))


def judged_doc_count(corpus: Corpus, query_id: str) -> int:
    return len(corpus.qrels.get(query_id, {}))


def split_head_tail(corpus: Corpus, tail_fraction: float = 0.5) -> tuple[list[str], list[str]]:
    """Splits query ids into head/tail halves by how many human judgments
    each query already has. Tail = fewer judgments = the sparse-signal
    regime the paper targets."""
    judged_queries = [q for q in corpus.queries if q in corpus.qrels and corpus.qrels[q]]
    ranked = sorted(judged_queries, key=lambda q: judged_doc_count(corpus, q))
    cut = int(len(ranked) * tail_fraction)
    tail, head = ranked[:cut], ranked[cut:]
    return head, tail
