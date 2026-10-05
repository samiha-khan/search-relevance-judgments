"""The actual test of the paper's claim: does filling in missing relevance
judgments with an LLM improve ranking quality, and does the improvement
concentrate on tail queries (the ones with little human signal to begin
with), the way Apple's paper reports?

Two conditions, same ranker, same BM25 candidates, same evaluation:
  (a) human-only:    ranker trained on real human judgments alone
  (b) augmented:      ranker also trained on LLM-generated judgments for
                       (query, doc) pairs in the BM25 candidate pool that
                       no human ever rated

Evaluated on a held-out half of each query's human judgments (never used
for training in either condition), so both conditions are scored against
the same real ground truth.

Judging every unjudged BM25 candidate across all 323 queries would be
several thousand LLM calls (hours on a laptop GPU for no added rigor over
a smaller, fixed, reproducible sample). This runs on a fixed random sample
of head and tail queries instead (seeded, so the sample is reproducible),
with a smaller candidate pool per query -- a documented scope reduction,
not a hidden one.

Run from the repo root:
    python scripts/run_augmentation_experiment.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judgment_augmentation.data import load_nfcorpus, judged_doc_count, split_head_tail
from judgment_augmentation.evaluate import ndcg_at_k
from judgment_augmentation.llm_judge import LocalLLMJudge
from judgment_augmentation.ranker import RelevanceRanker, extract_features
from judgment_augmentation.retrieval import BM25Retriever

K_CANDIDATES = 15
MIN_JUDGMENTS_TO_EVAL = 2  # need at least 1 train + 1 held-out judgment
QUERIES_PER_GROUP = 50  # sampled from head and from tail, each


def build_training_rows(query_ids, corpus, retriever, llm_judge=None):
    """For each query: take its human judgments, hold back half for eval,
    train on the rest. If llm_judge is given, also label every BM25
    candidate that no human judged at all (neither train nor held-out
    half), and add those as extra training rows.

    The eval pool is the FULL top-K BM25 candidate list for the query
    (minus whatever went into training), with any candidate NFCorpus never
    judged treated as relevance 0 -- the standard IR convention, and the
    only way to pose a real ranking problem. Evaluating only on the
    held-out judged docs (tried first) gave tail queries an eval set of
    0-1 documents on average, making NDCG trivially close to 1 and
    silently uninformative."""
    train_rows = []  # (features, label)
    eval_rows = []    # (query_id, [(doc_id, bm25_score), ...])

    for qid in query_ids:
        human_judgments = corpus.qrels.get(qid, {})
        judged_doc_ids = list(human_judgments.keys())
        if len(judged_doc_ids) < MIN_JUDGMENTS_TO_EVAL:
            continue

        random.Random(hash(qid) % (2**31)).shuffle(judged_doc_ids)
        half = max(1, len(judged_doc_ids) // 2)
        train_docs, eval_docs = set(judged_doc_ids[:half]), set(judged_doc_ids[half:])
        if not eval_docs:
            continue

        candidates = retriever.search(corpus.queries[qid], k=K_CANDIDATES)
        candidate_scores = dict(candidates)

        for did in train_docs:
            bm25_score = candidate_scores.get(did, 0.0)
            features = extract_features(corpus.queries[qid], corpus.docs[did], bm25_score)
            train_rows.append((features, human_judgments[did]))

        if llm_judge is not None:
            unjudged_candidates = [did for did, _ in candidates if did not in human_judgments]
            for did in unjudged_candidates:
                llm_rel = llm_judge.judge(corpus.queries[qid], corpus.docs[did])
                features = extract_features(corpus.queries[qid], corpus.docs[did], candidate_scores[did])
                train_rows.append((features, llm_rel))

        # Eval pool = every BM25 candidate not used for training. Most
        # will be unjudged (truly or by construction) and count as
        # relevance 0; the held-out judged docs (if present in top-K) are
        # the real positives this is testing whether the ranker finds.
        eval_candidates = [(did, score) for did, score in candidates if did not in train_docs]
        if any(did in eval_docs for did, _ in eval_candidates):
            eval_rows.append((qid, eval_candidates))

    return train_rows, eval_rows


def evaluate_condition(ranker, eval_rows, corpus, retriever, k=10):
    ndcgs = []
    for qid, eval_candidates in eval_rows:
        features = [extract_features(corpus.queries[qid], corpus.docs[did], score)
                    for did, score in eval_candidates]
        predicted = ranker.predict(features)
        order = sorted(range(len(eval_candidates)), key=lambda i: -predicted[i])
        relevances = [corpus.qrels[qid].get(did, 0) for did, _ in eval_candidates]
        ranked_relevances = [relevances[i] for i in order]
        ndcgs.append(ndcg_at_k(ranked_relevances, relevances, k=k))
    return ndcgs


def main():
    print("Loading NFCorpus...")
    corpus = load_nfcorpus()
    retriever = BM25Retriever(corpus.docs)
    full_head_ids, full_tail_ids = split_head_tail(corpus)

    rng = random.Random(42)
    head_ids = rng.sample(full_head_ids, min(QUERIES_PER_GROUP, len(full_head_ids)))
    tail_ids = rng.sample(full_tail_ids, min(QUERIES_PER_GROUP, len(full_tail_ids)))
    all_ids = head_ids + tail_ids
    print(f"  Sampled {len(head_ids)} head queries and {len(tail_ids)} tail queries "
          f"(of {len(full_head_ids)} / {len(full_tail_ids)} total, seed=42)\n")

    print("=== Condition A: human judgments only ===")
    train_rows_a, eval_rows = build_training_rows(all_ids, corpus, retriever, llm_judge=None)
    ranker_a = RelevanceRanker()
    ranker_a.fit([f for f, _ in train_rows_a], [l for _, l in train_rows_a])
    print(f"  Trained on {len(train_rows_a)} human-judged rows")

    print("\nLoading local LLM judge for condition B (this is the slow part)...")
    judge = LocalLLMJudge()

    print("\n=== Condition B: human + LLM-generated judgments ===")
    train_rows_b, _ = build_training_rows(all_ids, corpus, retriever, llm_judge=judge)
    ranker_b = RelevanceRanker()
    ranker_b.fit([f for f, _ in train_rows_b], [l for _, l in train_rows_b])
    print(f"  Trained on {len(train_rows_b)} rows "
          f"({len(train_rows_a)} human + {len(train_rows_b) - len(train_rows_a)} LLM-generated)")

    eval_head = [(q, c) for q, c in eval_rows if q in head_ids]
    eval_tail = [(q, c) for q, c in eval_rows if q in tail_ids]

    def report(label, ranker):
        ndcg_all = evaluate_condition(ranker, eval_rows, corpus, retriever)
        ndcg_head = evaluate_condition(ranker, eval_head, corpus, retriever)
        ndcg_tail = evaluate_condition(ranker, eval_tail, corpus, retriever)
        print(f"\n{label}")
        print(f"  Overall NDCG@10: {sum(ndcg_all) / len(ndcg_all):.4f}  (n={len(ndcg_all)} queries)")
        print(f"  Head-query NDCG@10: {sum(ndcg_head) / len(ndcg_head):.4f}  (n={len(ndcg_head)})")
        print(f"  Tail-query NDCG@10: {sum(ndcg_tail) / len(ndcg_tail):.4f}  (n={len(ndcg_tail)})")
        return ndcg_all, ndcg_head, ndcg_tail

    results_a = report("Condition A (human only):", ranker_a)
    results_b = report("Condition B (human + LLM):", ranker_b)

    print("\n=== Summary ===")
    print(f"  Overall NDCG@10:    {sum(results_a[0])/len(results_a[0]):.4f} -> {sum(results_b[0])/len(results_b[0]):.4f}")
    print(f"  Head-query NDCG@10: {sum(results_a[1])/len(results_a[1]):.4f} -> {sum(results_b[1])/len(results_b[1]):.4f}")
    print(f"  Tail-query NDCG@10: {sum(results_a[2])/len(results_a[2]):.4f} -> {sum(results_b[2])/len(results_b[2]):.4f}")


if __name__ == "__main__":
    main()
