"""Step 1 of the real experiment: before trusting the LLM's relevance
judgments for anything, check them against real human judgments on pairs
both have labeled. If the LLM doesn't agree with humans here, there's no
point using it to fill in gaps where humans said nothing at all.

Run from the repo root:
    python scripts/run_judge_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judgment_augmentation.data import load_nfcorpus
from judgment_augmentation.evaluate import cohen_kappa, exact_agreement_rate
from judgment_augmentation.llm_judge import LocalLLMJudge


def main():
    print("Loading NFCorpus (real human-judged medical search queries)...")
    corpus = load_nfcorpus()
    judged_pairs = [
        (qid, did, rel)
        for qid, docs in corpus.qrels.items()
        for did, rel in docs.items()
    ]
    print(f"  {len(corpus.queries)} queries, {len(corpus.docs)} documents, "
          f"{len(judged_pairs)} human-judged (query, doc) pairs\n")

    print("Loading local LLM judge (Qwen2.5-1.5B-Instruct)...")
    judge = LocalLLMJudge()

    sample_size = min(300, len(judged_pairs))
    import random
    random.seed(0)
    sample = random.sample(judged_pairs, sample_size)

    print(f"Scoring {sample_size} pairs with the LLM judge (this is the slow part)...")
    human_labels, llm_labels = [], []
    for i, (qid, did, human_rel) in enumerate(sample):
        llm_rel = judge.judge(corpus.queries[qid], corpus.docs[did])
        human_labels.append(human_rel)
        llm_labels.append(llm_rel)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{sample_size}")

    agreement = exact_agreement_rate(human_labels, llm_labels)
    kappa = cohen_kappa(human_labels, llm_labels)

    print(f"\n=== LLM judge vs. real human judgments ({sample_size} pairs) ===")
    print(f"  Exact agreement: {agreement:.1%}")
    print(f"  Cohen's kappa:   {kappa:.3f}")
    print("\nRule of thumb for kappa: <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, "
          "0.6-0.8 substantial, >0.8 near-perfect agreement.")

    with open("judge_validation_results.txt", "w") as f:
        f.write(f"sample_size={sample_size}\nagreement={agreement}\nkappa={kappa}\n")
        for (qid, did, h), l in zip(sample, llm_labels):
            f.write(f"{qid}\t{did}\t{h}\t{l}\n")
    print("\nSaved raw labels to judge_validation_results.txt")


if __name__ == "__main__":
    main()
