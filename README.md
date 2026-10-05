# search-relevance-judgments

**The question this answers:** a search ranker learns what's relevant from
judgments, usually clicks or human ratings. Popular queries get plenty of
signal. Rare queries get almost none, so the ranker is guessing blind for
exactly the users it serves worst. Can an LLM, asked to judge "is this
document relevant to this query," fill in the missing judgments well enough
to actually improve ranking where it's weakest?

This is a small-scale replication of
[Apple's "Scaling Search Relevance: Augmenting App Store Ranking with
LLM-Generated Judgments"](https://arxiv.org/abs/2602.23234) (Christakopoulou,
Patel, Velaga, Gaikwad, Suchter and Sundaranatha, submitted Feb 2026). Their
paper fine-tunes a 3B-parameter LLM on millions of real App Store relevance
labels, generates millions more, and validates the result with a live
worldwide A/B test (+0.24% conversion, concentrated on tail queries). None of
that scale is reproducible without Apple's data or budget, and this project
doesn't claim to. What's reproducible is the core method and the core
question, on a public benchmark with a free local model instead of a
fine-tuned proprietary one.

Checked directly (arXiv page and GitHub search by exact title) before
starting: this paper has no public code release.

## What it actually does

Three pieces:

- **A real human-judged search dataset.** [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
  (Boteva et al., 2016), via the [BEIR](https://github.com/beir-cellar/beir)
  benchmark: 323 real medical/nutrition queries, 3,633 real documents, 12,334
  graded human relevance judgments (0/1/2). Judgments are naturally uneven
  across queries (some queries have dozens, most have a handful), which
  stands in for the paper's head/tail query split without synthetically
  deleting anything.
- **A local LLM playing the judge role.** `judgment_augmentation/llm_judge.py`
  prompts [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  (run locally, not a paid API) to grade (query, document) pairs on the same
  0/1/2 scale NFCorpus's human annotators used.
- **A judgment-quality check before trusting any of it.**
  `scripts/run_judge_validation.py` compares the LLM's judgments against real
  held-out human judgments (exact agreement, Cohen's kappa) on pairs both
  have labeled. This is the step a naive replication would skip: generating
  synthetic labels and immediately using them without checking whether
  they're any good.

## The experiment

`scripts/run_augmentation_experiment.py` trains the same small reranker two
ways:

- **Condition A:** trained on real human judgments only.
- **Condition B:** trained on real human judgments *plus* LLM-generated
  judgments for every BM25 candidate document that no human ever rated.

Both are evaluated against the same held-out human judgments, split by
whether the query is "head" (many existing human judgments) or "tail" (few).
The paper's actual claim isn't "LLM judgments help on average," it's that
they help most specifically where behavioral signal is thin, so that's the
split this checks, not just an overall number.

## What it actually found

**The judge validation came back weak: Cohen's kappa 0.037 (chance-level),
39.3% exact agreement, on 300 real human-judged pairs.** The confusion
matrix shows a specific, not generic, failure: on the 15 pairs humans
graded "directly answers the query" (2), the LLM got 12 right -- it's
genuinely decent at recognizing strong relevance. On the 285 pairs graded
"on-topic but doesn't directly answer" (1), it correctly matched only 106;
it over-graded 155 of them (54%) all the way up to "directly answers."
The model can tell relevant from irrelevant; it can't reliably tell
*how* relevant.

Getting to that number took two real fixes, caught by checking results
that didn't make sense rather than trusting them:

1. The first prompt asked for a bare digit in 5 generated tokens. It
   answered "0" (not relevant) on 6 of 8 documents humans had graded
   "highly relevant." Giving the model one worked example and room for
   one sentence of reasoning before answering fixed most of that.
2. The first truncation cut documents to 800 characters. NFCorpus's
   median document is 1,615 characters, so over 90% of documents were
   losing their second half, often exactly the part that answers the
   query. Raising the limit to 2,000 characters nearly doubled kappa
   on a held-out check (0.035 to 0.16) before the full 300-pair run
   above, which reflects the real, harder, class-imbalanced picture.

**The ranking experiment: LLM augmentation made things slightly worse, not
better, on every slice.** (50 sampled head queries, 50 sampled tail
queries, seed 42; only queries whose BM25 top-15 contained at least one
held-out human-judged document are counted, which was 38/50 head and
17/50 tail -- tail queries miss retrieval entirely much more often, a
separate and real finding of its own.)

```
                      Condition A        Condition B
                      (human only)       (human + LLM)
Overall NDCG@10       0.4967 (n=55)      0.4684
Head-query NDCG@10    0.5254 (n=38)      0.4911
Tail-query NDCG@10    0.4327 (n=17)      0.4178
```

This directly contradicts the paper's finding that LLM-generated judgments
help, especially on tail queries. It doesn't contradict the judge
validation, though -- it confirms it. A judge with near-chance agreement
with humans, especially one that systematically confuses "related" with
"directly relevant," adds label noise when used as training signal, not
missing signal. The paper's result depends on judge quality: Apple's
judge was a 3B model fine-tuned on millions of real labels; this one is a
1.5B model zero-shot prompted on none. With 17-38 queries per tail/head
slice, this isn't a large enough sample to call the *size* of the gap
precise, but the direction is consistent across all three rows, across
two separate query populations, and matches exactly what the kappa number
predicted beforehand. That consistency is the actual finding, not the
specific NDCG decimals.

An earlier version of this evaluation held out only each query's *judged*
documents as the eval set, which gave tail queries an average eval pool
of about 1 document (NFCorpus's tail queries have as few as 2 total
judgments) and head queries about 37. Ranking essentially one known
document against itself pushed NDCG to 0.96-0.99 almost by construction
regardless of which condition was used -- a measurement artifact, not a
result. Rebuilding the eval pool from the full BM25 candidate list (with
genuinely unjudged candidates scored 0, standard IR convention) fixed
that and produced the real numbers above.

## What this isn't

- **Not a reproduction of Apple's production system.** No 3B-parameter
  fine-tuned model, no millions of real labels, no live A/B test. A
  1.5B-parameter instruct model, zero-shot prompted, is the honest
  substitute for what's buildable without their data or budget.
- **Not a claim that this exact pipeline is production-ready.** The ranker
  is a small LightGBM model over lexical features (BM25 score, term
  overlap, lengths), not a learned embedding-based reranker. The point is
  testing the judgment-augmentation idea cleanly, not building a
  state-of-the-art ranker.
- **Not evidence the LLM's judgments are good in general, or that the
  paper's method is wrong.** The validation measures agreement on one
  dataset, one domain (medical nutrition queries), one 1.5B model,
  zero-shot. It found that specific judge is not good enough to help this
  specific ranker. It says nothing about whether a better-matched model
  (larger, fine-tuned, or just better-prompted) would clear the bar Apple's
  3B fine-tuned model did. The honest conclusion of this project is
  narrower than "augmentation doesn't work" -- it's "augmentation's value
  depends entirely on judge quality, and this free substitute didn't have
  enough of it," which is itself the thing a naive replication (one that
  skipped the validation step) would never have caught.

## Quickstart

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Correctness tests on synthetic data (no download needed):
PYTHONPATH=. pytest tests/ -v

# Real experiment (downloads NFCorpus + a ~3GB local model, runs on CPU/MPS):
python scripts/run_judge_validation.py
python scripts/run_augmentation_experiment.py
```

## Project layout

```
judgment_augmentation/
  data.py        NFCorpus loading, head/tail query split by judgment count
  retrieval.py   BM25 first-stage candidate retrieval
  llm_judge.py   local LLM relevance judging (Qwen2.5-1.5B-Instruct)
  ranker.py      small LightGBM reranker over lexical features
  evaluate.py    NDCG@k, Cohen's kappa, exact agreement rate
tests/           18 correctness tests on synthetic data
scripts/
  run_judge_validation.py        LLM judgments vs. real human judgments
  run_augmentation_experiment.py the actual head/tail NDCG comparison
```

## Status

Core pipeline implemented and tested on synthetic data (18 tests). Full
real-data run complete on NFCorpus: judge validation (300 pairs) and the
head/tail augmentation experiment (100 sampled queries), both reported
under "What it actually found" above.

Debugged along the way, each caught by a result that didn't make sense
rather than assumed correct: a judge prompt that defaulted to "not
relevant" under a 5-token budget, 800-character document truncation that
cut off most answers, a segfault from torch and LightGBM both loading
their own OpenMP runtime in the same process, and an evaluation design
whose eval pool was accidentally too small to measure anything on tail
queries.
