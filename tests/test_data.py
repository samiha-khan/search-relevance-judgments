from judgment_augmentation.data import Corpus, judged_doc_count, split_head_tail


def make_corpus() -> Corpus:
    queries = {"q1": "a", "q2": "b", "q3": "c", "q4": "d"}
    docs = {"d1": "x", "d2": "y"}
    qrels = {
        "q1": {"d1": 2},  # 1 judgment -> tail
        "q2": {"d1": 2, "d2": 1},  # 2 judgments
        "q3": {"d1": 2, "d2": 2},  # 2 judgments
        "q4": {"d1": 2, "d2": 1},  # fewer judgments overall should land in tail
    }
    return Corpus(queries=queries, docs=docs, qrels=qrels)


def test_judged_doc_count():
    corpus = make_corpus()
    assert judged_doc_count(corpus, "q1") == 1
    assert judged_doc_count(corpus, "q2") == 2


def test_judged_doc_count_for_unjudged_query_is_zero():
    corpus = make_corpus()
    assert judged_doc_count(corpus, "nonexistent") == 0


def test_split_head_tail_puts_fewest_judgments_in_tail():
    corpus = make_corpus()
    head, tail = split_head_tail(corpus, tail_fraction=0.5)
    assert "q1" in tail
    assert len(head) + len(tail) == 4


def test_split_head_tail_fractions_are_respected():
    corpus = make_corpus()
    head, tail = split_head_tail(corpus, tail_fraction=0.25)
    assert len(tail) == 1
    assert len(head) == 3
