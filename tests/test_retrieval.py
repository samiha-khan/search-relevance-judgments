from judgment_augmentation.retrieval import BM25Retriever


DOCS = {
    "d1": "turmeric reduces inflammation and joint pain in arthritis patients",
    "d2": "the stock market closed higher today on tech earnings",
    "d3": "arthritis patients may benefit from anti-inflammatory turmeric supplements",
}


def test_relevant_docs_rank_above_irrelevant_doc():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("turmeric arthritis", k=3)
    ranked_ids = [doc_id for doc_id, _ in results]
    assert ranked_ids.index("d2") > ranked_ids.index("d1")
    assert ranked_ids.index("d2") > ranked_ids.index("d3")


def test_search_respects_k():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("turmeric", k=1)
    assert len(results) == 1


def test_scores_are_sorted_descending():
    retriever = BM25Retriever(DOCS)
    results = retriever.search("turmeric arthritis", k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
