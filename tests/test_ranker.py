from judgment_augmentation.ranker import RelevanceRanker, extract_features


def test_extract_features_shape():
    features = extract_features("turmeric arthritis", "turmeric helps arthritis pain", bm25_score=3.5)
    assert len(features) == 4
    assert features[0] == 3.5


def test_extract_features_overlap_is_fraction_of_query_terms_matched():
    features = extract_features("turmeric arthritis", "turmeric helps joint pain", bm25_score=1.0)
    overlap = features[1]
    assert overlap == 0.5  # "turmeric" matches, "arthritis" doesn't


def test_ranker_learns_to_separate_relevant_from_irrelevant():
    features = [
        [5.0, 1.0, 10, 2],  # high BM25, full overlap -> relevant
        [5.0, 1.0, 10, 2],
        [0.0, 0.0, 10, 2],  # no overlap -> irrelevant
        [0.0, 0.0, 10, 2],
    ]
    labels = [2, 2, 0, 0]
    ranker = RelevanceRanker()
    ranker.fit(features, labels)
    predictions = ranker.predict(features)
    assert predictions[0] > predictions[2]
