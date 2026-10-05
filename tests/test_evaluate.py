from judgment_augmentation.evaluate import cohen_kappa, exact_agreement_rate, ndcg_at_k


def test_ndcg_perfect_ranking_is_one():
    relevances = [2, 1, 0]
    assert ndcg_at_k(relevances, relevances, k=3) == 1.0


def test_ndcg_worst_ranking_is_below_perfect():
    ideal = [2, 1, 0]
    worst = [0, 1, 2]
    assert ndcg_at_k(worst, ideal, k=3) < ndcg_at_k(ideal, ideal, k=3)


def test_ndcg_with_no_relevant_docs_is_zero():
    assert ndcg_at_k([0, 0, 0], [0, 0, 0], k=3) == 0.0


def test_ndcg_only_considers_top_k():
    ideal = [2, 1, 1, 1]
    ranked_but_buried = [1, 1, 1, 2]  # the relevant doc is outside top-1
    assert ndcg_at_k(ranked_but_buried, ideal, k=1) < ndcg_at_k(ideal, ideal, k=1)


def test_cohen_kappa_perfect_agreement():
    human = [0, 1, 2, 1, 0, 2]
    assert cohen_kappa(human, human) == 1.0


def test_cohen_kappa_below_one_when_judges_disagree():
    human = [0, 1, 2, 1, 0, 2]
    llm = [0, 1, 2, 0, 0, 1]
    assert 0.0 < cohen_kappa(human, llm) < 1.0


def test_cohen_kappa_near_zero_for_random_looking_disagreement():
    human = [0, 0, 0, 2, 2, 2]
    llm = [2, 2, 2, 0, 0, 0]  # perfectly inverted
    assert cohen_kappa(human, llm) < 0


def test_exact_agreement_rate():
    human = [0, 1, 2, 1]
    llm = [0, 1, 1, 1]
    assert exact_agreement_rate(human, llm) == 0.75
