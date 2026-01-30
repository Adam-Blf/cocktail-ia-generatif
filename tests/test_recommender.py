"""Tests unitaires pour le moteur de recommandation."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


def make_fake_recommender(n_cocktails=50):
    """Cree un recommender avec des donnees et embeddings synthetiques."""
    from src.recommender import CocktailRecommender
    from src.data_loader import _generate_synthetic_data

    df = _generate_synthetic_data(n=n_cocktails)
    engine = MagicMock()
    engine.encode.return_value = np.random.rand(n_cocktails, 384)
    engine.encode.return_value /= np.linalg.norm(
        engine.encode.return_value, axis=1, keepdims=True
    )
    engine.encode_single.return_value = np.random.rand(384)
    engine.encode_single.return_value /= np.linalg.norm(engine.encode_single.return_value)

    rec = CocktailRecommender(engine=engine)
    rec.fit(df)
    return rec, df


def test_fit_creates_embeddings():
    rec, df = make_fake_recommender()
    assert rec._embeddings is not None
    assert rec._embeddings.shape[0] == len(df)


def test_recommend_by_query_returns_top_k():
    rec, _ = make_fake_recommender()
    results = rec.recommend_by_query("fresh and fruity", top_k=5)
    assert len(results) == 5


def test_recommend_by_query_sorted_by_score():
    rec, _ = make_fake_recommender()
    results = rec.recommend_by_query("citrus cocktail", top_k=5)
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_result_fields():
    rec, _ = make_fake_recommender()
    results = rec.recommend_by_query("sweet tropical", top_k=1)
    r = results[0]
    assert hasattr(r, 'name')
    assert hasattr(r, 'similarity_score')
    assert hasattr(r, 'ingredients')
    assert hasattr(r, 'flavor_profile')
    assert r.rank == 1


def test_recommend_by_ingredients():
    rec, _ = make_fake_recommender()
    results = rec.recommend_by_ingredients(['vodka', 'lime'], top_k=5)
    assert len(results) <= 5
    assert all(hasattr(r, 'name') for r in results)


def test_recommend_raises_when_not_fitted():
    from src.recommender import CocktailRecommender
    rec = CocktailRecommender()
    with pytest.raises(RuntimeError):
        rec.recommend_by_query("test")


def test_precision_at_k():
    from src.recommender import precision_at_k
    relevant = {'A', 'B', 'C'}
    retrieved = ['A', 'D', 'B', 'E', 'C']
    p = precision_at_k(relevant, retrieved, 5)
    assert abs(p - 3/5) < 1e-6


def test_ndcg_at_k():
    from src.recommender import ndcg_at_k
    relevant = {'A'}
    retrieved = ['A', 'B', 'C']
    ndcg = ndcg_at_k(relevant, retrieved, 3)
    assert 0 <= ndcg <= 1
