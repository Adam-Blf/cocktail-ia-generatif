"""Tests unitaires pour le module embeddings."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_similarity_between_identical_vectors():
    """Deux vecteurs identiques normalises doivent avoir une similarite de 1."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    vec = np.array([0.6, 0.8, 0.0])  # norme = 1.0
    assert abs(engine.similarity(vec, vec) - 1.0) < 1e-6


def test_similarity_between_orthogonal_vectors():
    """Deux vecteurs orthogonaux doivent avoir une similarite de 0."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    assert abs(engine.similarity(vec_a, vec_b)) < 1e-6


def test_similarity_matrix_shape():
    """La matrice de similarite doit avoir les bonnes dimensions."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    mat = np.random.rand(10, 384)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sim_matrix = engine.similarity_matrix(mat)
    assert sim_matrix.shape == (10, 10)


def test_cache_key_consistency():
    """La meme liste de textes doit toujours produire la meme cle de cache."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.model_name = "all-MiniLM-L6-v2"
    texts = ["mojito", "negroni", "margarita"]
    key1 = engine._compute_cache_key(texts)
    key2 = engine._compute_cache_key(texts)
    assert key1 == key2


def test_cache_key_differs_for_different_texts():
    """Deux listes differentes doivent produire des cles distinctes."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine.model_name = "all-MiniLM-L6-v2"
    key1 = engine._compute_cache_key(["mojito"])
    key2 = engine._compute_cache_key(["negroni"])
    assert key1 != key2
