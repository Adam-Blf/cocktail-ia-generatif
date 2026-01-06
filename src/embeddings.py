"""
Module d'embeddings semantiques via SBERT.
Gere le cache disque pour eviter les recomputations couteuses.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Wrapper SBERT avec cache MD5 sur disque.
    Evite de recharger le modele et de recalculer les embeddings a chaque run.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache: bool = True):
        self.model_name = model_name
        self.use_cache = cache
        self._model = None

    @property
    def model(self):
        """Lazy load du modele SBERT."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Modele SBERT charge : %s", self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers requis : pip install sentence-transformers")
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs.
        Utilise le cache si disponible.

        Args:
            texts: Liste de textes a encoder.
            batch_size: Taille du batch pour l'inference.
            show_progress: Afficher la barre de progression.

        Returns:
            Matrice d'embeddings (n_texts, embedding_dim).
        """
        cache_key = self._compute_cache_key(texts)
        cache_path = CACHE_DIR / f"{cache_key}.pkl"

        if self.use_cache and cache_path.exists():
            logger.debug("Cache hit : %s", cache_key)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        logger.info("Calcul des embeddings pour %d textes...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalise pour la similarite cosinus via produit scalaire
        )

        if self.use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.debug("Embeddings sauvegardes : %s", cache_path)

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode un seul texte. Pas de cache (requete temps reel)."""
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Similarite cosinus entre deux vecteurs normalises (produit scalaire)."""
        return float(np.dot(vec_a, vec_b))

    def similarity_matrix(self, embeddings_a: np.ndarray, embeddings_b: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Matrice de similarite cosinus.
        Si embeddings_b est None, calcule la matrice auto (corpus x corpus).
        """
        b = embeddings_a if embeddings_b is None else embeddings_b
        return np.dot(embeddings_a, b.T)

    def _compute_cache_key(self, texts: list[str]) -> str:
        """Cle MD5 basee sur le contenu des textes et le nom du modele."""
        content = self.model_name + "||".join(texts)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @property
    def embedding_dim(self) -> int:
        """Dimension des vecteurs (384 pour all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()
