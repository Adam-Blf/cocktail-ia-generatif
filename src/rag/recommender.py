"""
Moteur de recommandation de cocktails par similarite semantique.
Supporte la recommandation par description libre et par ingredients disponibles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..nlp.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Resultat d'une recommandation : cocktail + scores."""
    name: str
    category: str
    ingredients: str
    instructions: str
    similarity_score: float
    flavor_profile: dict[str, float] = field(default_factory=dict)
    rank: int = 0


class CocktailRecommender:
    """
    Moteur de recommandation base sur les embeddings SBERT et la similarite cosinus.
    Supporte deux modes :
    - Mode texte libre : requete semantique (ex. "frais et agrumes")
    - Mode ingredients : liste d'ingredients disponibles
    """

    FLAVOR_COLS = ["sweet", "sour", "bitter", "strong", "fresh", "fruity"]

    def __init__(self, engine: Optional[EmbeddingEngine] = None):
        self.engine = engine or EmbeddingEngine()
        self._df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, text_col: str = "text_full") -> "CocktailRecommender":
        """
        Indexe le corpus cocktails.

        Args:
            df: DataFrame avec les cocktails (doit contenir text_col).
            text_col: Colonne de texte a encoder.

        Returns:
            self (pour le chaining).
        """
        if text_col not in df.columns:
            raise ValueError(f"Colonne '{text_col}' absente du DataFrame.")

        self._df = df.reset_index(drop=True)
        texts = df[text_col].fillna("").tolist()
        self._embeddings = self.engine.encode(texts, show_progress=True)
        logger.info("Index construit : %d cocktails, dim=%d", len(texts), self._embeddings.shape[1])
        return self

    def recommend_by_query(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[RecommendationResult]:
        """
        Recommande des cocktails a partir d'une description en langage naturel.

        Args:
            query: Description utilisateur (ex. "quelque chose de frais et fruité").
            top_k: Nombre de resultats.
            category_filter: Filtrer par categorie (optionnel).

        Returns:
            Liste de RecommendationResult triee par score decroissant.
        """
        self._check_fitted()
        query_vec = self.engine.encode_single(query)
        scores = self._embeddings @ query_vec

        df = self._df.copy()
        if category_filter:
            mask = df["category"].str.lower() == category_filter.lower()
            scores = np.where(mask, scores, -1.0)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._build_results(top_indices, scores)

    def recommend_by_ingredients(
        self,
        ingredients: list[str],
        top_k: int = 5,
        must_include: Optional[list[str]] = None,
    ) -> list[RecommendationResult]:
        """
        Recommande des cocktails a partir d'une liste d'ingredients disponibles.
        Pondere par le nombre d'ingredients communs.

        Args:
            ingredients: Liste des ingredients disponibles.
            top_k: Nombre de resultats.
            must_include: Ingredients qui doivent tous etre presents.

        Returns:
            Liste de RecommendationResult.
        """
        self._check_fitted()

        query_text = "cocktail avec " + ", ".join(ingredients)
        query_vec = self.engine.encode_single(query_text)
        semantic_scores = self._embeddings @ query_vec

        # Score de couverture : proportion d'ingredients disponibles trouves dans la recette
        ing_lower = [i.lower() for i in ingredients]
        coverage_scores = np.array([
            self._ingredient_coverage(row["ingredients"], ing_lower)
            for _, row in self._df.iterrows()
        ])

        combined_scores = 0.5 * semantic_scores + 0.5 * coverage_scores

        # Filtre must_include
        if must_include:
            must_lower = [m.lower() for m in must_include]
            for idx, row in self._df.iterrows():
                if not all(m in row["ingredients"].lower() for m in must_lower):
                    combined_scores[idx] = -1.0

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return self._build_results(top_indices, combined_scores)

    def _ingredient_coverage(self, recipe_ingredients: str, available: list[str]) -> float:
        """Taux d'ingredients disponibles couverts par la recette."""
        recipe_lower = str(recipe_ingredients).lower()
        matches = sum(1 for ing in available if ing in recipe_lower)
        return matches / max(len(available), 1)

    def _build_results(self, indices: np.ndarray, scores: np.ndarray) -> list[RecommendationResult]:
        """Construit la liste de resultats depuis les indices."""
        results = []
        flavor_cols = [c for c in self.FLAVOR_COLS if c in self._df.columns]

        for rank, idx in enumerate(indices):
            row = self._df.iloc[idx]
            profile = {c: float(row[c]) for c in flavor_cols}
            results.append(RecommendationResult(
                name=str(row.get("name", "")),
                category=str(row.get("category", "")),
                ingredients=str(row.get("ingredients", "")),
                instructions=str(row.get("instructions", "")),
                similarity_score=float(scores[idx]),
                flavor_profile=profile,
                rank=rank + 1,
            ))
        return results

    def _check_fitted(self) -> None:
        """Verifie que le moteur a ete indexe."""
        if self._df is None or self._embeddings is None:
            raise RuntimeError("Appeler .fit(df) avant de recommander.")


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Calcule Precision@K."""
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & relevant) / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Calcule Recall@K."""
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & relevant) / max(len(relevant), 1)


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Calcule NDCG@K."""
    retrieved_k = retrieved[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, item in enumerate(retrieved_k) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
