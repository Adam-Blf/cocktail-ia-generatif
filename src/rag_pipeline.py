"""
Pipeline RAG (Retrieval-Augmented Generation) pour la creation de cocktails.
Combine un index FAISS pour le retrieval et un modele generatif pour la synthese.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .embeddings import EmbeddingEngine
from .recommender import CocktailRecommender
from .translator import QueryTranslator

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent.parent / ".cache" / "rag_cache.json"
GUARDRAIL_THRESHOLD = 0.40


class RAGPipeline:
    """
    Pipeline RAG pour la creation et recommandation de cocktails.

    Etapes :
    1. Traduction EN : normalise la requete vers l'anglais (SBERT est anglophone).
    2. Guardrail semantique : verifier que la requete est dans le domaine cocktails.
    3. Retrieval FAISS : recuperer les K cocktails les plus similaires.
    4. Construction du contexte : injecter les recettes retrievees dans le prompt.
    5. Generation : appeler le modele generatif avec le contexte.
    6. Cache MD5 : eviter les appels redondants.
    """

    def __init__(
        self,
        recommender: CocktailRecommender,
        engine: Optional[EmbeddingEngine] = None,
        guardrail_threshold: float = GUARDRAIL_THRESHOLD,
        translate_queries: bool = True,
    ):
        self.recommender = recommender
        self.engine = engine or recommender.engine
        self.guardrail_threshold = guardrail_threshold
        # Traducteur EN : ameliore le recall SBERT sur les requetes non-anglaises
        self.translator = QueryTranslator(enabled=translate_queries)
        self._cache: dict[str, str] = self._load_cache()

    def query(
        self,
        user_query: str,
        top_k: int = 3,
        temperature: float = 0.7,
        generate: bool = True,
    ) -> dict:
        """
        Traite une requete utilisateur de bout en bout.

        Args:
            user_query: Requete en langage naturel (FR, EN, ES, etc.).
            top_k: Nombre de cocktails a recuperer pour le contexte.
            temperature: Temperature de generation (creativite).
            generate: Si False, retourne seulement le retrieval sans generation.

        Returns:
            Dict avec les cles : status, retrieved_cocktails, generated_recipe,
            cached, original_query, translated_query.
        """
        # Etape 1 : normalisation vers l'anglais pour SBERT
        en_query, was_translated = self.translator.to_english(user_query)

        # Guardrail sur la requete traduite (meilleure precision)
        guardrail_result = self._check_guardrail(en_query)
        if not guardrail_result["pass"]:
            return {
                "status": "rejected",
                "reason": "Requete hors-domaine cocktails.",
                "max_similarity": guardrail_result["max_similarity"],
                "retrieved_cocktails": [],
                "generated_recipe": None,
                "cached": False,
                "original_query": user_query,
                "translated_query": en_query if was_translated else None,
            }

        # Cache check sur la requete originale (stable entre sessions)
        cache_key = self._md5(user_query + str(top_k))
        if cache_key in self._cache:
            logger.debug("RAG cache hit : %s", cache_key[:8])
            return {
                "status": "success",
                "retrieved_cocktails": [],
                "generated_recipe": self._cache[cache_key],
                "cached": True,
                "max_similarity": guardrail_result["max_similarity"],
                "original_query": user_query,
                "translated_query": en_query if was_translated else None,
            }

        # Retrieval sur la requete traduite EN pour maximiser le recall SBERT
        results = self.recommender.recommend_by_query(en_query, top_k=top_k)

        if not generate:
            return {
                "status": "success",
                "retrieved_cocktails": [r.__dict__ for r in results],
                "generated_recipe": None,
                "cached": False,
                "max_similarity": guardrail_result["max_similarity"],
                "original_query": user_query,
                "translated_query": en_query if was_translated else None,
            }

        # Construction du contexte
        context = self._build_context(results)

        # Generation avec la requete EN pour le prompt GPT-2 (corpus anglophone)
        generated = self._generate(en_query, context, temperature)

        # Mise en cache (cle = requete originale pour stabilite multi-langue)
        self._cache[cache_key] = generated
        self._save_cache()

        return {
            "status": "success",
            "retrieved_cocktails": [r.__dict__ for r in results],
            "generated_recipe": generated,
            "cached": False,
            "max_similarity": guardrail_result["max_similarity"],
            "original_query": user_query,
            "translated_query": en_query if was_translated else None,
        }

    def _check_guardrail(self, query: str) -> dict:
        """
        Verifie que la requete appartient au domaine cocktails.
        Calcule la similarite avec l'ensemble du corpus.
        Seuil = 0.40 (calibre empiriquement sur 30 requetes labelisees).
        """
        query_vec = self.engine.encode_single(query)
        if self.recommender._embeddings is None:
            return {"pass": True, "max_similarity": 1.0}

        similarities = self.recommender._embeddings @ query_vec
        max_sim = float(np.max(similarities))

        return {
            "pass": max_sim >= self.guardrail_threshold,
            "max_similarity": max_sim,
        }

    def _build_context(self, results: list) -> str:
        """Construit le contexte RAG depuis les cocktails retrieved."""
        context_parts = []
        for r in results:
            context_parts.append(
                f"Cocktail : {r.name}\n"
                f"Categorie : {r.category}\n"
                f"Ingredients : {r.ingredients}\n"
                f"Instructions : {r.instructions}\n"
            )
        return "\n---\n".join(context_parts)

    def _generate(self, query: str, context: str, temperature: float) -> str:
        """
        Generation d'une recette via le modele.
        Utilise GPT-2 fine-tune si disponible, sinon generation par template.
        """
        try:
            return self._generate_with_model(query, context, temperature)
        except Exception as e:
            logger.warning("Generation modele echouee (%s), fallback template.", e)
            return self._generate_template(query, context)

    def _generate_with_model(self, query: str, context: str, temperature: float) -> str:
        """Generation via transformers pipeline (GPT-2 fine-tune ou autre modele local)."""
        from transformers import pipeline as hf_pipeline

        prompt = (
            f"Sur la base de ces cocktails de reference :\n{context}\n\n"
            f"Cree une nouvelle recette de cocktail correspondant a : {query}\n\n"
            f"Recette proposee :\nNom : "
        )
        generator = hf_pipeline("text-generation", model="gpt2", max_new_tokens=200, temperature=temperature)
        output = generator(prompt, return_full_text=False)[0]["generated_text"]
        return output.strip()

    def _generate_template(self, query: str, context: str) -> str:
        """Fallback : generation par template si le modele n'est pas disponible."""
        first_context_lines = context.split("\n")[:6]
        return (
            f"Recette generee pour '{query}' (mode template)\n\n"
            + "\n".join(first_context_lines) + "\n\n"
            "Preparation : Combiner les ingredients dans un shaker avec de la glace. "
            "Agiter 15 secondes. Filtrer et servir dans le verre approprie."
        )

    @staticmethod
    def _md5(text: str) -> str:
        """Hash MD5 pour la cle de cache."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
