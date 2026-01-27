"""
Module d'evaluation du systeme MixCraft.
Metriques de recommandation (Precision@K, Recall@K, NDCG@K) et de generation (BLEU, ROUGE).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration de l'evaluation."""
    k_values: list[int] = None
    n_queries: int = 50
    guardrail_test_size: int = 30
    out_of_domain_queries: list[str] = None

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]
        if self.out_of_domain_queries is None:
            self.out_of_domain_queries = [
                "repare mon velo",
                "quelle est la capitale de la France",
                "recommande-moi un film",
                "code un algorithme de tri",
                "meteorologie de demain",
            ]


@dataclass
class EvalResults:
    """Resultats d'une evaluation complete."""
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    guardrail_precision: float
    guardrail_recall: float
    mean_bleu4: float
    mean_rouge_l: float
    n_queries_evaluated: int
    config: dict


def evaluate_recommender(
    recommender,
    test_queries: list[dict],
    config: Optional[EvalConfig] = None,
) -> EvalResults:
    """
    Evalue le moteur de recommandation sur un jeu de requetes de test.

    Args:
        recommender: Instance de CocktailRecommender fittee.
        test_queries: Liste de dicts {query, relevant_names}.
        config: Configuration de l'evaluation.

    Returns:
        EvalResults avec toutes les metriques.
    """
    cfg = config or EvalConfig()
    max_k = max(cfg.k_values)

    p_at_k = {k: [] for k in cfg.k_values}
    r_at_k = {k: [] for k in cfg.k_values}
    n_at_k = {k: [] for k in cfg.k_values}

    for item in test_queries[:cfg.n_queries]:
        query = item["query"]
        relevant = set(item.get("relevant_names", []))
        if not relevant:
            continue

        results = recommender.recommend_by_query(query, top_k=max_k)
        retrieved_names = [r.name for r in results]

        for k in cfg.k_values:
            p_at_k[k].append(_precision_at_k(relevant, retrieved_names, k))
            r_at_k[k].append(_recall_at_k(relevant, retrieved_names, k))
            n_at_k[k].append(_ndcg_at_k(relevant, retrieved_names, k))

    return EvalResults(
        precision_at_k={k: float(np.mean(v)) for k, v in p_at_k.items()},
        recall_at_k={k: float(np.mean(v)) for k, v in r_at_k.items()},
        ndcg_at_k={k: float(np.mean(v)) for k, v in n_at_k.items()},
        guardrail_precision=0.0,
        guardrail_recall=0.0,
        mean_bleu4=0.0,
        mean_rouge_l=0.0,
        n_queries_evaluated=min(len(test_queries), cfg.n_queries),
        config=asdict(cfg),
    )


def evaluate_guardrail(pipeline, config: Optional[EvalConfig] = None) -> dict:
    """
    Evalue le guardrail semantique sur des requetes in-domain et hors-domain.

    Args:
        pipeline: Instance de RAGPipeline.
        config: Configuration.

    Returns:
        Dict avec precision, recall, F1 du guardrail.
    """
    cfg = config or EvalConfig()

    # Requetes hors-domaine (doivent etre rejetees)
    ood_queries = cfg.out_of_domain_queries

    # Requetes in-domain (doivent passer)
    in_domain_queries = [
        "cocktail avec vodka et citron",
        "quelque chose de frais et sucre",
        "mojito ou daiquiri",
        "recette avec rhum et menthe",
        "long drink pour ete",
    ]

    tp = fp = tn = fn = 0

    for q in ood_queries:
        result = pipeline._check_guardrail(q)
        if not result["pass"]:
            tn += 1  # Rejete correctement
        else:
            fp += 1  # Passe a tort

    for q in in_domain_queries:
        result = pipeline._check_guardrail(q)
        if result["pass"]:
            tp += 1  # Accepte correctement
        else:
            fn += 1  # Rejete a tort

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rejection_rate = tn / len(ood_queries) if ood_queries else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "rejection_rate_ood": round(rejection_rate, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def _recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(len(relevant), 1)


def _ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(retrieved[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def save_results(results: EvalResults, path: Path) -> None:
    """Sauvegarde les resultats en JSON."""
    data = asdict(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Resultats sauvegardes : %s", path)


def print_summary(results: EvalResults) -> None:
    """Affiche un resume des metriques."""
    print("\n=== Resultats d'evaluation MixCraft ===\n")
    print(f"Requetes evaluees : {results.n_queries_evaluated}")
    print("\nRecommandation (Precision@K) :")
    for k, v in results.precision_at_k.items():
        print(f"  P@{k:2d} = {v:.3f}")
    print("\nRecommandation (NDCG@K) :")
    for k, v in results.ndcg_at_k.items():
        print(f"  NDCG@{k:2d} = {v:.3f}")
    print(f"\nGeneration - BLEU-4  : {results.mean_bleu4:.3f}")
    print(f"Generation - ROUGE-L : {results.mean_rouge_l:.3f}")
    print("=" * 40)
