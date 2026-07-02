"""
MixCraft - Systeme Generatif Intelligent pour la Creation de Cocktails
M1 Data Engineering & IA, EFREI Paris, 2025-2026

Structure par domaine metier :
  src/data/   - chargement, normalisation, deduplication du corpus
  src/nlp/    - embeddings SBERT, traduction multilingue
  src/rag/    - recommandeur semantique, pipeline RAG, generation Gemini
  src/eval/   - metriques Precision@K, NDCG@K, BLEU, ROUGE
"""

__version__ = "1.1.0"
__authors__ = ["Adam Beloucif", "Emilien Morice"]

# Retrocompatibilite : les imports plats fonctionnent toujours
from .data.data_loader import load_all_datasets
from .nlp.embeddings import EmbeddingEngine
from .nlp.translator import QueryTranslator
from .rag.recommender import CocktailRecommender
from .rag.rag_pipeline import RAGPipeline
from .rag.generator import CocktailGenerator
from .eval.evaluation import EvalResults, evaluate_recommender, evaluate_guardrail
