# JURY_GUIDE - MixCraft IA Generative Cocktails

Guide de navigation rapide pour la soutenance M1 Data Engineering & IA.
Chaque section repond a une question type jury avec les references exactes fichier:ligne.

---

## 1. DONNEES - Sources, volume, fusion

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Quelles sont vos sources de donnees ?" | `src/data_loader.py` | 1-9 | Docstring module : 5 sources listees |
| "Combien de cocktails avez-vous ?" | `src/data_loader.py` | 201-207 | `load_all_datasets` : log "Final corpus: 1280 unique cocktails" |
| "Comment vous gerez les doublons ?" | `src/data_loader.py` | 214-217 | `_deduplicate` : cle MD5 sur nom normalise |
| "Pourquoi parquet et pas CSV ?" | `src/data_loader.py` | 41-53 | `_load_parquet_or_csv` : fallback parquet -> CSV |
| "Comment vous construisez le texte pour SBERT ?" | `src/data_loader.py` | 220-227 | `_compute_text_full` : concat name+category+ingredients+instructions |
| "Comment vous calculez les profils de saveur ?" | `src/data_loader.py` | 24-32 | `FLAVOR_KEYWORDS` : 6 dimensions (sweet/sour/bitter/strong/fresh/fruity) |
| "Vous avez des datasets Kaggle ?" | `src/data_loader.py` | 154-164 | `load_kaggle_extras` : glob kaggle_*.parquet auto-discovery |
| "Schema de normalisation entre sources ?" | `src/data_loader.py` | 91-123 | `_normalize_generic` : detection colonnes par priorite |

---

## 2. NLP - Embeddings SBERT

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Quel modele d'embedding utilisez-vous ?" | `src/embeddings.py` | 21 | `DEFAULT_MODEL = "all-MiniLM-L6-v2"` (384 dims) |
| "Pourquoi L2-normalize les vecteurs ?" | `src/embeddings.py` | 79 | Commentaire inline : cosinus via produit scalaire |
| "Vous avez un cache d'embeddings ?" | `src/embeddings.py` | 65-86 | Cache pickle MD5 sur disque, evite recomputation |
| "Comment vous calculez la similarite ?" | `src/embeddings.py` | 93-95 | `similarity` : produit scalaire (vecteurs normalises) |
| "Quelle est la dimension des embeddings ?" | `src/embeddings.py` | 110-113 | `embedding_dim` : 384 pour all-MiniLM-L6-v2 |
| "Comment vous gerez les requetes en francais ?" | `src/translator.py` | 1-13 | Docstring : pipeline detection -> traduction EN |
| "Quel librairie de traduction ?" | `src/translator.py` | 43-44 | `GoogleTranslator` (deep_translator, gratuit, offline detection) |
| "Vous avez un cache pour les traductions ?" | `src/translator.py` | 36-47 | `@lru_cache(maxsize=512)` sur `_translate_cached` |
| "Pourquoi traduire en anglais ?" | `src/translator.py` | 6-9 | Docstring : SBERT anglophone, +15-25% recall empirique sur FR |

---

## 3. RECOMMANDATION - Moteur semantique

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Comment fonctionne la recommandation ?" | `src/recommender.py` | 32-38 | Docstring classe : SBERT + cosinus, 2 modes |
| "Comment vous indexez le corpus ?" | `src/recommender.py` | 47-65 | `fit()` : encode tous les textes, stocke la matrice |
| "Requete par description libre ?" | `src/recommender.py` | 67-94 | `recommend_by_query` : encode query, produit matriciel, top-K |
| "Requete par ingredients disponibles ?" | `src/recommender.py` | 96-137 | `recommend_by_ingredients` : score semantique 50% + coverage 50% |
| "Comment vous filtrez par categorie ?" | `src/recommender.py` | 89-92 | Masque booleens sur scores avant argsort |
| "Precision@K, NDCG@K ?" | `src/recommender.py` | 170-190 | Fonctions standalone `precision_at_k`, `recall_at_k`, `ndcg_at_k` |

---

## 4. GENERATION - GPT-2 Fine-tune

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Quel modele generatif utilisez-vous ?" | `src/generator.py` | 15 | `MODEL_DIR = models/gpt2-cocktails` (GPT-2 fine-tune) |
| "Comment vous avez fait le fine-tuning ?" | `src/generator.py` | 39-122 | `train()` : HuggingFace Trainer, 3 epochs, batch 8, lr 5e-5 |
| "Sur combien de recettes ?" | `src/generator.py` | 110 | Log "Debut du fine-tuning GPT-2 (N textes, 3 epochs)" -> 447 recettes |
| "Comment vous generez une recette ?" | `src/generator.py` | 124-160 | `generate()` : prompt "Recette avec X : Nom : " + top_p sampling |
| "Vous evaluez la generation ?" | `src/generator.py` | 175-209 | `evaluate_generation` : BLEU-4 + ROUGE-L |
| "Fallback si le modele n'est pas dispo ?" | `src/generator.py` | 162-167 | `_load_pipeline` : fallback sur gpt2 base si fine-tune absent |

---

## 5. PIPELINE RAG - Retrieval Augmented Generation

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Expliquez votre pipeline RAG" | `src/rag_pipeline.py` | 27-52 | Docstring classe : 6 etapes (traduction -> guardrail -> retrieval -> contexte -> generation -> cache) |
| "C'est quoi le guardrail ?" | `src/rag_pipeline.py` | 139-155 | `_check_guardrail` : similarite max corpus, seuil 0.40 |
| "Pourquoi seuil 0.40 ?" | `src/rag_pipeline.py` | 24 | `GUARDRAIL_THRESHOLD = 0.40` : calibre sur 30 requetes labelisees |
| "Vous avez un cache pour le RAG ?" | `src/rag_pipeline.py` | 91-103 | Cache JSON MD5(query+top_k), stable entre sessions |
| "Comment vous construisez le contexte ?" | `src/rag_pipeline.py` | 157-167 | `_build_context` : concatene nom+categorie+ingredients+instructions |
| "Le RAG traduit les requetes ?" | `src/rag_pipeline.py` | 74-75 | `en_query, was_translated = self.translator.to_english(user_query)` |
| "La requete originale est dans la reponse ?" | `src/rag_pipeline.py` | 128-137 | `original_query` et `translated_query` dans le dict retourne |
| "Sequence complete d'une requete ?" | `src/rag_pipeline.py` | 54-137 | Methode `query()` : traduction -> guardrail -> cache -> retrieval -> generation |

---

## 6. EVALUATION - Metriques

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Comment vous evaluez le systeme ?" | `src/evaluation.py` | 1-4 | Docstring : Precision@K, Recall@K, NDCG@K + BLEU/ROUGE |
| "Evaluation du recommandeur ?" | `src/evaluation.py` | 55-102 | `evaluate_recommender` : boucle sur test_queries, calcule P@K/R@K/NDCG@K |
| "Evaluation du guardrail ?" | `src/evaluation.py` | 105-157 | `evaluate_guardrail` : TP/FP/TN/FN sur requetes in/out-domain |
| "C'est quoi les requetes hors-domaine test ?" | `src/evaluation.py` | 32-38 | `out_of_domain_queries` default : velo, capitale, film, algorithme, meteo |
| "Format des resultats ?" | `src/evaluation.py` | 40-52 | `EvalResults` dataclass : tous les scores + config |
| "Vous sauvegardez les resultats ?" | `src/evaluation.py` | 175-180 | `save_results` : JSON prettify |

---

## 7. UI - Interface Streamlit

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "C'est quoi l'interface ?" | `app/app.py` | 1-4 | Docstring : 3 onglets (Recommandation, Generation, Analyse) |
| "Quelle palette de couleurs ?" | `app/app.py` | 30-42 | Variables CSS : navy #163767, pink #FF43B8 |
| "Comment vous chargez les donnees ?" | `app/app.py` | 16-20 | Imports : `load_all_datasets`, `CocktailRecommender`, `RAGPipeline` |

---

## 8. DONNEES BAS NIVEAU - Scripts

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Comment vous telechargez les donnees ?" | `scripts/download_datasets.py` | 1-9 | Docstring : TheCocktailDB API + Kaggle |
| "Source API principale ?" | `scripts/download_datasets.py` | 27 | `COCKTAILDB_BASE = "https://www.thecocktaildb.com/api/json/v1/1"` |

---

## 9. TESTS

| Question jury | Fichier | Ligne | Detail |
|---|---|---|---|
| "Vous testez les embeddings ?" | `tests/test_embeddings.py` | - | Tests EmbeddingEngine : encode, cache, similarity |
| "Tests du generateur ?" | `tests/test_generator.py` | - | Tests CocktailGenerator : generate, evaluate_generation |
| "Tests du recommandeur ?" | `tests/test_recommender.py` | - | Tests CocktailRecommender : fit, recommend_by_query |

---

## 10. CHIFFRES CLES - A connaitre par coeur

| Metrique | Valeur | Reference |
|---|---|---|
| Corpus cocktails | 1280 uniques | `src/data_loader.py:206` |
| Modele embeddings | all-MiniLM-L6-v2 (384 dims) | `src/embeddings.py:21` |
| Seuil guardrail | 0.40 | `src/rag_pipeline.py:24` |
| GPT-2 fine-tune | 3 epochs, 447 recettes | `src/generator.py:39-110` |
| Cache embeddings | Pickle MD5 | `src/embeddings.py:65-86` |
| Cache RAG | JSON MD5 | `src/rag_pipeline.py:92` |
| Traduction | langdetect + deep-translator | `src/translator.py:36-47` |
| Profils saveur | 6 dimensions | `src/data_loader.py:24-32` |
