"""
Module de generation de recettes de cocktails.
Fine-tuning GPT-2 sur le corpus cocktails pour la generation conditionelle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "gpt2-cocktails"


class CocktailGenerator:
    """
    Generateur de recettes de cocktails via GPT-2 fine-tune.

    Usage :
        gen = CocktailGenerator()
        gen.train(corpus_texts)                 # Fine-tuning
        recipe = gen.generate("vodka, citron")  # Generation
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_DIR
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy load du pipeline de generation."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def train(
        self,
        texts: list[str],
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 5e-5,
        save: bool = True,
    ) -> dict:
        """
        Fine-tuning GPT-2 sur le corpus cocktails.

        Args:
            texts: Liste de textes d'entrainement (recettes completes).
            epochs: Nombre d'epochs.
            batch_size: Taille de batch.
            lr: Taux d'apprentissage.
            save: Sauvegarder le modele apres entrainement.

        Returns:
            Dict avec les metriques d'entrainement.
        """
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
            import torch
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError("transformers et torch requis : pip install transformers torch")

        logger.info("Chargement tokenizer GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        class CocktailDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

            def __len__(self):
                return len(self.encodings["input_ids"])

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = item["input_ids"].clone()
                return item

        dataset = CocktailDataset(texts, tokenizer)

        model = GPT2LMHeadModel.from_pretrained("gpt2")

        training_args = TrainingArguments(
            output_dir=str(self.model_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            save_steps=500,
            logging_steps=100,
            overwrite_output_dir=True,
            no_cuda=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Debut du fine-tuning GPT-2 (%d textes, %d epochs)...", len(texts), epochs)
        train_result = trainer.train()

        if save:
            trainer.save_model(str(self.model_path))
            tokenizer.save_pretrained(str(self.model_path))
            logger.info("Modele sauvegarde : %s", self.model_path)

        return {
            "train_loss": train_result.training_loss,
            "epochs": epochs,
            "samples": len(texts),
        }

    def generate(
        self,
        ingredients: str | list[str],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """
        Genere une ou plusieurs recettes a partir d'ingredients.

        Args:
            ingredients: Ingredients disponibles (str ou liste).
            max_new_tokens: Longueur max de la generation.
            temperature: Creativite (0=deterministique, 1=aleatoire).
            top_p: Nucleus sampling.
            num_return_sequences: Nombre de recettes a generer.

        Returns:
            Liste de recettes generees.
        """
        if isinstance(ingredients, list):
            ingredients = ", ".join(ingredients)

        prompt = f"Recette avec {ingredients} :\nNom : "

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            return_full_text=True,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )

        return [o["generated_text"].strip() for o in outputs]

    def _load_pipeline(self) -> None:
        """Charge le pipeline depuis le modele fine-tune ou le modele de base."""
        from transformers import pipeline as hf_pipeline

        model_id = str(self.model_path) if self.model_path.exists() else "gpt2"
        logger.info("Chargement du generateur depuis : %s", model_id)
        self._pipeline = hf_pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
        )


def evaluate_generation(generated: str, reference: str) -> dict[str, float]:
    """
    Evalue la qualite d'une recette generee par rapport a une reference.

    Metriques :
    - BLEU-4 : n-gram overlap
    - ROUGE-L : longest common subsequence

    Args:
        generated: Texte genere.
        reference: Texte de reference.

    Returns:
        Dict avec les scores BLEU et ROUGE.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
    except ImportError:
        return {"bleu4": 0.0, "rouge_l": 0.0, "error": "nltk/rouge_score manquants"}

    # BLEU-4
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    sf = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=sf)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, generated)

    return {
        "bleu4": round(bleu, 4),
        "rouge_l": round(rouge["rougeL"].fmeasure, 4),
    }
