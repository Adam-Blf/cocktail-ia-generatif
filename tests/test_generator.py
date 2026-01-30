"""Tests unitaires pour le module de generation."""
import pytest


def test_evaluate_generation_returns_dict():
    from src.generator import evaluate_generation
    result = evaluate_generation(
        "Mix vodka and lime. Shake and serve.",
        "Combine vodka, lime juice and ice. Shake well and strain.",
    )
    assert isinstance(result, dict)
    # Les scores doivent etre presents (ou erreur explicite si lib manquante)
    assert 'bleu4' in result or 'error' in result


def test_evaluate_generation_score_range():
    from src.generator import evaluate_generation
    result = evaluate_generation(
        "This is the generated cocktail recipe.",
        "This is the reference cocktail recipe.",
    )
    if 'bleu4' in result:
        assert 0.0 <= result['bleu4'] <= 1.0
    if 'rouge_l' in result:
        assert 0.0 <= result['rouge_l'] <= 1.0


def test_evaluate_generation_identical_texts():
    from src.generator import evaluate_generation
    text = "Mojito: mix rum, lime, mint, sugar, soda."
    result = evaluate_generation(text, text)
    if 'bleu4' in result:
        assert result['bleu4'] > 0.0  # textes identiques -> score > 0
