"""
Module de chargement et fusion des datasets cocktails.
Gere les 4 sources Kaggle et produit un DataFrame unifie.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# Profils de saveurs cibles pour chaque ingredient cle
FLAVOR_KEYWORDS: dict[str, list[str]] = {
    "sweet": ["sugar", "syrup", "liqueur", "triple sec", "grenadine", "honey", "creme", "baileys"],
    "sour": ["lemon", "lime", "citrus", "juice", "sour"],
    "bitter": ["bitters", "campari", "aperol", "angostura", "amaro", "vermouth dry"],
    "strong": ["vodka", "gin", "rum", "tequila", "whiskey", "bourbon", "brandy", "cognac"],
    "fresh": ["mint", "cucumber", "basil", "soda", "tonic", "ginger beer"],
    "fruity": ["orange", "pineapple", "mango", "berry", "strawberry", "passion", "peach", "coconut"],
}


def load_main_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Charge le dataset principal aadyasingh55/cocktails.
    Retourne un DataFrame normalise avec colonnes standardisees.
    """
    filepath = path or DATA_RAW / "cocktails.csv"

    if not filepath.exists():
        logger.warning("Dataset principal introuvable, generation de donnees simulees.")
        return _generate_synthetic_data(n=400)

    df = pd.read_csv(filepath)
    logger.info("Dataset principal charge : %d cocktails", len(df))
    return _normalize_main(df)


def load_iba_dataset(path: Path | None = None) -> pd.DataFrame:
    """Charge le dataset IBA officiel (90 cocktails reference)."""
    filepath = path or DATA_RAW / "iba_cocktails.csv"

    if not filepath.exists():
        logger.warning("Dataset IBA introuvable.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    return _normalize_iba(df)


def load_all_datasets() -> pd.DataFrame:
    """
    Fusionne les 4 datasets en un corpus unique deduplique.
    Retourne un DataFrame avec colonnes : name, category, glass, ingredients,
    instructions, flavor_profile, source, text_full.
    """
    frames = []

    main = load_main_dataset()
    if not main.empty:
        main["source"] = "cocktaildb_main"
        frames.append(main)

    iba = load_iba_dataset()
    if not iba.empty:
        iba["source"] = "iba_official"
        frames.append(iba)

    if not frames:
        logger.warning("Aucun dataset charge, utilisation des donnees synthetiques.")
        return _generate_synthetic_data(n=500)

    df = pd.concat(frames, ignore_index=True)
    df = _deduplicate(df)
    df = _compute_text_full(df)
    df = _compute_flavor_profiles(df)

    logger.info("Corpus final : %d cocktails uniques", len(df))
    return df


def _normalize_main(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes du dataset principal."""
    col_map = {
        "strDrink": "name",
        "strCategory": "category",
        "strGlass": "glass",
        "strInstructions": "instructions",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Fusionner les colonnes ingredients (strIngredient1..15)
    ingredient_cols = [c for c in df.columns if re.match(r"strIngredient\d+", c)]
    measure_cols = [c for c in df.columns if re.match(r"strMeasure\d+", c)]

    def _build_ingredients(row: pd.Series) -> str:
        parts = []
        for ing_col, meas_col in zip(ingredient_cols, measure_cols):
            ing = row.get(ing_col)
            meas = row.get(meas_col)
            if pd.notna(ing) and str(ing).strip():
                part = str(ing).strip()
                if pd.notna(meas) and str(meas).strip():
                    part = f"{str(meas).strip()} {part}"
                parts.append(part)
        return ", ".join(parts)

    df["ingredients"] = df.apply(_build_ingredients, axis=1)

    keep = ["name", "category", "glass", "ingredients", "instructions"]
    return df[[c for c in keep if c in df.columns]].copy()


def _normalize_iba(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise le dataset IBA."""
    col_map = {
        "name": "name",
        "category": "category",
        "glass": "glass",
        "ingredients": "ingredients",
        "preparation": "instructions",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    keep = ["name", "category", "glass", "ingredients", "instructions"]
    return df[[c for c in keep if c in df.columns]].copy()


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplication par nom normalise (casse, espaces)."""
    df["_name_key"] = df["name"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["_name_key"], keep="first")
    df = df.drop(columns=["_name_key"])
    return df.reset_index(drop=True)


def _compute_text_full(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le champ texte concatene pour l'embedding."""
    df["text_full"] = (
        df.get("name", "").fillna("") + ". "
        + "Categorie : " + df.get("category", "").fillna("") + ". "
        + "Ingredients : " + df.get("ingredients", "").fillna("") + ". "
        + df.get("instructions", "").fillna("")
    )
    return df


def _compute_flavor_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribue un score de saveur a chaque cocktail en fonction des ingredients.
    Produit un vecteur de 6 dimensions : sweet, sour, bitter, strong, fresh, fruity.
    """
    profiles = []
    ingredient_texts = df["ingredients"].fillna("").str.lower()

    for text in ingredient_texts:
        profile = {}
        for flavor, keywords in FLAVOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            profile[flavor] = min(score / 3.0, 1.0)  # normalise [0, 1]
        profiles.append(profile)

    profile_df = pd.DataFrame(profiles)
    return pd.concat([df, profile_df], axis=1)


def _generate_synthetic_data(n: int = 500) -> pd.DataFrame:
    """
    Genere des donnees synthetiques realistes si les CSV Kaggle ne sont pas disponibles.
    Utilise pour les tests et le developpement hors-ligne.
    """
    rng = np.random.default_rng(42)

    categories = ["Cocktail", "Shot", "Punch / Party Drink", "Ordinary Drink", "Beer", "Shake"]
    glasses = ["Highball glass", "Cocktail glass", "Old-fashioned glass", "Collins glass", "Shot glass"]

    base_spirits = ["Vodka", "Gin", "Rum", "Tequila", "Whiskey", "Bourbon", "Brandy"]
    mixers = ["Lime juice", "Lemon juice", "Orange juice", "Cranberry juice", "Soda water", "Tonic water", "Ginger beer"]
    sweeteners = ["Simple syrup", "Triple sec", "Grenadine", "Agave syrup", "Honey"]
    garnishes = ["Mint", "Lime wedge", "Salt rim", "Cherry", "Orange slice", "Cucumber"]

    cocktail_names = [
        "Mojito", "Margarita", "Cosmopolitan", "Negroni", "Old Fashioned",
        "Daiquiri", "Aperol Spritz", "Pina Colada", "Long Island Iced Tea", "Moscow Mule",
        "Espresso Martini", "Whiskey Sour", "Bloody Mary", "Sex on the Beach", "Mai Tai",
        "Tom Collins", "Gimlet", "Sidecar", "Manhattan", "Rob Roy",
        "Singapore Sling", "Dark and Stormy", "Paloma", "Tequila Sunrise", "Blue Lagoon",
    ]
    # Completer jusqu'a n noms uniques
    for i in range(len(cocktail_names), n):
        cocktail_names.append(f"Cocktail Special {i + 1}")

    records = []
    for i in range(n):
        spirit = rng.choice(base_spirits)
        mixer = rng.choice(mixers)
        sweetener = rng.choice(sweeteners)
        garnish = rng.choice(garnishes)
        n_extra = rng.integers(0, 3)
        extras = rng.choice(mixers + sweeteners, size=n_extra, replace=False).tolist() if n_extra > 0 else []

        ingredients_list = [f"60ml {spirit}", f"30ml {mixer}", f"15ml {sweetener}"] + [f"10ml {e}" for e in extras] + [garnish]

        instructions = (
            f"Dans un shaker avec des glacons, combiner {spirit.lower()}, {mixer.lower()} "
            f"et {sweetener.lower()}. Agiter vigoureusement 10 secondes. "
            f"Filtrer dans un verre et garnir avec {garnish.lower()}."
        )

        records.append({
            "name": cocktail_names[i % len(cocktail_names)] + (f" #{i}" if i >= 25 else ""),
            "category": rng.choice(categories),
            "glass": rng.choice(glasses),
            "ingredients": ", ".join(ingredients_list),
            "instructions": instructions,
        })

    df = pd.DataFrame(records)
    df = _compute_text_full(df)
    df = _compute_flavor_profiles(df)
    df["source"] = "synthetic"
    return df
