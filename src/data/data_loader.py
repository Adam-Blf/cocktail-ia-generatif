"""
Chargement et fusion des datasets cocktails.
Priorite : Parquet > CSV > synthetics.
Sources supportees :
  - cocktails_merged.parquet  (produit par scripts/download_datasets.py)
  - cocktaildb_main.parquet / .csv
  - iba_cocktails.parquet / .csv
  - kaggle_*.parquet  (si Kaggle credentials dispo)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Racine projet : data_loader.py vit dans src/data/, il faut donc remonter
# 3 niveaux (data -> src -> racine). Un .parent manquant ferait pointer
# vers src/data/raw (inexistant) et basculerait silencieusement sur les
# donnees synthetiques au lieu du vrai corpus de 1280 cocktails.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = _PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed"

FLAVOR_KEYWORDS: dict[str, list[str]] = {
    "sweet": ["sugar", "syrup", "liqueur", "triple sec", "grenadine", "honey", "creme", "baileys", "amaretto", "kahlua"],
    "sour":  ["lemon", "lime", "citrus", "juice", "sour", "vinegar", "tamarind"],
    "bitter": ["bitters", "campari", "aperol", "angostura", "amaro", "vermouth dry", "cynar", "fernet"],
    "strong": ["vodka", "gin", "rum", "tequila", "whiskey", "bourbon", "brandy", "cognac", "mezcal", "absinthe", "everclear"],
    "fresh":  ["mint", "cucumber", "basil", "soda", "tonic", "ginger beer", "club soda", "sparkling", "prosecco", "champagne"],
    "fruity": ["orange", "pineapple", "mango", "berry", "strawberry", "passion", "peach", "coconut", "raspberry", "blueberry", "cherry"],
}

REQUIRED_COLS = ["name", "category", "glass", "ingredients", "instructions"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet_or_csv(stem: str, path: Path | None = None) -> pd.DataFrame:
    """Charge un dataset depuis parquet si present, sinon CSV."""
    base = path or DATA_RAW
    for ext in (".parquet", ".csv"):
        candidate = base / f"{stem}{ext}"
        if candidate.exists() and candidate.stat().st_size > 100:
            try:
                df = pd.read_parquet(candidate) if ext == ".parquet" else pd.read_csv(candidate, on_bad_lines="skip")
                logger.info("Loaded %s (%d rows)", candidate.name, len(df))
                return df
            except Exception as exc:
                logger.warning("Failed to load %s: %s", candidate.name, exc)
    return pd.DataFrame()


def _normalize_cocktaildb(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un DataFrame issu de TheCocktailDB (schema direct ou schema brut Kaggle)."""
    # Schema direct (download_datasets.py)
    if "ingredients" in df.columns and "name" in df.columns:
        keep = [c for c in REQUIRED_COLS if c in df.columns]
        extra = [c for c in ["alcoholic", "thumbnail", "tags", "iba", "source"] if c in df.columns]
        return df[keep + extra].copy()

    # Schema brut TheCocktailDB (colonnes strIngredient1..15)
    col_map = {
        "strDrink": "name", "strCategory": "category",
        "strGlass": "glass", "strInstructions": "instructions",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    ing_cols = [c for c in df.columns if re.match(r"strIngredient\d+", c)]
    meas_cols = [c for c in df.columns if re.match(r"strMeasure\d+", c)]

    def _build(row: pd.Series) -> str:
        parts = []
        for ic, mc in zip(ing_cols, meas_cols):
            ing = row.get(ic)
            meas = row.get(mc)
            if pd.notna(ing) and str(ing).strip():
                part = str(ing).strip()
                if pd.notna(meas) and str(meas).strip():
                    part = f"{str(meas).strip()} {part}"
                parts.append(part)
        return ", ".join(parts)

    df["ingredients"] = df.apply(_build, axis=1)
    keep = [c for c in REQUIRED_COLS if c in df.columns]
    return df[keep].copy()


def _normalize_generic(df: pd.DataFrame, source_tag: str = "unknown") -> pd.DataFrame:
    """Normalisation generique : detecte les colonnes et les renomme."""
    name_candidates = ["name", "strDrink", "drink_name", "cocktail_name", "title"]
    ing_candidates = ["ingredients", "ingredient_list", "strIngredients", "recipe"]
    inst_candidates = ["instructions", "preparation", "directions", "method", "strInstructions"]
    cat_candidates = ["category", "strCategory", "type", "cocktail_type"]
    glass_candidates = ["glass", "strGlass", "glass_type", "serve_in"]

    def _find(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    rename = {}
    if (c := _find(name_candidates)) and c != "name":
        rename[c] = "name"
    if (c := _find(ing_candidates)) and c != "ingredients":
        rename[c] = "ingredients"
    if (c := _find(inst_candidates)) and c != "instructions":
        rename[c] = "instructions"
    if (c := _find(cat_candidates)) and c != "category":
        rename[c] = "category"
    if (c := _find(glass_candidates)) and c != "glass":
        rename[c] = "glass"

    df = df.rename(columns=rename)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""

    df["source"] = source_tag
    return df[REQUIRED_COLS + ["source"]].copy()


# ---------------------------------------------------------------------------
# Loaders principaux
# ---------------------------------------------------------------------------

def load_merged() -> pd.DataFrame:
    """Charge le dataset fusionne produit par download_datasets.py."""
    df = _load_parquet_or_csv("cocktails_merged")
    if not df.empty:
        return _normalize_generic(df, "cocktaildb_merged")
    return pd.DataFrame()


def load_cocktaildb_main() -> pd.DataFrame:
    df = _load_parquet_or_csv("cocktaildb_main")
    if not df.empty:
        return _normalize_cocktaildb(df)
    # Fallback legacy CSV (nom sans prefixe)
    df2 = _load_parquet_or_csv("cocktails")
    return _normalize_cocktaildb(df2) if not df2.empty else pd.DataFrame()


def load_iba() -> pd.DataFrame:
    df = _load_parquet_or_csv("iba_cocktails")
    if df.empty:
        df = _load_parquet_or_csv("cocktaildb_nonalcoholic")
    return _normalize_generic(df, "iba") if not df.empty else pd.DataFrame()


def load_kaggle_extras() -> list[pd.DataFrame]:
    """Charge tous les fichiers kaggle_*.parquet disponibles."""
    frames = []
    for f in sorted(DATA_RAW.glob("kaggle_*.parquet")):
        try:
            df = pd.read_parquet(f)
            frames.append(_normalize_generic(df, f.stem))
            logger.info("Loaded Kaggle extra: %s (%d rows)", f.name, len(df))
        except Exception as exc:
            logger.warning("Kaggle extra %s failed: %s", f.name, exc)
    return frames


# ---------------------------------------------------------------------------
# Pipeline public
# ---------------------------------------------------------------------------

def load_all_datasets() -> pd.DataFrame:
    """
    Charge et fusionne tous les datasets disponibles.
    Ordre de priorite :
      1. cocktails_merged.parquet  (le plus complet)
      2. cocktaildb_main + iba
      3. kaggle_*.parquet
      4. synthetics si rien d'autre
    """
    frames: list[pd.DataFrame] = []

    merged = load_merged()
    if not merged.empty:
        frames.append(merged)
        logger.info("Primary source: merged (%d rows)", len(merged))
    else:
        main = load_cocktaildb_main()
        if not main.empty:
            frames.append(main)
        iba = load_iba()
        if not iba.empty:
            frames.append(iba)

    extras = load_kaggle_extras()
    frames.extend(extras)

    if not frames:
        logger.warning("No dataset found, using synthetics.")
        return _generate_synthetic_data(n=500)

    df = pd.concat(frames, ignore_index=True)
    df = _deduplicate(df)
    df = _compute_text_full(df)
    df = _compute_flavor_profiles(df)

    logger.info("Final corpus: %d unique cocktails", len(df))
    return df


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df["_key"] = df["name"].fillna("").str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"])
    return df.reset_index(drop=True)


def _compute_text_full(df: pd.DataFrame) -> pd.DataFrame:
    df["text_full"] = (
        df.get("name", pd.Series("", index=df.index)).fillna("") + ". "
        + "Category: " + df.get("category", pd.Series("", index=df.index)).fillna("") + ". "
        + "Ingredients: " + df.get("ingredients", pd.Series("", index=df.index)).fillna("") + ". "
        + df.get("instructions", pd.Series("", index=df.index)).fillna("")
    )
    return df


def _compute_flavor_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    texts = df["ingredients"].fillna("").str.lower()
    for text in texts:
        profile = {}
        for flavor, keywords in FLAVOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            profile[flavor] = min(score / 3.0, 1.0)
        profiles.append(profile)
    profile_df = pd.DataFrame(profiles)
    return pd.concat([df.reset_index(drop=True), profile_df], axis=1)


def _generate_synthetic_data(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    categories = ["Cocktail", "Shot", "Punch / Party Drink", "Ordinary Drink", "Beer", "Shake", "Mocktail"]
    glasses = ["Highball glass", "Cocktail glass", "Old-fashioned glass", "Collins glass", "Shot glass", "Margarita glass"]
    base_spirits = ["Vodka", "Gin", "Rum", "Tequila", "Whiskey", "Bourbon", "Brandy", "Mezcal", "Pisco"]
    mixers = ["Lime juice", "Lemon juice", "Orange juice", "Cranberry juice", "Soda water", "Tonic water", "Ginger beer", "Pineapple juice", "Coconut cream"]
    sweeteners = ["Simple syrup", "Triple sec", "Grenadine", "Agave syrup", "Honey", "Chambord", "Amaretto", "Kahlua"]
    garnishes = ["Mint", "Lime wedge", "Salt rim", "Cherry", "Orange slice", "Cucumber", "Rosemary", "Basil"]

    cocktail_names = [
        "Mojito", "Margarita", "Cosmopolitan", "Negroni", "Old Fashioned", "Daiquiri",
        "Aperol Spritz", "Pina Colada", "Long Island Iced Tea", "Moscow Mule", "Espresso Martini",
        "Whiskey Sour", "Bloody Mary", "Sex on the Beach", "Mai Tai", "Tom Collins", "Gimlet",
        "Sidecar", "Manhattan", "Rob Roy", "Singapore Sling", "Dark and Stormy", "Paloma",
        "Tequila Sunrise", "Blue Lagoon", "Caipirinha", "French 75", "Last Word", "Paper Plane",
        "Bee's Knees", "White Negroni", "Spicy Margarita", "Tommy's Margarita", "Jungle Bird",
    ]
    for i in range(len(cocktail_names), n):
        cocktail_names.append(f"Cocktail Special {i + 1}")

    records = []
    for i in range(n):
        spirit = rng.choice(base_spirits)
        mixer = rng.choice(mixers)
        sweetener = rng.choice(sweeteners)
        garnish = rng.choice(garnishes)
        n_extra = rng.integers(0, 3)
        extras = rng.choice(mixers + sweeteners, size=int(n_extra), replace=False).tolist() if n_extra > 0 else []

        ingredients_list = [f"60ml {spirit}", f"30ml {mixer}", f"15ml {sweetener}"] + [f"10ml {e}" for e in extras] + [garnish]
        instructions = (
            f"Combine {spirit.lower()}, {mixer.lower()} and {sweetener.lower()} in a shaker with ice. "
            f"Shake well for 10 seconds. Strain into a glass and garnish with {garnish.lower()}."
        )

        records.append({
            "name": cocktail_names[i % len(cocktail_names)] + (f" #{i}" if i >= len(cocktail_names) else ""),
            "category": rng.choice(categories),
            "glass": rng.choice(glasses),
            "ingredients": ", ".join(ingredients_list),
            "instructions": instructions,
            "source": "synthetic",
        })

    df = pd.DataFrame(records)
    df = _compute_text_full(df)
    df = _compute_flavor_profiles(df)
    return df
