"""
Telechargement des datasets cocktails depuis sources publiques.
Sources :
  - TheCocktailDB public API (no-auth, ~600 cocktails)
  - Fichiers Kaggle si credentials disponibles (optionnel)

Usage : python scripts/download_datasets.py
Output : data/raw/*.parquet  + data/raw/*.csv
"""

from __future__ import annotations

import logging
import string
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

COCKTAILDB_BASE = "https://www.thecocktaildb.com/api/json/v1/1"


# ---------------------------------------------------------------------------
# TheCocktailDB
# ---------------------------------------------------------------------------

def _fetch_by_letter(letter: str, session: requests.Session) -> list[dict]:
    url = f"{COCKTAILDB_BASE}/search.php?f={letter}"
    try:
        r = session.get(url, timeout=12)
        r.raise_for_status()
        return r.json().get("drinks") or []
    except Exception as exc:
        log.warning("letter %s failed: %s", letter, exc)
        return []


def _parse_drink(d: dict) -> dict:
    ing_parts = []
    for i in range(1, 16):
        ing = (d.get(f"strIngredient{i}") or "").strip()
        meas = (d.get(f"strMeasure{i}") or "").strip()
        if ing:
            ing_parts.append(f"{meas} {ing}".strip() if meas else ing)

    return {
        "id": d.get("idDrink", ""),
        "name": (d.get("strDrink") or "").strip(),
        "category": (d.get("strCategory") or "").strip(),
        "alcoholic": (d.get("strAlcoholic") or "").strip(),
        "glass": (d.get("strGlass") or "").strip(),
        "ingredients": ", ".join(ing_parts),
        "instructions": (d.get("strInstructions") or "").strip(),
        "thumbnail": (d.get("strDrinkThumb") or "").strip(),
        "tags": (d.get("strTags") or "").strip(),
        "iba": (d.get("strIBA") or "").strip(),
        "source": "cocktaildb",
    }


def download_cocktaildb() -> pd.DataFrame:
    log.info("TheCocktailDB - fetch all 26 letters ...")
    session = requests.Session()
    session.headers["User-Agent"] = "MixCraft-DataPipeline/1.0"

    records: list[dict] = []
    for letter in string.ascii_lowercase:
        drinks = _fetch_by_letter(letter, session)
        records.extend([_parse_drink(d) for d in drinks])
        log.info("  %s -> %d drinks  (cumul %d)", letter, len(drinks), len(records))
        time.sleep(0.12)

    df = pd.DataFrame(records).drop_duplicates(subset=["id"]).reset_index(drop=True)
    log.info("CocktailDB total: %d cocktails", len(df))

    df.to_parquet(DATA_RAW / "cocktaildb_main.parquet", index=False)
    df.to_csv(DATA_RAW / "cocktaildb_main.csv", index=False)
    log.info("Saved: cocktaildb_main.parquet + .csv")
    return df


def download_by_category() -> pd.DataFrame:
    """Recupere tous les cocktails par categorie (complement de la recherche alphabetique)."""
    log.info("TheCocktailDB - fetch by category ...")
    session = requests.Session()

    categories_url = f"{COCKTAILDB_BASE}/list.php?c=list"
    try:
        cats = [c["strCategory"] for c in (session.get(categories_url, timeout=10).json().get("drinks") or [])]
    except Exception:
        cats = []
    log.info("  categories found: %d", len(cats))

    records: list[dict] = []
    for cat in cats:
        try:
            r = session.get(f"{COCKTAILDB_BASE}/filter.php?c={requests.utils.quote(cat)}", timeout=10)
            items = r.json().get("drinks") or []
            for item in items:
                drink_id = item.get("idDrink")
                if drink_id:
                    r2 = session.get(f"{COCKTAILDB_BASE}/lookup.php?i={drink_id}", timeout=10)
                    details = (r2.json().get("drinks") or [None])[0]
                    if details:
                        records.append(_parse_drink(details))
                    time.sleep(0.08)
        except Exception as exc:
            log.warning("category %s: %s", cat, exc)

    df = pd.DataFrame(records).drop_duplicates(subset=["id"]).reset_index(drop=True) if records else pd.DataFrame()
    if not df.empty:
        df.to_parquet(DATA_RAW / "cocktaildb_by_category.parquet", index=False)
        log.info("Saved: cocktaildb_by_category.parquet  (%d rows)", len(df))
    return df


def download_non_alcoholic() -> pd.DataFrame:
    """Cocktails sans alcool (mocktails)."""
    log.info("TheCocktailDB - non-alcoholic ...")
    session = requests.Session()
    try:
        r = session.get(f"{COCKTAILDB_BASE}/filter.php?a=Non_Alcoholic", timeout=10)
        items = r.json().get("drinks") or []
    except Exception:
        return pd.DataFrame()

    records = []
    for item in items:
        drink_id = item.get("idDrink")
        if drink_id:
            try:
                r2 = session.get(f"{COCKTAILDB_BASE}/lookup.php?i={drink_id}", timeout=10)
                details = (r2.json().get("drinks") or [None])[0]
                if details:
                    records.append(_parse_drink(details))
                time.sleep(0.08)
            except Exception:
                pass

    df = pd.DataFrame(records).drop_duplicates(subset=["id"]).reset_index(drop=True) if records else pd.DataFrame()
    if not df.empty:
        df.to_parquet(DATA_RAW / "cocktaildb_nonalcoholic.parquet", index=False)
        log.info("Saved: cocktaildb_nonalcoholic.parquet  (%d rows)", len(df))
    return df


def build_ingredient_index(df: pd.DataFrame) -> None:
    """Index des ingredients pour les analyses."""
    import re
    all_ings = (
        df["ingredients"]
        .fillna("")
        .str.lower()
        .str.split(r"[,\n]")
        .explode()
        .str.strip()
        .str.replace(r"\d+(\.\d+)?\s*(ml|oz|cl|dash|tsp|tbsp|cup|part|drop|pinch|splash|shot)\s*", "", regex=True)
        .str.strip()
    )
    counts = all_ings[all_ings.str.len() > 2].value_counts()
    out = DATA_RAW / "ingredient_index.parquet"
    counts.reset_index().to_parquet(out, index=False)
    log.info("Ingredient index -> %s  (%d unique ingredients)", out.name, len(counts))


def try_kaggle_downloads() -> None:
    """Tente de telecharger les datasets Kaggle si credentials presents (Kaggle CLI v2+)."""
    import subprocess
    import shutil

    if not shutil.which("kaggle"):
        log.info("Kaggle CLI not found - skip.")
        return

    # Verify auth by listing datasets (fast no-op)
    check = subprocess.run(["kaggle", "datasets", "list", "--max-size", "1"], capture_output=True, text=True)
    if check.returncode != 0:
        log.info("Kaggle auth not valid - skip. Run: kaggle auth login")
        return
    log.info("Kaggle auth OK")

    datasets = [
        ("aadyasingh55", "cocktails"),
        ("pxxthik", "the-cocktail-db-recipe-collection"),
        ("joakark", "cocktail-ingredients-and-instructions"),
        ("mexwell", "iba-cocktails"),
        ("shuyangli94", "cocktail-datasets"),
        ("ai-first", "cocktaildb"),
    ]

    for owner, ds in datasets:
        dest = DATA_RAW / f"kaggle_{ds.replace('-', '_')}"
        dest.mkdir(exist_ok=True)
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", f"{owner}/{ds}", "--unzip", "--path", str(dest)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log.warning("  Kaggle %s/%s failed: %s", owner, ds, result.stderr.strip())
                continue
            log.info("  Kaggle %s/%s -> %s", owner, ds, dest.name)
            for f in dest.glob("*.csv"):
                try:
                    df_k = pd.read_csv(f, on_bad_lines="skip")
                    out_parquet = DATA_RAW / f"kaggle_{ds.replace('-', '_')}.parquet"
                    df_k.to_parquet(out_parquet, index=False)
                    log.info("    converted -> %s  (%d rows)", out_parquet.name, len(df_k))
                except Exception as exc:
                    log.warning("    CSV parse failed %s: %s", f.name, exc)
        except Exception as exc:
            log.warning("  Kaggle %s/%s failed: %s", owner, ds, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    log.info("=== MixCraft dataset downloader ===")
    log.info("Output dir: %s", DATA_RAW.resolve())

    # check requests
    try:
        import requests  # noqa: F811
        import pandas  # noqa: F811
        import pyarrow  # noqa: F401
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        log.error("Run: pip install requests pandas pyarrow")
        sys.exit(1)

    df_main = download_cocktaildb()
    df_cat = download_by_category()
    df_na = download_non_alcoholic()

    frames = [f for f in [df_main, df_cat, df_na] if not f.empty]
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"]).reset_index(drop=True)

    merged_out = DATA_RAW / "cocktails_merged.parquet"
    df_all.to_parquet(merged_out, index=False)
    df_all.to_csv(DATA_RAW / "cocktails_merged.csv", index=False)
    log.info("Merged -> cocktails_merged.parquet  (%d unique cocktails)", len(df_all))

    build_ingredient_index(df_all)
    try_kaggle_downloads()

    log.info("\n=== Fichiers generes ===")
    for f in sorted(DATA_RAW.iterdir()):
        if f.suffix in {".parquet", ".csv"} and f.stat().st_size > 100:
            log.info("  %-45s  %6.1f KB", f.name, f.stat().st_size / 1024)
