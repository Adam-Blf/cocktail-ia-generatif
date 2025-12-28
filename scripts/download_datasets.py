"""
Telechargement des datasets Kaggle necessaires au projet MixCraft.
Necessite un fichier ~/.kaggle/kaggle.json avec les credentials.
"""

import subprocess
import sys
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("aadyasingh55/cocktails", "cocktails"),
    ("pxxthik/the-cocktail-db-recipe-collection", "cocktail_db"),
    ("joakark/cocktail-ingredients-and-instructions", "cocktail_ingredients"),
    ("mexwell/iba-cocktails", "iba_cocktails"),
]


def download(dataset_ref: str, output_name: str) -> None:
    target = DATA_RAW / output_name
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {dataset_ref} deja telecharge.")
        return

    print(f"[download] {dataset_ref}...")
    target.mkdir(exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(target), "--unzip"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [error] {result.stderr.strip()}")
    else:
        print(f"  [ok] Sauvegarde dans {target}")


if __name__ == "__main__":
    print("=== Telechargement des datasets MixCraft ===\n")
    for ref, name in DATASETS:
        download(ref, name)
    print("\nTelechargemants termines. Verifier data/raw/")
