"""Extrae el dataset Kaggle (archive.zip) a data/raw/.

Uso:
    python scripts/setup_data.py
    python scripts/setup_data.py --zip "C:/ruta/al/archive.zip"

El zip esperado contiene: results.csv, shootouts.csv, goalscorers.csv, former_names.csv
(dataset "International football results from 1872 to present").
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Permite ejecutar el script directamente (añade la raíz del proyecto al path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

EXPECTED = {"results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"}


def extract(zip_path: Path, dest: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"No se encontró el zip de Kaggle en: {zip_path}\n"
            "Descárgalo de "
            "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-present "
            "y pásalo con --zip <ruta>."
        )

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        faltan = EXPECTED - names
        if faltan:
            print(f"[aviso] El zip no contiene: {sorted(faltan)}", file=sys.stderr)
        zf.extractall(dest)

    extraidos = sorted(p.name for p in dest.glob("*.csv"))
    print(f"[ok] Extraído en {dest}")
    for name in extraidos:
        size = (dest / name).stat().st_size
        print(f"     - {name} ({size/1_000_000:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae el dataset Kaggle a data/raw/")
    parser.add_argument(
        "--zip",
        type=Path,
        default=settings.KAGGLE_ZIP,
        help=f"Ruta al archive.zip (por defecto: {settings.KAGGLE_ZIP})",
    )
    args = parser.parse_args()
    extract(args.zip, settings.RAW_DIR)


if __name__ == "__main__":
    main()
