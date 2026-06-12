"""Descarga el Ranking Mundial FIFA y lo guarda en models/fifa_snapshot.csv.

Uso:
    python scripts/fetch_fifa.py
    python scripts/fetch_fifa.py --locale en

Selecciona automáticamente la edición más reciente con datos. Si la descarga
falla, se mantiene el snapshot previo (o el fallback manual de config).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src import fifa


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el ranking FIFA -> snapshot CSV")
    parser.add_argument("--locale", default="en", help="Idioma de los nombres (en/es)")
    args = parser.parse_args()

    try:
        df = fifa.refresh(locale=args.locale)
    except Exception as e:  # red caída, FIFA cambió el sitio, etc.
        print(f"[error] No se pudo descargar el ranking FIFA: {e}", file=sys.stderr)
        existing = fifa.load_snapshot()
        if existing is not None:
            print(f"[info] Se conserva el snapshot previo: {settings.FIFA_SNAPSHOT_CSV}")
        else:
            print("[info] No hay snapshot previo. Usa el fallback manual en config/wc2026.py")
        sys.exit(1)

    fecha = df["ranking_date"].iloc[0] if not df.empty else "?"
    print(f"[ok] Ranking FIFA del {fecha}: {len(df)} selecciones")
    print(f"     Guardado en {settings.FIFA_SNAPSHOT_CSV}")
    print("     Top 10:")
    for _, row in df.head(10).iterrows():
        print(f"       {int(row['rank']):>3}. {row['team']:<22} {row['points']:.2f}")


if __name__ == "__main__":
    main()
