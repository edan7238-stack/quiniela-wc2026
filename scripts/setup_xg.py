"""Prepara `data/xg.csv` a partir del CSV de partidos con xG que aporta el usuario.

El archivo de origen (por defecto `~/Downloads/xG.csv`) trae una fila por partido de
muchas ligas (clubes y selecciones) con ~100 columnas. Aquí lo adelgazamos a las pocas
columnas que el cargador necesita y lo dejamos en `data/xg.csv`. El FILTRADO a selecciones
+ recencia + decaimiento lo hace `src/team_ratings.py` al cargar (no aquí), de modo que el
archivo guardado sigue siendo el dataset por-partido genérico.

Uso:
    python scripts/setup_xg.py                      # usa ~/Downloads/xG.csv
    python scripts/setup_xg.py ruta\\a\\otro.csv     # fuente alternativa
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from src import data_loader, team_ratings  # noqa: E402

# Columnas que conservamos (las que el cargador usa + contexto útil). Si alguna no está
# en el origen, simplemente se omite.
_KEEP = ["league_division", "date", "home_team", "away_team",
         "home_goals", "away_goals", "home_xg", "away_xg"]


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else Path.home() / "Downloads" / "xG.csv"
    if not src.exists():
        print(f"[ERROR] No existe el archivo de origen: {src}")
        print("Pásalo como argumento: python scripts/setup_xg.py <ruta.csv>")
        return 1

    raw = pd.read_csv(src, low_memory=False)
    keep = [c for c in _KEEP if c in raw.columns]
    if not {"date", "home_team", "away_team", "home_xg", "away_xg"}.issubset(keep):
        print(f"[ERROR] El origen no parece un CSV por-partido con xG. Columnas: {list(raw.columns)[:20]}…")
        return 1

    slim = raw[keep].copy()
    settings.XG_CSV.parent.mkdir(parents=True, exist_ok=True)
    slim.to_csv(settings.XG_CSV, index=False)
    print(f"[OK] {len(slim)} partidos -> {settings.XG_CSV} "
          f"({settings.XG_CSV.stat().st_size/1e6:.2f} MB, {len(keep)} columnas)")

    # Resumen de lo que el cargador extraerá realmente (selecciones + decaimiento).
    ratings = team_ratings.load()
    teams = ratings.teams_with_xg()
    print(f"[INFO] Modo detectado: {ratings.source}")
    print(f"[INFO] Selecciones con xG (>= {settings.XG_MIN_MATCHES} partidos): {len(teams)}")
    print(f"[INFO] Media liga (xG/partido): {ratings.league_avg_xg():.2f}")
    # Cobertura de participantes del Mundial.
    try:
        from config import wc2026
        parts = set(wc2026.all_participants())
        cov = [t for t in teams if t in parts]
        faltan = sorted(parts - set(teams))
        print(f"[INFO] Participantes WC2026 con xG: {len(cov)}/{len(parts)}")
        print(f"[INFO] Sin xG (fuerza pura): {', '.join(faltan)}")
    except Exception as e:  # pragma: no cover
        print(f"[WARN] No pude calcular cobertura WC2026: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
