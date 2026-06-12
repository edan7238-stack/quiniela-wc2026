"""Entrena el modelo y genera los artefactos.

Por ahora cubre el Nivel 1 (Elo + ancla FIFA). El Nivel 2 (ML) se añadirá en su fase.

Uso:
    python scripts/train.py            # usa el snapshot FIFA existente
    python scripts/train.py --no-fifa  # solo Elo, sin ancla FIFA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # consola de Windows en cp1252 -> forzar UTF-8 para acentos y símbolos
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src import data_loader, elo, fifa, ml_model, poisson


def train_elo(use_fifa: bool = True):
    print("[1/3] Cargando histórico...")
    matches = data_loader.load_matches()  # TODO el histórico para el Elo
    print(f"      {len(matches):,} partidos, {len(data_loader.all_teams(matches))} selecciones")

    print("[2/3] Calculando Elo dinámico (recorrido cronológico)...")
    model = elo.EloModel()
    model.process(matches)

    fifa_points = {}
    if use_fifa:
        snap = fifa.load_snapshot()
        if snap is None:
            print("      [aviso] No hay snapshot FIFA. Ejecuta scripts/fetch_fifa.py "
                  "(se continúa solo con Elo).")
        else:
            fifa_points = fifa.points_map(snap)
            print(f"      Ancla FIFA: {len(fifa_points)} selecciones "
                  f"(edición {snap['ranking_date'].iloc[0]})")

    print("[3/3] Guardando ratings...")
    df = elo.save_ratings(model, fifa_points or None)
    print(f"      -> {settings.ELO_RATINGS_CSV}")
    print()
    print("Top 20 por fuerza (Elo mezclado con FIFA):")
    for i, row in df.head(20).iterrows():
        extra = f"  | FIFA-anchored {row['strength']:.0f}" if "strength" in df else ""
        print(f"  {i+1:>2}. {row['team']:<22} Elo {row['elo']:.0f}{extra}")
    return df


def train_ml():
    print()
    print("[ML] Entrenando clasificador 1X2 (Nivel 2)...")
    ml_model.train()
    print(f"      -> {settings.ML_MODEL_PKL}")


def train_poisson():
    print()
    print("[Poisson] Ajustando Dixon-Coles (Nivel 3, ventana reciente)...")
    recent = data_loader.load_matches(recent_only=True)
    params = poisson.fit(recent)
    poisson.save(params)
    print(f"      intercept={params.intercept:.3f}  home_adv={params.home_adv:.3f}  "
          f"rho={params.rho:.4f}  | {len(params.teams)} selecciones")
    print(f"      -> {settings.POISSON_PARAMS_PKL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Elo (+FIFA) y el clasificador ML.")
    parser.add_argument("--no-fifa", action="store_true", help="No usar el ancla FIFA")
    parser.add_argument("--solo-elo", action="store_true", help="Entrenar solo el Elo (sin ML)")
    args = parser.parse_args()
    train_elo(use_fifa=not args.no_fifa)
    if not args.solo_elo:
        train_ml()
        train_poisson()


if __name__ == "__main__":
    main()
