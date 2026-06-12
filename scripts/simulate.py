"""Simula el Mundial 2026 con Montecarlo y muestra los favoritos.

Uso:
    python scripts/simulate.py
    python scripts/simulate.py --sims 50000 --top 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings, wc2026
from src import montecarlo


def main() -> None:
    parser = argparse.ArgumentParser(description="Montecarlo del Mundial 2026")
    parser.add_argument("--sims", type=int, default=settings.N_SIMULACIONES)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Simulando el Mundial {args.sims:,} veces "
          f"({wc2026.N_GROUPS} grupos, {len(wc2026.all_participants())} selecciones)...")
    df = montecarlo.simulate(n_sims=args.sims, seed=args.seed)

    print()
    print(f"{'#':>3} {'Selección':<20}{'Grupo':>7}{'R32':>7}{'R16':>7}"
          f"{'QF':>7}{'SF':>7}{'Final':>8}{'Campeón':>9}")
    for i, row in df.head(args.top).iterrows():
        print(f"{i+1:>3} {row['team']:<20}{row['P_grupo']:>7.0%}{row['P_R32']:>7.0%}"
              f"{row['P_R16']:>7.0%}{row['P_QF']:>7.0%}{row['P_SF']:>7.0%}"
              f"{row['P_Final']:>8.1%}{row['P_Campeon']:>9.1%}")


if __name__ == "__main__":
    main()
