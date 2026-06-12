"""Hito G — Backtest de honestidad del optimizador de quiniela.

Mide, sobre la ventana de TEST (fuera de muestra del ML), qué tan seguido acierta el
modelo el **marcador exacto** y el **resultado (1X2)**, y cuántos puntos por partido
sacaría con la regla 3/1/0, comparando:
  - PICK ÓPTIMO  (maximiza 2·P(s)+P(o))
  - MARCADOR MODAL (el "más probable" a secas)

Sirve para fijar expectativas realistas: el exacto tiene techo (~12-18%); el 1X2 ~55%.

Uso:  python scripts/backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import settings
from src import features, ml_model, poisson, quiniela as q


def _lambdas(elo_diff: float, neutral: bool) -> tuple[float, float]:
    sup = settings.GOAL_SLOPE_PER_ELO * elo_diff
    if not neutral:
        sup += settings.GOAL_SLOPE_PER_ELO * settings.ELO_HOME_ADVANTAGE
    base = settings.GOAL_BASE_TOTAL
    lo = settings.GOAL_LAMBDA_MIN
    return max((base + sup) / 2, lo), max((base - sup) / 2, lo)


def main() -> None:
    bundle = ml_model.load_bundle()
    params = poisson.load()
    if bundle is None or params is None:
        print("Faltan artefactos. Ejecuta: python scripts/train.py", file=sys.stderr)
        sys.exit(1)
    rho, maxg = params.rho, params.max_goals

    frame = ml_model.build_training_frame()
    recent = frame[frame["year"] >= settings.CORTE_RECIENTE].sort_values("date").reset_index(drop=True)
    _, _, s_te = ml_model._temporal_split(len(recent), settings.VALIDACION_HOLDOUT_FRAC, 0.15)
    test = recent.iloc[s_te].reset_index(drop=True)

    X = features.assemble_features(test)
    proba = ml_model.predict_proba(bundle, X)  # columnas H/D/A

    n = len(test)
    hit_exact_opt = hit_exact_mod = hit_outcome = 0
    pts_opt = pts_mod = 0.0
    sc = q.DEFAULT

    for k in range(n):
        elo_diff = float(test["elo_diff"].iloc[k])
        neutral = bool(test["neutral"].iloc[k])
        lh, la = _lambdas(elo_diff, neutral)
        m = poisson.score_matrix(lh, la, rho, maxg)
        ml = {c: float(proba[c].iloc[k]) for c in ("H", "D", "A")}
        m = poisson.reconcile_with_1x2(m, ml)

        opt = q.best_scoreline(m, ml)["score"]
        mod = q.most_likely_scoreline(m)
        ah = int(min(test["home_score"].iloc[k], maxg))
        aa = int(min(test["away_score"].iloc[k], maxg))
        actual_outcome = q.outcome_of(ah, aa)

        # puntos 3/1/0
        def points(pick):
            if pick == (ah, aa):
                return sc.exacto
            return sc.resultado if q.outcome_of(*pick) == actual_outcome else sc.fallo

        if opt == (ah, aa):
            hit_exact_opt += 1
        if mod == (ah, aa):
            hit_exact_mod += 1
        if q.outcome_of(*opt) == actual_outcome:
            hit_outcome += 1
        pts_opt += points(opt)
        pts_mod += points(mod)

    print(f"Backtest fuera de muestra — {n} partidos "
          f"({test['date'].iloc[0].date()} a {test['date'].iloc[-1].date()})")
    print("-" * 60)
    print(f"  Acierto MARCADOR EXACTO (pick óptimo): {hit_exact_opt/n:.1%}")
    print(f"  Acierto MARCADOR EXACTO (modal):       {hit_exact_mod/n:.1%}")
    print(f"  Acierto RESULTADO 1X2 (pick óptimo):   {hit_outcome/n:.1%}")
    print("-" * 60)
    print(f"  Puntos/partido (3/1/0)  pick óptimo: {pts_opt/n:.3f}   total: {pts_opt:.0f}")
    print(f"  Puntos/partido (3/1/0)  modal:       {pts_mod/n:.3f}   total: {pts_mod:.0f}")
    print(f"  Mejora del optimizador sobre el modal: {(pts_opt-pts_mod)/n:+.3f} pts/partido")


if __name__ == "__main__":
    main()
