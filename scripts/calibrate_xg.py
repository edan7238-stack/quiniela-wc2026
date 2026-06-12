"""Calibración de `XG_BLEND_W` (y chequeo de `XG_DECAY_PER_DAY`) por backtest point-in-time.

Mide, sobre la MISMA ventana de test fuera de muestra que `scripts/backtest.py`, cuántos
puntos/partido (regla 3/1/0, pick óptimo) saca el modelo de marcador para distintos pesos
`w` de la mezcla xG↔fuerza. Clave metodológica: para cada partido de test, el xG de cada
selección se agrega **solo con sus partidos anteriores a la fecha del encuentro** (decaimiento
relativo a esa fecha) — sin fuga de futuro. El elo_diff ya es point-in-time.

Como la matriz se **reconcilia con el 1X2 del ML** (que no depende de `w`), el 1X2 es
prácticamente invariante en `w`: la mezcla xG solo reordena el MARCADOR EXACTO dentro de cada
resultado. Por eso el efecto se concentra en el subconjunto de partidos donde AMBAS selecciones
tienen historial xG suficiente; ahí es donde se decide el `w` óptimo.

Uso:  python scripts/calibrate_xg.py
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
from src import features, ml_model, poisson, team_ratings, quiniela as q

W_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)
DECAY_GRID = [0.0006, 0.0012, 0.0024]   # semividas ~3.2 a / 1.6 a / 0.8 a


def _strength_lambdas(elo_diff: float, neutral: bool) -> tuple[float, float]:
    """λ por fuerza pura (idéntico a backtest._lambdas)."""
    sup = settings.GOAL_SLOPE_PER_ELO * elo_diff
    if not neutral:
        sup += settings.GOAL_SLOPE_PER_ELO * settings.ELO_HOME_ADVANTAGE
    base = settings.GOAL_BASE_TOTAL
    lo = settings.GOAL_LAMBDA_MIN
    return max((base + sup) / 2, lo), max((base - sup) / 2, lo)


class PITxG:
    """Agregador point-in-time del xG por selección (solo partidos anteriores a una fecha)."""

    def __init__(self, long_df):
        self.by_team: dict[str, tuple] = {}
        for team, g in long_df.groupby("team"):
            g = g.sort_values("date")
            self.by_team[team] = (
                g["date"].values.astype("datetime64[D]"),
                g["xgf"].to_numpy(dtype=float),
                g["xga"].to_numpy(dtype=float),
            )

    def prior_count(self, team: str, as_of) -> int:
        arr = self.by_team.get(team)
        if arr is None:
            return 0
        return int((arr[0] < as_of).sum())

    def weighted(self, team: str, as_of, decay: float):
        """(xg_for, xg_against) ponderado por decaimiento con partidos < as_of. None si nada."""
        arr = self.by_team.get(team)
        if arr is None:
            return None
        dates, xgf, xga = arr
        mask = dates < as_of
        if not mask.any():
            return None
        age = (as_of - dates[mask]).astype("timedelta64[D]").astype(float)
        w = np.exp(-decay * age)
        sw = w.sum()
        return float((xgf[mask] * w).sum() / sw), float((xga[mask] * w).sum() / sw)


def main() -> None:
    bundle = ml_model.load_bundle()
    params = poisson.load()
    if bundle is None or params is None:
        print("Faltan artefactos. Ejecuta: python scripts/train.py", file=sys.stderr)
        sys.exit(1)
    rho, maxg = params.rho, params.max_goals
    sc = q.DEFAULT

    long = team_ratings.national_xg_long()
    if long is None or long.empty:
        print("No hay datos xG por-partido en data/xg.csv. Ejecuta: python scripts/setup_xg.py",
              file=sys.stderr)
        sys.exit(1)
    pit = PITxG(long)
    avg = team_ratings.load().league_avg_xg()   # mismo normalizador que el modelo desplegado
    lo = settings.GOAL_LAMBDA_MIN
    min_n = settings.XG_MIN_MATCHES

    # Ventana de test (idéntica a backtest.py).
    frame = ml_model.build_training_frame()
    recent = frame[frame["year"] >= settings.CORTE_RECIENTE].sort_values("date").reset_index(drop=True)
    _, _, s_te = ml_model._temporal_split(len(recent), settings.VALIDACION_HOLDOUT_FRAC, 0.15)
    test = recent.iloc[s_te].reset_index(drop=True)
    X = features.assemble_features(test)
    proba = ml_model.predict_proba(bundle, X)
    n = len(test)

    def points(pick, ah, aa, outcome):
        if pick == (ah, aa):
            return sc.exacto
        return sc.resultado if q.outcome_of(*pick) == outcome else sc.fallo

    # ---- Paso 1: puntos por fuerza pura (w=0) para TODOS; marca el subconjunto con xG. ----
    base_pts = np.zeros(n)
    base_exact = np.zeros(n, dtype=bool)
    applicable = []   # índices con xG suficiente en AMBAS selecciones (point-in-time)
    cache = {}        # idx -> (ml, lh0, la0, ah, aa, outcome, home, away, as_of)
    for k in range(n):
        home = str(test["home_team"].iloc[k])
        away = str(test["away_team"].iloc[k])
        as_of = np.datetime64(test["date"].iloc[k], "D")
        elo_diff = float(test["elo_diff"].iloc[k])
        neutral = bool(test["neutral"].iloc[k])
        ml = {c: float(proba[c].iloc[k]) for c in ("H", "D", "A")}
        lh0, la0 = _strength_lambdas(elo_diff, neutral)
        m = poisson.reconcile_with_1x2(poisson.score_matrix(lh0, la0, rho, maxg), ml)
        ah = int(min(test["home_score"].iloc[k], maxg))
        aa = int(min(test["away_score"].iloc[k], maxg))
        outcome = q.outcome_of(ah, aa)
        pick = q.best_scoreline(m, ml)["score"]
        base_pts[k] = points(pick, ah, aa, outcome)
        base_exact[k] = (pick == (ah, aa))
        if pit.prior_count(home, as_of) >= min_n and pit.prior_count(away, as_of) >= min_n:
            applicable.append(k)
            cache[k] = (ml, lh0, la0, ah, aa, outcome, home, away, as_of)

    n_app = len(applicable)
    print(f"Calibración xG — ventana test {test['date'].iloc[0].date()} a {test['date'].iloc[-1].date()}")
    print(f"  Partidos de test: {n}   |   con xG point-in-time en ambas selecciones: {n_app}")
    print(f"  Normalizador (league_avg xG): {avg:.3f}   |   XG_MIN_MATCHES={min_n}")
    if n_app < 20:
        print("  [AVISO] Muy pocos partidos aplicables; la calibración es orientativa.")
    base_app_pts = sum(base_pts[k] for k in applicable)
    base_app_exact = sum(int(base_exact[k]) for k in applicable)
    print("-" * 74)
    print(f"  Baseline subconjunto (w=0, fuerza pura): exacto {base_app_exact}/{n_app} "
          f"({base_app_exact/n_app:.1%})  ·  {base_app_pts/n_app:.3f} pts/partido")

    def sweep(decay: float):
        """Devuelve dict w -> (exact_app, pts_app, pts_overall)."""
        # Precalcula el xG ponderado point-in-time de cada equipo aplicable para ESTE decay.
        out = {}
        wx = {}
        for k in applicable:
            _, _, _, _, _, _, home, away, as_of = cache[k]
            wx[k] = (pit.weighted(home, as_of, decay), pit.weighted(away, as_of, decay))
        nonapp_pts = float(sum(base_pts[k] for k in range(n) if k not in cache))
        for w in W_GRID:
            pts_app = 0.0
            exact_app = 0
            for k in applicable:
                ml, lh0, la0, ah, aa, outcome, *_ = cache[k]
                (xgf_h, xga_h), (xgf_a, xga_a) = wx[k]
                lh_xg = xgf_h * xga_a / avg
                la_xg = xgf_a * xga_h / avg
                lh = max((1 - w) * lh0 + w * lh_xg, lo)
                la = max((1 - w) * la0 + w * la_xg, lo)
                m = poisson.reconcile_with_1x2(poisson.score_matrix(lh, la, rho, maxg), ml)
                pick = q.best_scoreline(m, ml)["score"]
                p = points(pick, ah, aa, outcome)
                pts_app += p
                exact_app += int(pick == (ah, aa))
            out[float(w)] = (exact_app, pts_app, (nonapp_pts + pts_app) / n)
        return out

    # ---- Tabla principal: decay por defecto ----
    default_decay = settings.XG_DECAY_PER_DAY
    print("-" * 74)
    print(f"BARRIDO de w  (decay={default_decay}, semivida ~{np.log(2)/default_decay/365:.1f} años)")
    print(f"  {'w':>4}   {'exacto subset':>14}   {'pts/part subset':>16}   {'pts/part total':>15}")
    res = sweep(default_decay)
    best_w, best_val = None, -1.0
    for w in W_GRID:
        ex, pa, ov = res[float(w)]
        flag = ""
        if pa / n_app > best_val + 1e-9:
            best_val, best_w = pa / n_app, float(w)
        print(f"  {w:>4.1f}   {ex:>4d}/{n_app:<4d} {ex/n_app:>6.1%}   {pa/n_app:>10.3f}      {ov:>13.3f}")
    print(f"  -> mejor w (por pts/part subset) = {best_w} con {best_val:.3f} pts/partido "
          f"(baseline {base_app_pts/n_app:.3f})")

    # ---- Chequeo de decaimiento: mejor w por cada decay ----
    print("-" * 74)
    print("CHEQUEO de decaimiento (mejor w por cada valor):")
    print(f"  {'decay':>8} {'semivida':>10}   {'mejor w':>8}   {'pts/part subset':>16}")
    grid_best = (None, None, -1.0)
    for dec in DECAY_GRID:
        r = sweep(dec)
        bw, bv = max(((float(w), r[float(w)][1] / n_app) for w in W_GRID), key=lambda t: t[1])
        if bv > grid_best[2]:
            grid_best = (dec, bw, bv)
        print(f"  {dec:>8} {np.log(2)/dec/365:>8.1f} a   {bw:>8.1f}   {bv:>14.3f}")
    print(f"  -> mejor (decay, w) global = ({grid_best[0]}, {grid_best[1]}) "
          f"con {grid_best[2]:.3f} pts/partido en el subconjunto")
    print("-" * 74)
    print("Nota: el 1X2 se reconcilia con el ML, así que la ganancia es en MARCADOR EXACTO. "
          "Fija XG_BLEND_W (y, si procede, XG_DECAY_PER_DAY) en config/settings.py con el óptimo.")


if __name__ == "__main__":
    main()
