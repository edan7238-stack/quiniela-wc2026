"""Nivel 3 — Poisson bivariado con ajuste de Dixon-Coles (marcador exacto).

Modelo de goles:
    λ_local     = exp(intercept + ataque[local]  + defensa[visitante] + γ·localía)
    λ_visitante = exp(intercept + ataque[visit.] + defensa[local])

Los goles de cada equipo siguen una Poisson, pero la **corrección de Dixon-Coles**
ajusta la interdependencia en los marcadores bajos (0-0, 1-0, 0-1, 1-1), corrigiendo
el exceso de empates que la Poisson independiente subestima:

    P(x,y) = τ(x,y; λ, μ, ρ) · Poisson(x;λ) · Poisson(y;μ)

Estimación (eficiente y estable):
1. Ataque/defensa + γ con una **regresión de Poisson regularizada** (sklearn
   `PoissonRegressor`) sobre el histórico en formato largo (2 filas por partido),
   con **pesos de decaimiento temporal** exp(-ξ·días) que priorizan lo reciente.
2. **ρ** de Dixon-Coles por MLE 1-D sobre los marcadores de esquina.

La salida es una matriz de probabilidades de marcador, de la que se derivan el
marcador más probable, el 1X2 (cross-check del ML) y mercados (Over/Under, BTTS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as _poisson

from config import settings


@dataclass
class DixonColesParams:
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_adv: float
    rho: float
    max_goals: int = settings.POISSON_MAX_GOALS
    fit_date: str = ""
    teams: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #
def _time_weights(dates, ref_date, xi: float) -> np.ndarray:
    age_days = (ref_date - dates).dt.days.to_numpy().astype(float)
    return np.exp(-xi * np.clip(age_days, 0, None))


def fit(matches, xi: float = settings.DIXON_COLES_XI, alpha: float = 0.01,
        max_goals: int = settings.POISSON_MAX_GOALS) -> DixonColesParams:
    """Estima los parámetros Dixon-Coles a partir de un histórico de partidos.

    `matches` debe tener: date, home_team, away_team, home_score, away_score, neutral.
    Se recomienda pasar la ventana reciente (la del ML) para reflejar el nivel actual.
    `xi` controla el decaimiento temporal; `alpha` la regularización L2 del GLM.
    """
    from sklearn.linear_model import PoissonRegressor

    df = matches.reset_index(drop=True)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)

    ref_date = df["date"].max()
    w_match = _time_weights(df["date"], ref_date, xi)

    home = df["home_team"].map(idx).to_numpy()
    away = df["away_team"].map(idx).to_numpy()
    hs = df["home_score"].to_numpy().astype(float)
    as_ = df["away_score"].to_numpy().astype(float)
    neutral = df["neutral"].astype(bool).to_numpy()
    M = len(df)

    # Formato largo: 2 filas por partido (cada equipo "anotando").
    # Columnas: [ataque (T dummies), defensa (T dummies), indicador_localía]
    X = np.zeros((2 * M, 2 * T + 1), dtype=np.float32)
    y = np.empty(2 * M, dtype=np.float64)
    sw = np.empty(2 * M, dtype=np.float64)

    r = np.arange(M)
    # Fila A: el LOCAL anota
    X[r, home] = 1.0                 # ataque local
    X[r, T + away] = 1.0             # defensa visitante
    X[r, 2 * T] = np.where(neutral, 0.0, 1.0)  # localía solo si no es neutral
    y[r] = hs
    sw[r] = w_match
    # Fila B: el VISITANTE anota
    X[M + r, away] = 1.0             # ataque visitante
    X[M + r, T + home] = 1.0         # defensa local
    X[M + r, 2 * T] = 0.0            # el visitante nunca tiene localía
    y[M + r] = as_
    sw[M + r] = w_match

    glm = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=500)
    glm.fit(X, y, sample_weight=sw)

    coef = glm.coef_
    attack = {t: float(coef[idx[t]]) for t in teams}
    defense = {t: float(coef[T + idx[t]]) for t in teams}
    home_adv = float(coef[2 * T])
    intercept = float(glm.intercept_)

    # λ por partido (con localía real) para estimar ρ
    lam_h = np.exp(intercept + coef[home] + coef[T + away]
                   + np.where(neutral, 0.0, home_adv))
    lam_a = np.exp(intercept + coef[away] + coef[T + home])
    rho = _fit_rho(lam_h, lam_a, hs, as_, w_match)

    return DixonColesParams(
        attack=attack, defense=defense, intercept=intercept, home_adv=home_adv,
        rho=rho, max_goals=max_goals, fit_date=str(ref_date.date()), teams=teams,
    )


def _fit_rho(lam_h, lam_a, hs, as_, w) -> float:
    """MLE 1-D del parámetro de dependencia ρ sobre los marcadores de esquina."""
    corner = (hs <= 1) & (as_ <= 1)
    lh, la = lam_h[corner], lam_a[corner]
    x, y, ww = hs[corner], as_[corner], w[corner]
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)

    def neg_ll(rho: float) -> float:
        tau = np.ones_like(lh)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-9, None)
        return -np.sum(ww * np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


# --------------------------------------------------------------------------- #
# Predicción de marcador
# --------------------------------------------------------------------------- #
def _lambdas(params: DixonColesParams, home: str, away: str, neutral: bool) -> tuple[float, float]:
    ah, dh = params.attack.get(home, 0.0), params.defense.get(home, 0.0)
    aa, da = params.attack.get(away, 0.0), params.defense.get(away, 0.0)
    gamma = 0.0 if neutral else params.home_adv
    lam_h = np.exp(params.intercept + ah + da + gamma)
    lam_a = np.exp(params.intercept + aa + dh)
    return float(lam_h), float(lam_a)


def score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int) -> np.ndarray:
    """Matriz (max_goals+1)x(max_goals+1) de P(local=x, visitante=y) con Dixon-Coles."""
    gh = _poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = _poisson.pmf(np.arange(max_goals + 1), lam_a)
    m = np.outer(gh, ga)
    # Corrección de Dixon-Coles en las esquinas (x=local, y=visitante)
    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def outcome_probs(m: np.ndarray) -> dict[str, float]:
    """Probabilidades 1X2 a partir de una matriz de marcadores."""
    return {"H": float(np.tril(m, -1).sum()),   # local > visitante
            "D": float(np.trace(m)),             # empate
            "A": float(np.triu(m, 1).sum())}     # visitante > local


def reconcile_with_1x2(m: np.ndarray, p1x2: dict[str, float]) -> np.ndarray:
    """Reescala la matriz para que sus marginales 1X2 igualen `p1x2` (p. ej. el del ML),
    conservando la forma de los marcadores. Devuelve una matriz nueva normalizada."""
    cur = outcome_probs(m)
    out = m.copy()
    n = m.shape[0]
    iu = np.triu_indices(n, 1)      # visitante gana (x < y)
    il = np.tril_indices(n, -1)     # local gana (x > y)
    di = np.diag_indices(n)         # empate
    for idx, k in ((il, "H"), (di, "D"), (iu, "A")):
        if cur[k] > 1e-12:
            out[idx] *= p1x2[k] / cur[k]
    s = out.sum()
    return out / s if s > 0 else out


def markets_from_matrix(m: np.ndarray, top_n: int = 5) -> dict:
    """Deriva mercados y marcadores de una matriz: 1X2, marcador modal, top-N,
    Over/Under 2.5 y BTTS."""
    flat = [((i, j), float(m[i, j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
    flat.sort(key=lambda t: t[1], reverse=True)
    total = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    over25 = float(m[total >= 3].sum())
    btts = float(m[1:, 1:].sum())
    return {
        "prob_1x2": outcome_probs(m),
        "most_likely_score": flat[0][0],
        "top_scores": flat[:top_n],
        "over_2_5": over25, "under_2_5": 1.0 - over25,
        "btts_yes": btts, "btts_no": 1.0 - btts,
    }


def predict_match(params: DixonColesParams, home: str, away: str,
                  neutral: bool = True) -> dict:
    """Predice el marcador y mercados con el modelo GLM Dixon-Coles (cross-check)."""
    lam_h, lam_a = _lambdas(params, home, away, neutral)
    m = score_matrix(lam_h, lam_a, params.rho, params.max_goals)
    out = {"home": home, "away": away, "neutral": neutral,
           "lambda_home": lam_h, "lambda_away": lam_a, "score_matrix": m}
    out.update(markets_from_matrix(m))
    return out


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def save(params: DixonColesParams, path=settings.POISSON_PARAMS_PKL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(params, path)


def load(path=settings.POISSON_PARAMS_PKL) -> DixonColesParams | None:
    if not path.exists():
        return None
    return joblib.load(path)
