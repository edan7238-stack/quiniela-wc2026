"""Tests del Nivel 3 (Poisson + Dixon-Coles)."""
import numpy as np
import pandas as pd
import pytest

from src import poisson
from src.poisson import DixonColesParams


def _params(rho=0.0, home_adv=0.3):
    # 'Strong' ataca mucho y defiende bien; 'Weak' lo contrario.
    return DixonColesParams(
        attack={"Strong": 0.5, "Weak": -0.5},
        defense={"Strong": -0.4, "Weak": 0.4},
        intercept=0.2, home_adv=home_adv, rho=rho, max_goals=10,
        teams=["Strong", "Weak"],
    )


# --------------------------------------------------------------------------- #
# score_matrix
# --------------------------------------------------------------------------- #
def test_score_matrix_suma_uno():
    m = poisson.score_matrix(1.6, 1.1, rho=-0.03, max_goals=10)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_sin_rho_es_poisson_independiente():
    from scipy.stats import poisson as poi
    lam_h, lam_a, K = 1.5, 1.2, 10
    m = poisson.score_matrix(lam_h, lam_a, rho=0.0, max_goals=K)
    indep = np.outer(poi.pmf(np.arange(K + 1), lam_h), poi.pmf(np.arange(K + 1), lam_a))
    indep /= indep.sum()
    assert np.allclose(m, indep)


def test_rho_negativo_aumenta_empate():
    K = 10
    m0 = poisson.score_matrix(1.3, 1.2, rho=0.0, max_goals=K)
    mneg = poisson.score_matrix(1.3, 1.2, rho=-0.06, max_goals=K)
    draw0 = np.trace(m0)
    drawneg = np.trace(mneg)
    assert drawneg > draw0  # Dixon-Coles añade empates bajos


# --------------------------------------------------------------------------- #
# lambdas y localía
# --------------------------------------------------------------------------- #
def test_localia_sube_lambda_local():
    p = _params(home_adv=0.3)
    lam_h_loc, _ = poisson._lambdas(p, "Strong", "Weak", neutral=False)
    lam_h_neu, _ = poisson._lambdas(p, "Strong", "Weak", neutral=True)
    assert lam_h_loc > lam_h_neu


def test_equipo_desconocido_usa_promedio():
    p = _params()
    # 'Marte' no existe -> ataque/defensa 0 (equipo promedio), sin error.
    lam_h, lam_a = poisson._lambdas(p, "Marte", "Weak", neutral=True)
    assert lam_h > 0 and lam_a > 0


# --------------------------------------------------------------------------- #
# predict_match
# --------------------------------------------------------------------------- #
def test_predict_match_estructura_y_favorito():
    p = _params(rho=-0.03)
    r = poisson.predict_match(p, "Strong", "Weak", neutral=True)
    s = r["prob_1x2"]
    assert s["H"] + s["D"] + s["A"] == pytest.approx(1.0)
    assert s["H"] > s["A"]                       # el fuerte es favorito
    assert 0 <= r["over_2_5"] <= 1
    assert 0 <= r["btts_yes"] <= 1
    assert r["lambda_home"] > r["lambda_away"]


# --------------------------------------------------------------------------- #
# fit (integración ligera, sintética)
# --------------------------------------------------------------------------- #
def test_fit_recupera_jerarquia_de_ataque():
    rng = np.random.default_rng(0)
    rows = []
    dates = pd.date_range("2023-01-01", periods=60, freq="W")
    for d in dates:
        # A (fuerte) golea; B y C (débiles) marcan poco.
        rows.append((d, "A", "B", rng.poisson(3), rng.poisson(0), True))
        rows.append((d, "A", "C", rng.poisson(3), rng.poisson(1), True))
        rows.append((d, "B", "C", rng.poisson(1), rng.poisson(1), True))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "neutral"])
    params = poisson.fit(df, xi=0.0)  # sin decaimiento para el test
    assert set(params.teams) == {"A", "B", "C"}
    assert params.attack["A"] > params.attack["B"]
    assert params.attack["A"] > params.attack["C"]
    # A vs B: el local fuerte tiene mayor lambda.
    r = poisson.predict_match(params, "A", "B", neutral=True)
    assert r["lambda_home"] > r["lambda_away"]
