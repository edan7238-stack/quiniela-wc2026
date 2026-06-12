"""Tests del Nivel 1 (Elo)."""
import numpy as np
import pytest

from src import elo


# --------------------------------------------------------------------------- #
# expected_score
# --------------------------------------------------------------------------- #
def test_expected_score_neutral_equal_es_50():
    assert elo.expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_ventaja_localia_sube_la_esperanza():
    # Con ratings iguales pero en casa, la esperanza del local supera 0.5.
    assert elo.expected_score(1500, 1500, neutral=False, home_advantage=65) > 0.5


def test_expected_score_es_monotona():
    base = elo.expected_score(1600, 1500, neutral=True)
    mas_fuerte = elo.expected_score(1700, 1500, neutral=True)
    assert mas_fuerte > base


def test_expected_score_simetria():
    # We(A vs B) + We(B vs A) = 1 en campo neutral.
    a = elo.expected_score(1600, 1480, neutral=True)
    b = elo.expected_score(1480, 1600, neutral=True)
    assert a + b == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# goal_multiplier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gd,esperado", [(0, 1.0), (1, 1.0), (-1, 1.0),
                                         (2, 1.5), (-2, 1.5),
                                         (3, 1.75), (4, 1.875), (5, 2.0)])
def test_goal_multiplier(gd, esperado):
    assert elo.goal_multiplier(gd) == pytest.approx(esperado)


# --------------------------------------------------------------------------- #
# Actualización
# --------------------------------------------------------------------------- #
def test_update_es_suma_cero():
    m = elo.EloModel()
    suma_antes = m.rating("A") + m.rating("B")
    m.update_one("A", "B", 2, 0, importance="friendly", neutral=True)
    suma_despues = m.ratings["A"] + m.ratings["B"]
    assert suma_despues == pytest.approx(suma_antes)


def test_ganar_sube_y_perder_baja():
    m = elo.EloModel()
    m.update_one("A", "B", 1, 0, importance="friendly", neutral=True)
    assert m.ratings["A"] > 1500 > m.ratings["B"]


def test_sorpresa_mueve_mas_que_lo_esperado():
    # Un débil que gana a un fuerte gana más puntos que un fuerte que gana a un débil.
    m1 = elo.EloModel(); m1.ratings.update({"fuerte": 1900, "debil": 1300})
    pre = m1.ratings["debil"]
    m1.update_one("debil", "fuerte", 1, 0, importance="friendly", neutral=True)
    ganancia_sorpresa = m1.ratings["debil"] - pre

    m2 = elo.EloModel(); m2.ratings.update({"fuerte": 1900, "debil": 1300})
    pre2 = m2.ratings["fuerte"]
    m2.update_one("fuerte", "debil", 1, 0, importance="friendly", neutral=True)
    ganancia_esperada = m2.ratings["fuerte"] - pre2

    assert ganancia_sorpresa > ganancia_esperada


def test_mayor_importancia_mueve_mas():
    m1 = elo.EloModel(); pre1 = m1.rating("A")
    m1.update_one("A", "B", 1, 0, importance="world_cup", neutral=True)
    d_wc = m1.ratings["A"] - pre1

    m2 = elo.EloModel(); pre2 = m2.rating("A")
    m2.update_one("A", "B", 1, 0, importance="friendly", neutral=True)
    d_friendly = m2.ratings["A"] - pre2

    assert d_wc > d_friendly


# --------------------------------------------------------------------------- #
# process (sin fuga: el Elo es PREVIO al partido)
# --------------------------------------------------------------------------- #
def test_process_elo_previo_sin_fuga():
    import pandas as pd
    matches = pd.DataFrame({
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_score": [3, 0],
        "away_score": [0, 0],
        "importance": ["friendly", "friendly"],
        "neutral": [True, True],
    })
    m = elo.EloModel()
    out = m.process(matches)
    # El primer partido se valora con el rating inicial (sin conocer el resultado).
    assert out["elo_home"].iloc[0] == pytest.approx(1500)
    # Tras la goleada de A, el segundo partido ya parte con A por encima de B.
    assert out["elo_diff"].iloc[1] > 0
    assert not out[["elo_home", "elo_away", "elo_diff"]].isna().any().any()


# --------------------------------------------------------------------------- #
# Mezcla con FIFA
# --------------------------------------------------------------------------- #
def test_blend_respeta_el_orden():
    elo_r = {f"T{i}": 1500 + i * 50 for i in range(10)}
    fifa = {f"T{i}": 1000 + i * 40 for i in range(10)}
    blended = elo.blend_with_fifa(elo_r, fifa, w=0.7)
    orden = [t for t, _ in sorted(blended.items(), key=lambda kv: kv[1], reverse=True)]
    assert orden[0] == "T9" and orden[-1] == "T0"


def test_blend_sin_fifa_no_cambia():
    elo_r = {"A": 1600, "B": 1500}
    assert elo.blend_with_fifa(elo_r, {}, w=0.7) == elo_r
