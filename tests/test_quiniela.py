"""Tests del Hito C (optimizador de quiniela)."""
import numpy as np
import pytest

from src import quiniela as q


def test_outcome_of():
    assert q.outcome_of(2, 0) == "H"
    assert q.outcome_of(1, 1) == "D"
    assert q.outcome_of(0, 2) == "A"


def test_expected_points_es_2ps_mas_po():
    # Para 3/1/0, E[pts(s)] = 2·P(s) + P(o).
    m = np.zeros((4, 4)); m[2, 0] = 0.1
    po = {"H": 0.5, "D": 0.3, "A": 0.2}
    ep = q.expected_points(m, 2, 0, po, q.DEFAULT)
    assert ep == pytest.approx(2 * 0.1 + 0.5)


def test_expected_points_allin():
    # all-in: E = 7·P(s) + 11·P(o) − 6.
    m = np.zeros((4, 4)); m[1, 0] = 0.12
    po = {"H": 0.55, "D": 0.25, "A": 0.20}
    ep = q.expected_points(m, 1, 0, po, q.ALLIN)
    assert ep == pytest.approx(7 * 0.12 + 11 * 0.55 - 6)


def test_best_scoreline_no_es_siempre_el_modal():
    # Matriz donde el modal es un empate 1-1 pero el resultado dominante es el local.
    m = np.zeros((3, 3))
    m[1, 1] = 0.12                       # modal (empate)
    m[1, 0] = 0.11; m[2, 0] = 0.10; m[2, 1] = 0.10; m[0, 0] = 0.04
    m[0, 1] = 0.02; m[0, 2] = 0.01; m[1, 2] = 0.02
    m = m / m.sum()
    best = q.best_scoreline(m)
    assert q.most_likely_scoreline(m) == (1, 1)        # el modal es empate
    assert best["outcome"] == "H"                       # pero el óptimo es victoria local
    # el óptimo nunca rinde menos que apostar al modal
    po = q.poisson.outcome_probs(m)
    ep_modal = q.expected_points(m, 1, 1, po)
    assert best["ep"] >= ep_modal


def test_best_scoreline_coincide_con_fuerza_bruta():
    rng = np.random.default_rng(0)
    m = rng.random((5, 5)); m /= m.sum()
    po = q.poisson.outcome_probs(m)
    brute = max(((i, j) for i in range(5) for j in range(5)),
                key=lambda s: q.expected_points(m, s[0], s[1], po))
    assert q.best_scoreline(m)["score"] == brute


def test_group_fixtures_calendario_oficial_ordenado():
    """Con los grupos oficiales: 72 fixtures con `date`, en ORDEN CRONOLOGICO REAL del Mundial."""
    fx = q.group_fixtures()
    assert len(fx) == 72
    assert all("date" in f for f in fx)
    # Orden cronológico (no decreciente).
    dates = [f["date"] for f in fx]
    assert dates == sorted(dates)
    # Empieza el 11/06 y acaba el 27/06.
    assert dates[0] == "2026-06-11" and dates[-1] == "2026-06-27"
    # El anfitrión del primer partido va como local (México).
    assert fx[0]["home"] == "Mexico" and fx[0]["neutral"] is False


def test_group_fixtures_fallback_sin_fecha_para_grupos_custom():
    custom = {"X": ["Spain", "Brazil", "Germany", "France"]}
    fx = q.group_fixtures(custom)
    assert len(fx) == 6
    assert all("date" not in f for f in fx)  # cae al fallback (sin calendario)


def test_allocate_comodines_respeta_cupos_y_prioriza():
    evals = [
        {"normal": {"ep": 1.0}, "allin": {"ep": 0.0}},
        {"normal": {"ep": 0.9}, "allin": {"ep": 0.0}},
        {"normal": {"ep": 0.8}, "allin": {"ep": 0.0}},
        {"normal": {"ep": 0.7}, "allin": {"ep": 0.0}},
        {"normal": {"ep": 0.6}, "allin": {"ep": 5.0}},  # all-in brillante aquí
    ]
    plan = q.allocate_comodines(evals)
    a = plan["assignments"]
    # cada partido a lo sumo un comodín; tipos dentro de cupo
    assert len(set(a.keys())) == len(a)
    assert sum(1 for v in a.values() if v == "triple") <= 2
    assert sum(1 for v in a.values() if v == "double") <= 2
    assert sum(1 for v in a.values() if v == "allin") <= 2
    # el all-in va al partido 4; los triples a los de mayor EP normal (0 y 1)
    assert a[4] == "allin"
    assert a[0] == "triple" and a[1] == "triple"
    # el total mejora sobre la base
    assert plan["total_ep"] > plan["baseline_ep"]


def _fake_mc_df():
    import pandas as pd
    # 4 equipos de un grupo "A" con marginales claras + placements.
    return pd.DataFrame({
        "team": ["A", "B", "C", "D"],
        "P_g1": [0.7, 0.2, 0.08, 0.02], "P_g2": [0.2, 0.5, 0.25, 0.05],
        "P_g3": [0.08, 0.25, 0.42, 0.25], "P_g4": [0.02, 0.05, 0.25, 0.68],
        "P_best_third": [0.01, 0.10, 0.30, 0.20],
        "P_Campeon": [0.4, 0.1, 0.05, 0.0],
        "P_subcampeon": [0.2, 0.3, 0.1, 0.02],
        "P_3er_puesto": [0.1, 0.2, 0.2, 0.05],
        "P_4to_puesto": [0.1, 0.1, 0.2, 0.1],
    })


def test_group_position_picks():
    df = _fake_mc_df()
    picks = q.group_position_picks(df, groups={"A": ["A", "B", "C", "D"]})
    row = picks.iloc[0]
    assert row["1º"] == "A" and row["2º"] == "B"     # asignación óptima por marginal
    assert row["E_pts"] > 0


def test_placement_picks_campeon_es_el_mas_probable():
    df = _fake_mc_df()
    picks = q.placement_picks(df)
    campeon = picks[picks["puesto"] == "Campeón"].iloc[0]
    assert campeon["pick"] == "A"                     # A tiene la mayor P_Campeon
    # equipos distintos en cada puesto
    assert picks["pick"].nunique() == len(picks)


def test_best_thirds_picks_toma_los_top():
    df = _fake_mc_df()
    top, exp = q.best_thirds_picks(df, n=2)
    assert list(top["team"]) == ["C", "D"]            # mayores P_best_third (0.30, 0.20)
    assert exp == pytest.approx((0.30 + 0.20) * 3)


def test_group_fixtures_72_partidos_y_anfitriones():
    fx = q.group_fixtures()
    assert len(fx) == 72                       # 12 grupos × 6
    # En el grupo de México (A), sus 3 partidos son como local (no neutral).
    mex = [f for f in fx if "Mexico" in (f["home"], f["away"])]
    assert len(mex) == 3
    assert all(f["home"] == "Mexico" and not f["neutral"] for f in mex)
