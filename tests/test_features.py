"""Tests del Nivel 2 (features / forma)."""
import numpy as np
import pandas as pd
import pytest

from src import features


def _frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "home_team": ["A", "A", "B"],
        "away_team": ["B", "B", "A"],
        "home_score": [2, 1, 0],
        "away_score": [0, 1, 3],
    })


def test_forma_sin_fuga_primer_partido_es_nan():
    out = features.FormTracker().process(_frame())
    # El primer partido no tiene historia previa -> forma NaN.
    assert np.isnan(out["home_gf"].iloc[0])
    assert np.isnan(out["away_gf"].iloc[0])


def test_forma_refleja_partido_anterior():
    out = features.FormTracker().process(_frame())
    # En el 2º partido, A llega con la goleada 2-0 previa.
    assert out["home_gf"].iloc[1] == pytest.approx(2.0)
    assert out["home_ga"].iloc[1] == pytest.approx(0.0)
    assert out["home_ppg"].iloc[1] == pytest.approx(3.0)
    assert out["home_wr"].iloc[1] == pytest.approx(1.0)
    # B llega con 0-2 (derrota).
    assert out["away_ppg"].iloc[1] == pytest.approx(0.0)


def test_dias_de_descanso():
    out = features.FormTracker().process(_frame())
    assert out["home_rest"].iloc[1] == pytest.approx(31.0)  # 01-ene -> 01-feb


def test_current_form_tras_procesar():
    ft = features.FormTracker()
    ft.process(_frame())
    f = ft.current_form("A")
    assert f["ppg"] >= 0 and not np.isnan(f["gf"])


def test_assemble_features_orden_y_columnas():
    df = pd.DataFrame({
        "elo_home": [1600.0], "elo_away": [1500.0], "elo_diff": [100.0],
        "neutral": [False], "k": [60.0],
        "home_gf": [2.0], "home_ga": [0.5], "home_ppg": [2.5], "home_wr": [0.8],
        "away_gf": [1.0], "away_ga": [1.5], "away_ppg": [1.0], "away_wr": [0.3],
        "home_rest": [7.0], "away_rest": [4.0],
    })
    X = features.assemble_features(df)
    assert list(X.columns) == features.FEATURE_COLS
    assert X["is_home"].iloc[0] == 1
    assert X["form_gf_diff"].iloc[0] == pytest.approx(1.0)
    assert X["rest_diff"].iloc[0] == pytest.approx(3.0)


def test_single_match_row_coincide_con_assemble():
    row = features.single_match_row(
        elo_home=1600, elo_away=1500, neutral=True, importance_k=60,
        home_form={"gf": 2.0, "ga": 0.5, "ppg": 2.5, "wr": 0.8},
        away_form={"gf": 1.0, "ga": 1.5, "ppg": 1.0, "wr": 0.3},
        home_rest=7, away_rest=4,
    )
    assert list(row.columns) == features.FEATURE_COLS
    assert row["is_home"].iloc[0] == 0  # neutral
    assert row["elo_diff"].iloc[0] == pytest.approx(100.0)
