"""Tests del Nivel 2 (clasificador ML) — funciones puras + integración ligera."""
import numpy as np
import pandas as pd
import pytest

from src import features, ml_model


# --------------------------------------------------------------------------- #
# Funciones puras
# --------------------------------------------------------------------------- #
def test_onehot():
    oh = ml_model._onehot(["H", "A", "D"])
    # CLASSES = ['A','D','H']
    assert oh.tolist() == [[0, 0, 1], [1, 0, 0], [0, 1, 0]]


def test_temporal_split_tamanos():
    s_tr, s_ca, s_te = ml_model._temporal_split(1000, frac_test=0.2, frac_calib=0.1)
    assert (s_tr.start, s_tr.stop) == (0, 700)
    assert (s_ca.start, s_ca.stop) == (700, 800)
    assert (s_te.start, s_te.stop) == (800, 1000)


def test_ece_perfecto_es_bajo():
    # Predicciones 100% confiadas y todas correctas -> ECE ~ 0.
    proba = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])
    y = np.array(["H", "A", "D"])
    assert ml_model._ece(proba, y) == pytest.approx(0.0, abs=1e-9)


def test_ece_detecta_exceso_de_confianza():
    # Muy confiado (0.99) pero siempre se equivoca -> ECE alto.
    proba = np.array([[0.99, 0.005, 0.005]] * 4)
    y = np.array(["H", "H", "H", "H"])  # predice A, acierta 0
    assert ml_model._ece(proba, y) > 0.9


def test_metrics_tiene_claves():
    proba = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    y = np.array(["H", "A"])
    m = ml_model._metrics(proba, y)
    assert set(m) == {"log_loss", "brier", "accuracy", "ece"}


# --------------------------------------------------------------------------- #
# Integración ligera (Regresión Logística, rápida) + roundtrip del bundle
# --------------------------------------------------------------------------- #
def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 200, n)
    X = pd.DataFrame({
        "elo_diff": elo_diff,
        "elo_home": 1500 + rng.normal(0, 100, n),
        "elo_away": 1500 + rng.normal(0, 100, n),
        "is_home": rng.integers(0, 2, n),
        "importance_k": rng.choice([20, 40, 60], n).astype(float),
        "form_gf_diff": rng.normal(0, 1, n),
        "form_ga_diff": rng.normal(0, 1, n),
        "form_ppg_diff": rng.normal(0, 1, n),
        "form_winrate_diff": rng.normal(0, 0.3, n),
        "rest_diff": rng.normal(0, 5, n),
    })[features.FEATURE_COLS]
    # objetivo correlacionado con elo_diff
    p = 1 / (1 + np.exp(-elo_diff / 200))
    y = np.where(rng.random(n) < p * 0.8, "H", np.where(rng.random(n) < 0.5, "D", "A"))
    return X, pd.Series(y)


def test_predict_proba_suma_uno_y_columnas(tmp_path):
    X, y = _synthetic()
    base = ml_model._fit_logistic(X, y)
    bundle = {"model": base, "features": features.FEATURE_COLS, "classes": ml_model.CLASSES}
    proba = ml_model.predict_proba(bundle, X)
    assert list(proba.columns) == ["A", "D", "H"]
    assert proba.to_numpy().sum(axis=1) == pytest.approx(np.ones(len(X)), abs=1e-6)


def test_bundle_roundtrip(tmp_path):
    import joblib
    X, y = _synthetic()
    base = ml_model._fit_logistic(X, y)
    bundle = {"model": base, "features": features.FEATURE_COLS, "classes": ml_model.CLASSES}
    path = tmp_path / "ml.pkl"
    joblib.dump(bundle, path)
    loaded = ml_model.load_bundle(path)
    p1 = ml_model.predict_proba(bundle, X)
    p2 = ml_model.predict_proba(loaded, X)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
