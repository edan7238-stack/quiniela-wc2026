"""Nivel 2 — Clasificador ML 1X2 (eje principal).

Predice probabilidades netas de Local (H) / Empate (D) / Visitante (A).

- Algoritmo: **LightGBM multiclase** (fallback automático a Regresión Logística
  multinomial si LightGBM no estuviera disponible).
- **Calibración** de probabilidades (isotónica, prefit) sobre un tramo de validación
  — imprescindible para que las cuotas justas (1/p) sirvan para detectar +EV.
- **Validación temporal**: split por fecha (train < calib < test), sin fuga.
- Entrena solo con la ventana reciente (`settings.CORTE_RECIENTE`), pero la forma y
  el Elo se calculan sobre TODO el histórico para que las features de 2018 sean correctas.

El artefacto se guarda en `models/ml_1x2.pkl` como un dict con el modelo, el orden de
features y las clases.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

from config import settings
from src import data_loader, elo, features

CLASSES = ["A", "D", "H"]  # orden alfabético (el que usa el clasificador)


# --------------------------------------------------------------------------- #
# Construcción del dataset (Elo + forma sobre todo el histórico)
# --------------------------------------------------------------------------- #
def build_training_frame() -> pd.DataFrame:
    """Histórico completo con Elo previo y forma reciente añadidos."""
    matches = data_loader.load_matches()
    matches = elo.EloModel().process(matches)
    matches = features.FormTracker().process(matches)
    return matches


def _temporal_split(n: int, frac_test: float, frac_calib: float) -> tuple[slice, slice, slice]:
    i_train = int(n * (1.0 - frac_test - frac_calib))
    i_calib = int(n * (1.0 - frac_test))
    return slice(0, i_train), slice(i_train, i_calib), slice(i_calib, n)


# --------------------------------------------------------------------------- #
# Estimadores
# --------------------------------------------------------------------------- #
def _fit_lightgbm(Xtr, ytr, Xca, yca):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xca, yca)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    return clf


def _fit_logistic(Xtr, ytr, *_):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                  random_state=settings.RANDOM_STATE)),
    ])
    pipe.fit(Xtr, ytr)
    return pipe


def _calibrate(estimator, Xca, yca, method: str = "sigmoid"):
    """Calibra (sigmoid/isotonic) sin reentrenar el estimador base."""
    try:  # sklearn >= 1.6
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except Exception:  # versiones previas
        cal = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    cal.fit(Xca, yca)
    return cal


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _onehot(y) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = np.zeros((len(y), len(CLASSES)))
    for i, v in enumerate(y):
        m[i, idx[v]] = 1.0
    return m


def _brier(proba: np.ndarray, y) -> float:
    return float(np.mean(np.sum((proba - _onehot(y)) ** 2, axis=1)))


def _ece(proba: np.ndarray, y, n_bins: int = 10) -> float:
    """Expected Calibration Error (basado en la confianza de la clase predicha).

    Mide cuánto se desvían las probabilidades de la frecuencia real. 0 = perfecto.
    Es la métrica clave para confiar en las cuotas justas (1/p) del +EV.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    true = np.array([CLASSES.index(v) for v in y])
    correct = (pred == true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def _metrics(proba: np.ndarray, y) -> dict[str, float]:
    return {
        "log_loss": float(log_loss(y, proba, labels=CLASSES)),
        "brier": _brier(proba, y),
        "accuracy": float(accuracy_score(y, [CLASSES[i] for i in proba.argmax(1)])),
        "ece": _ece(proba, np.asarray(y)),
    }


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def _proba(model, X) -> np.ndarray:
    order = [list(model.classes_).index(c) for c in CLASSES]
    return model.predict_proba(X)[:, order]


def train(frac_test: float = settings.VALIDACION_HOLDOUT_FRAC,
          frac_calib: float = 0.15, calibrate: str = "none",
          save: bool = True, verbose: bool = True) -> dict:
    """Entrena el clasificador 1X2 con validación temporal.

    `calibrate`: "none" (por defecto; el LightGBM crudo ya está bien calibrado),
    "sigmoid" o "isotonic". El modelo final se elige según este parámetro.
    """
    matches = build_training_frame()
    recent = matches[matches["year"] >= settings.CORTE_RECIENTE].reset_index(drop=True)
    recent = recent.sort_values("date", kind="stable").reset_index(drop=True)

    X = features.assemble_features(recent)
    y = recent["result"]

    s_tr, s_ca, s_te = _temporal_split(len(recent), frac_test, frac_calib)
    Xtr, ytr = X.iloc[s_tr], y.iloc[s_tr]
    Xca, yca = X.iloc[s_ca], y.iloc[s_ca]
    Xte, yte = X.iloc[s_te], y.iloc[s_te]

    used = "LightGBM"
    try:
        base = _fit_lightgbm(Xtr, ytr, Xca, yca)
    except Exception as e:  # pragma: no cover - solo si falla LightGBM
        if verbose:
            print(f"      [aviso] LightGBM no disponible ({e}); usando Regresión Logística.")
        base = _fit_logistic(Xtr, ytr, Xca, yca)
        used = "LogisticRegression"

    model = base if calibrate == "none" else _calibrate(base, Xca, yca, method=calibrate)

    priors = ytr.value_counts(normalize=True).reindex(CLASSES).fillna(0).to_numpy()
    yte_np = yte.to_numpy()
    metrics = {
        "algoritmo": used,
        "calibracion": calibrate,
        "n_train": int(len(Xtr)), "n_calib": int(len(Xca)), "n_test": int(len(Xte)),
        "fecha_train": (str(recent["date"].iloc[s_tr.start].date()),
                        str(recent["date"].iloc[s_tr.stop - 1].date())),
        "fecha_test": (str(recent["date"].iloc[s_te.start].date()),
                       str(recent["date"].iloc[len(recent) - 1].date())),
        "test": _metrics(_proba(model, Xte), yte_np),
        "test_sin_calibrar": _metrics(_proba(base, Xte), yte_np),
        "baseline_priors": _metrics(np.tile(priors, (len(yte), 1)), yte_np),
    }

    if save:
        bundle = {
            "model": model,
            "features": features.FEATURE_COLS,
            "classes": CLASSES,
            "algoritmo": used,
            "calibracion": calibrate,
            "metrics": metrics,
        }
        settings.ML_MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, settings.ML_MODEL_PKL)

    if verbose:
        _print_metrics(metrics, base)
    return metrics


def _print_metrics(m: dict, base=None) -> None:
    print(f"      Algoritmo: {m['algoritmo']} | calibración: {m['calibracion']}")
    print(f"      Train: {m['n_train']:,} ({m['fecha_train'][0]} -> {m['fecha_train'][1]}) | "
          f"Calib: {m['n_calib']:,} | Test: {m['n_test']:,} "
          f"({m['fecha_test'][0]} -> {m['fecha_test'][1]})")
    print("      " + "-" * 55)
    print("      Conjunto            log_loss   brier   accuracy    ECE")
    filas = [("baseline_priors", "Baseline (priors)"),
             ("test_sin_calibrar", "Test sin calibrar")]
    if m["calibracion"] != "none":
        filas.append(("test", "Test CALIBRADO"))
    else:
        filas[-1] = ("test", "Test (modelo final)")
    for k, label in filas:
        d = m[k]
        print(f"      {label:<18} {d['log_loss']:.4f}   {d['brier']:.4f}  {d['accuracy']:.3f}"
              f"     {d['ece']:.3f}")
    if base is not None and hasattr(base, "feature_importances_"):
        imp = sorted(zip(features.FEATURE_COLS, base.feature_importances_),
                     key=lambda t: t[1], reverse=True)
        print("      Importancia de features:",
              ", ".join(f"{f}={v}" for f, v in imp[:5]))


# --------------------------------------------------------------------------- #
# Carga y predicción
# --------------------------------------------------------------------------- #
def load_bundle(path=settings.ML_MODEL_PKL):
    if not path.exists():
        return None
    return joblib.load(path)


def predict_proba(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas H/D/A (probabilidades) para cada fila de X."""
    model = bundle["model"]
    proba = model.predict_proba(X[bundle["features"]])
    order = [list(model.classes_).index(c) for c in bundle["classes"]]
    proba = proba[:, order]
    return pd.DataFrame(proba, columns=bundle["classes"], index=X.index)
