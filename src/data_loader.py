"""Capa de datos: carga, limpieza y normalización del histórico de partidos.

Responsabilidades (Fase 1):
- Cargar el dataset Kaggle (`results.csv`, `shootouts.csv`, ...).
- Normalizar nombres de selecciones (cambios históricos vía `former_names.csv`
  + alias para casar con los nombres oficiales del Mundial 2026).
- Clasificar cada torneo por importancia (para el K del Elo y como feature).
- Etiquetar el resultado 1X2 y los puntos desde la perspectiva del local.
- Aplicar el FILTRO DURO de recencia (`CORTE_RECIENTE`) cuando se pida.
- Integrar los resultados del torneo ingresados manualmente.

Niveles superiores (Elo, ML, Poisson) consumen el DataFrame que devuelve
`load_matches()`.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from config import settings

# --------------------------------------------------------------------------- #
# Nombre canónico = el del dataset Kaggle (es la fuente de entrenamiento, así que
# el modelo aprende con esos nombres). Las demás fuentes (FIFA, cuotas, config del
# Mundial) se mapean HACIA estos nombres. Este diccionario queda como gancho para
# corregir, si hiciera falta, alguna grafía interna del propio dataset; hoy va vacío.
# --------------------------------------------------------------------------- #
CANONICAL_ALIASES: dict[str, str] = {}

# Columnas que esperamos en results.csv
_RESULT_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


# --------------------------------------------------------------------------- #
# Clasificación de torneos por importancia
# --------------------------------------------------------------------------- #
def tournament_importance(name: str) -> str:
    """Mapea el texto libre de `tournament` a una clave de importancia.

    Las claves coinciden con `settings.ELO_K_BY_IMPORTANCE`.
    """
    if not isinstance(name, str):
        return "other"
    low = name.lower()

    if "friendly" in low:
        return "friendly"
    if "nations league" in low:
        return "nations_league"
    if "confederations" in low:
        return "confederations"

    # Mundial: distinguir fase final de la clasificatoria.
    if "world cup" in low:
        return "world_cup_qual" if "qualif" in low else "world_cup"

    # Cualquier otra clasificatoria continental.
    if "qualif" in low:
        return "continental_qual"

    # Fases finales continentales principales.
    majors = (
        "uefa euro", "copa américa", "copa america",
        "african cup of nations", "afcon",
        "asian cup", "gold cup", "concacaf championship",
        "oceania nations", "nations cup", "confederation",
    )
    if any(k in low for k in majors):
        return "continental"

    return "other"


def k_factor(importance: str) -> float:
    """K base del Elo para una categoría de importancia."""
    return settings.ELO_K_BY_IMPORTANCE.get(importance, settings.ELO_K_BY_IMPORTANCE["other"])


# --------------------------------------------------------------------------- #
# Normalización de nombres de selecciones
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _former_names() -> pd.DataFrame:
    fn = pd.read_csv(settings.FORMER_NAMES_CSV)
    fn["start_date"] = pd.to_datetime(fn["start_date"], errors="coerce")
    fn["end_date"] = pd.to_datetime(fn["end_date"], errors="coerce")
    return fn.dropna(subset=["current", "former"])


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza nombres antiguos por el actual (dependiente de la fecha) y
    aplica los alias canónicos. Opera sobre `home_team` y `away_team`.
    """
    df = df.copy()
    fn = _former_names()

    # 1) Cambios históricos dependientes de fecha (p. ej. Swaziland -> Eswatini).
    for row in fn.itertuples(index=False):
        in_range = (df["date"] >= row.start_date) & (df["date"] <= row.end_date)
        for col in ("home_team", "away_team"):
            df.loc[in_range & (df[col] == row.former), col] = row.current

    # 2) Alias canónicos (no dependen de fecha).
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(CANONICAL_ALIASES)

    return df


def canonical_name(name: str) -> str:
    """Normaliza un nombre suelto (sin fecha) usando solo los alias canónicos."""
    return CANONICAL_ALIASES.get(name, name)


# --------------------------------------------------------------------------- #
# Etiquetado de resultados
# --------------------------------------------------------------------------- #
def _label_results(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas derivadas desde la perspectiva del local:

    - result: 'H' (gana local), 'D' (empate), 'A' (gana visitante)
    - goal_diff: home_score - away_score
    - home_points / away_points: 3/1/0
    """
    df = df.copy()
    home = df["home_score"].astype("Int64")
    away = df["away_score"].astype("Int64")

    conds = [home > away, home == away]
    df["result"] = np.select(conds, ["H", "D"], default="A")
    df["goal_diff"] = (home - away).astype("Int64")
    df["home_points"] = np.select(conds, [3, 1], default=0)
    df["away_points"] = np.select([away > home, home == away], [3, 1], default=0)
    return df


# --------------------------------------------------------------------------- #
# Carga principal
# --------------------------------------------------------------------------- #
def load_results_raw() -> pd.DataFrame:
    """Lee results.csv tal cual (con fechas parseadas y `neutral` booleano)."""
    df = pd.read_csv(settings.RESULTS_CSV)
    missing = set(_RESULT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"results.csv no tiene las columnas esperadas: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    if df["neutral"].dtype != bool:
        df["neutral"] = (
            df["neutral"].astype(str).str.strip().str.upper().isin({"TRUE", "1", "T"})
        )
    return df.reset_index(drop=True)


def load_wc2026_manual() -> pd.DataFrame:
    """Carga los resultados del torneo ingresados manualmente (si existen).

    El archivo `data/wc2026_results.csv` comparte el esquema de results.csv y se
    rellena desde el dashboard tras cada jornada. Devuelve un DataFrame vacío con
    el esquema correcto si aún no hay datos.
    """
    cols = _RESULT_COLS
    if not settings.WC2026_RESULTS_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(settings.WC2026_RESULTS_CSV)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.upper().isin({"TRUE", "1", "T"})
        if "neutral" in df.columns else True  # los partidos del Mundial son en sede neutral
    )
    return df.dropna(subset=["date", "home_score", "away_score"]).reset_index(drop=True)


def load_matches(
    *,
    recent_only: bool = False,
    since_year: int | None = None,
    include_manual: bool = True,
) -> pd.DataFrame:
    """Devuelve el histórico limpio, normalizado y etiquetado.

    Parámetros
    ----------
    recent_only:
        Si True, aplica el FILTRO DURO de recencia (`settings.CORTE_RECIENTE`).
        Úsalo para entrenar el ML (Nivel 2). El Elo (Nivel 1) usa todo el histórico.
    since_year:
        Año de corte explícito (anula `CORTE_RECIENTE`).
    include_manual:
        Si True, añade los resultados del Mundial ingresados manualmente.

    Devuelve un DataFrame ordenado cronológicamente con columnas extra:
    `importance`, `k`, `result`, `goal_diff`, `home_points`, `away_points`, `year`.
    """
    df = load_results_raw()

    if include_manual:
        manual = load_wc2026_manual()
        if not manual.empty:
            df = pd.concat([df, manual[_RESULT_COLS]], ignore_index=True)

    df = normalize_team_names(df)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["importance"] = df["tournament"].map(tournament_importance)
    df["k"] = df["importance"].map(k_factor)
    df = _label_results(df)

    cutoff = since_year if since_year is not None else settings.CORTE_RECIENTE
    if recent_only:
        df = df[df["year"] >= cutoff].reset_index(drop=True)

    return df


def all_teams(df: pd.DataFrame | None = None) -> list[str]:
    """Lista ordenada de todas las selecciones presentes (nombres canónicos)."""
    if df is None:
        df = load_matches()
    teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
    return sorted(map(str, teams))
