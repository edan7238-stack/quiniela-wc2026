"""Hito A2/A3 — Carga pre-torneo de señales por selección (xG/xGA, valor de plantilla).

El usuario aporta un CSV (`data/xg.csv`) cuyo esquema exacto puede variar. Este cargador
es **flexible** y soporta DOS formatos, que autodetecta por sus columnas:

1. **Agregado por equipo** (una fila por selección): columnas tipo `team, xg_for, xg_against`.
2. **Por partido** (una fila por encuentro, muchas ligas/clubes): columnas tipo
   `date, home_team, away_team, home_xg, away_xg`. En este modo el cargador:
     (a) autodetecta las columnas;
     (b) filtra a partidos **entre selecciones** (ambos equipos en `data_loader.all_teams()`),
         lo que descarta de forma natural todos los partidos de clubes;
     (c) aplica el corte de recencia (`CORTE_RECIENTE`) y pondera por **decaimiento temporal**
         (`XG_DECAY_PER_DAY`): los partidos recientes pesan más;
     (d) promedia (ponderado) el xG a favor / en contra de cada selección.

En ambos casos los nombres se normalizan a los canónicos del dataset Kaggle. Si no hay CSV,
devuelve mapas vacíos y el modelo de marcador funciona solo con la fuerza (Elo+FIFA).

xG/xGA de selecciones es escaso → se usa como **señal mezclada** (ver `XG_BLEND_W`), no única.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src import data_loader

# Alias de cabeceras (en minúsculas) -> nombre interno, para el formato AGREGADO por equipo.
_COL_ALIASES: dict[str, str] = {
    # equipo
    "team": "team", "equipo": "team", "selection": "team", "seleccion": "team",
    "selección": "team", "country": "team", "national_team": "team", "nation": "team",
    # xG a favor
    "xg_for": "xg_for", "xgf": "xg_for", "xg": "xg_for", "xg_scored": "xg_for",
    "xg_for_per_match": "xg_for", "xgfor": "xg_for", "xg_a_favor": "xg_for",
    "xg_per_game": "xg_for", "npxg": "xg_for",
    # xG en contra
    "xg_against": "xg_against", "xga": "xg_against", "xg_conceded": "xg_against",
    "xgagainst": "xg_against", "xg_en_contra": "xg_against", "xgad": "xg_against",
    # valor de plantilla
    "squad_value": "squad_value", "value": "squad_value", "market_value": "squad_value",
    "valor": "squad_value", "valor_plantilla": "squad_value",
    # partidos (para ponderar)
    "matches": "matches", "games": "matches", "partidos": "matches", "mp": "matches",
}

# Alias de cabeceras (en minúsculas) -> nombre interno, para el formato POR PARTIDO.
_MATCH_ALIASES: dict[str, str] = {
    # fecha
    "date": "date", "fecha": "date", "match_date": "date", "kickoff": "date",
    "kickoff_time": "kickoff_time",  # se ignora (hora suelta), no confundir con la fecha
    # equipo local / visitante
    "home_team": "home_team", "home": "home_team", "hometeam": "home_team",
    "local": "home_team", "equipo_local": "home_team",
    "away_team": "away_team", "away": "away_team", "awayteam": "away_team",
    "visitante": "away_team", "equipo_visitante": "away_team",
    # xG local / visitante
    "home_xg": "home_xg", "hxg": "home_xg", "xg_home": "home_xg", "home_xg_for": "home_xg",
    "xg_local": "home_xg", "home_npxg": "home_xg",
    "away_xg": "away_xg", "axg": "away_xg", "xg_away": "away_xg", "away_xg_for": "away_xg",
    "xg_visitante": "away_xg", "away_npxg": "away_xg",
    # liga / competición (opcional, solo informativo)
    "league_division": "league", "league": "league", "competition": "league",
    "tournament": "league", "competicion": "league", "liga": "league",
}

# Equivalencias de nombres que suelen venir distinto en fuentes de xG (-> nombre del dataset).
_TEAM_ALIASES: dict[str, str] = {
    "USA": "United States", "Korea Republic": "South Korea", "South Korea": "South Korea",
    "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "China PR": "China", "Czechia": "Czech Republic", "Türkiye": "Turkey",
    "Turkiye": "Turkey", "Cabo Verde": "Cape Verde", "Bosnia": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo", "Curacao": "Curaçao",
    # Variantes vistas en CSV de xG por partido (selecciones):
    "Bosnia & Herzegovina": "Bosnia and Herzegovina", "D.R. Congo": "DR Congo",
    "Central Africa": "Central African Republic", "Chinese Taipei": "Taiwan",
    "Guinea Bissau": "Guinea-Bissau", "Ireland": "Republic of Ireland",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


class TeamRatings:
    """Acceso a las señales pre-torneo. Vacío si no hay CSV.

    `source` indica de dónde salieron los datos: ``"per_match"``, ``"aggregate"`` o
    ``"empty"`` (útil para mostrarlo en el dashboard / depurar).
    """

    def __init__(self, df: pd.DataFrame | None, source: str = "aggregate"):
        self.df = df if df is not None else pd.DataFrame(columns=["team"])
        self.source = source if df is not None and not self.df.empty else "empty"
        self._xgf = dict(zip(self.df.get("team", []), self.df.get("xg_for", [])))
        self._xga = dict(zip(self.df.get("team", []), self.df.get("xg_against", [])))
        self._val = (dict(zip(self.df["team"], self.df["squad_value"]))
                     if "squad_value" in self.df else {})
        self._n = (dict(zip(self.df["team"], self.df["matches"]))
                   if "matches" in self.df else {})

    @property
    def has_xg(self) -> bool:
        return bool(self._xgf) and bool(self._xga)

    def xg_for(self, team: str) -> float | None:
        v = self._xgf.get(team)
        return float(v) if v is not None and pd.notna(v) else None

    def xg_against(self, team: str) -> float | None:
        v = self._xga.get(team)
        return float(v) if v is not None and pd.notna(v) else None

    def squad_value(self, team: str) -> float | None:
        return self._val.get(team)

    def n_matches(self, team: str) -> int:
        """Nº de partidos que respaldan el xG de la selección (0 si no hay)."""
        v = self._n.get(team)
        return int(v) if v is not None and pd.notna(v) else 0

    def teams_with_xg(self) -> list[str]:
        return sorted(t for t in self._xgf if pd.notna(self._xgf[t]))

    def league_avg_xg(self) -> float:
        """Media de xG a favor (referencia para normalizar). 1.36 si no hay datos."""
        vals = [v for v in self._xgf.values() if pd.notna(v)]
        return float(sum(vals) / len(vals)) if vals else settings.GOAL_BASE_TOTAL / 2.0


def _norm_team(name: str) -> str:
    name = str(name).strip()
    name = _TEAM_ALIASES.get(name, name)
    return data_loader.canonical_name(name)


def _rename_lower(raw: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Renombra columnas usando un mapa de alias en minúsculas."""
    rename = {c: aliases[c.strip().lower()] for c in raw.columns
              if c.strip().lower() in aliases}
    return raw.rename(columns=rename)


def _is_per_match(raw: pd.DataFrame) -> bool:
    """¿El CSV trae una fila por partido (home/away + xG de ambos)?"""
    cols = {c.strip().lower() for c in raw.columns}
    have = {_MATCH_ALIASES[c] for c in cols if c in _MATCH_ALIASES}
    return {"home_team", "away_team", "home_xg", "away_xg"}.issubset(have)


# --------------------------------------------------------------------------- #
# Formato POR PARTIDO (Hito A3)
# --------------------------------------------------------------------------- #
def national_xg_long(raw: pd.DataFrame | None = None, *,
                     teams: set[str] | None = None) -> pd.DataFrame | None:
    """Formato largo de partidos **selección-vs-selección** con xG, SIN agregar.

    Devuelve columnas `date, team, xgf, xga` (una fila por equipo y partido: su xG a
    favor y en contra), ya normalizado, filtrado a selecciones del histórico y recortado
    por `CORTE_RECIENTE`. Útil para backtests **point-in-time** (agregar solo con partidos
    anteriores a una fecha). Si `raw` es None, lee `settings.XG_CSV`. None si no hay datos.
    """
    if raw is None:
        if not settings.XG_CSV.exists():
            return None
        raw = pd.read_csv(settings.XG_CSV, low_memory=False)

    df = _rename_lower(raw, _MATCH_ALIASES)
    need = {"home_team", "away_team", "home_xg", "away_xg", "date"}
    if not need.issubset(df.columns):
        return None

    df = df[list(need)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")
    df = df.dropna(subset=["date", "home_xg", "away_xg"])

    # Normalizar nombres y quedarnos SOLO con partidos entre selecciones del histórico.
    if teams is None:
        teams = set(data_loader.all_teams())
    df["home_team"] = df["home_team"].map(_norm_team)
    df["away_team"] = df["away_team"].map(_norm_team)
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)]

    # Corte de recencia (filtro duro) — el decaimiento hace el resto del trabajo.
    df = df[df["date"].dt.year >= settings.CORTE_RECIENTE]
    if df.empty:
        return None

    # Cada partido aporta una fila por equipo (su xG a favor / en contra).
    home = df.rename(columns={"home_team": "team", "home_xg": "xgf", "away_xg": "xga"})
    away = df.rename(columns={"away_team": "team", "away_xg": "xgf", "home_xg": "xga"})
    long = pd.concat([home[["date", "team", "xgf", "xga"]],
                      away[["date", "team", "xgf", "xga"]]], ignore_index=True)
    return long.reset_index(drop=True)


def _aggregate_per_match(raw: pd.DataFrame, *, teams: set[str] | None = None) -> pd.DataFrame | None:
    """Filtra a partidos entre selecciones, pondera por decaimiento y promedia el xG.

    Devuelve un DataFrame agregado por equipo (`team, xg_for, xg_against, matches`) o
    None si tras filtrar no queda ningún partido válido.
    """
    long = national_xg_long(raw, teams=teams)
    if long is None or long.empty:
        return None

    # Peso por decaimiento temporal respecto al partido más reciente disponible.
    long = long.copy()
    ref = long["date"].max()
    age_days = (ref - long["date"]).dt.days.clip(lower=0)
    long["w"] = np.exp(-settings.XG_DECAY_PER_DAY * age_days)

    long["wxgf"] = long["xgf"] * long["w"]
    long["wxga"] = long["xga"] * long["w"]
    g = long.groupby("team", as_index=False).agg(
        wxgf=("wxgf", "sum"), wxga=("wxga", "sum"),
        w=("w", "sum"), matches=("team", "size"))
    g["xg_for"] = g["wxgf"] / g["w"]
    g["xg_against"] = g["wxga"] / g["w"]

    # Guardarraíl: descartar selecciones con muy pocos partidos (xG poco fiable).
    g = g[g["matches"] >= settings.XG_MIN_MATCHES]
    if g.empty:
        return None
    return g[["team", "xg_for", "xg_against", "matches"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Formato AGREGADO por equipo (Hito A2, original)
# --------------------------------------------------------------------------- #
def _aggregate_per_team(raw: pd.DataFrame, path_name: str) -> pd.DataFrame:
    df = _rename_lower(raw, _COL_ALIASES)
    if "team" not in df.columns:
        raise ValueError(
            f"{path_name}: no se encontró la columna de equipo. "
            f"Columnas vistas: {list(raw.columns)}")
    keep = [c for c in ("team", "xg_for", "xg_against", "squad_value", "matches")
            if c in df.columns]
    df = df[keep].copy()
    df["team"] = df["team"].map(_norm_team)
    for c in ("xg_for", "xg_against", "squad_value", "matches"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["team"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def load_xg(path=settings.XG_CSV) -> pd.DataFrame | None:
    """Lee el CSV de xG (autodetecta formato por-partido o agregado). None si no existe."""
    if not path.exists():
        return None
    raw = pd.read_csv(path, low_memory=False)
    if _is_per_match(raw):
        return _aggregate_per_match(raw)
    return _aggregate_per_team(raw, path.name)


def load() -> TeamRatings:
    if not settings.XG_CSV.exists():
        return TeamRatings(None)
    raw = pd.read_csv(settings.XG_CSV, low_memory=False)
    if _is_per_match(raw):
        return TeamRatings(_aggregate_per_match(raw), source="per_match")
    return TeamRatings(_aggregate_per_team(raw, settings.XG_CSV.name), source="aggregate")
