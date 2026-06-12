"""Nivel 2 (entrada) — Ingeniería de features para el clasificador ML 1X2.

Todas las features se calculan con información **previa** al partido (sin fuga):
- Elo previo (de `elo.EloModel.process`).
- Forma reciente por equipo: medias móviles de goles a favor/en contra, puntos por
  partido y % de victorias sobre los últimos `FORMA_VENTANA_ANIOS` años (máx. 20 partidos).
- Días de descanso (desde el último partido de cada equipo).
- Importancia del torneo (K) y condición de localía.

NO se incluye ninguna señal del mercado de apuestas (el modelo debe ser independiente
para poder detectar valor +EV).

`FormTracker` replica el patrón cronológico de `EloModel`: `process()` añade las columnas
de forma a cada partido, y `current_form()` da la forma actual de un equipo para predecir.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from config import settings

# Columnas que entran al modelo (orden fijo).
FEATURE_COLS: list[str] = [
    "elo_diff",
    "elo_home",
    "elo_away",
    "is_home",            # 1 si el local juega en casa; 0 si campo neutral
    "importance_k",
    "form_gf_diff",       # dif. de goles a favor (forma) local - visitante
    "form_ga_diff",       # dif. de goles en contra (forma)
    "form_ppg_diff",      # dif. de puntos por partido (forma)
    "form_winrate_diff",  # dif. de % de victorias (forma)
    "rest_diff",          # dif. de días de descanso
]

TARGET = "result"  # 'H' / 'D' / 'A'


class FormTracker:
    """Calcula la forma reciente de cada selección recorriendo el histórico."""

    def __init__(self, window_years: int = settings.FORMA_VENTANA_ANIOS, max_games: int = 20):
        self.window = pd.DateOffset(years=window_years)
        self.max_games = max_games
        # team -> deque de (date, gf, ga, pts)
        self._hist: dict[str, deque] = defaultdict(deque)
        self._last_date: dict[str, pd.Timestamp] = {}

    def _form(self, team: str, asof: pd.Timestamp) -> dict[str, float]:
        h = self._hist[team]
        cutoff = asof - self.window
        while h and h[0][0] < cutoff:      # descarta lo más viejo que la ventana
            h.popleft()
        if not h:
            return {"gf": np.nan, "ga": np.nan, "ppg": np.nan, "wr": np.nan}
        recent = list(h)[-self.max_games:]
        gf = np.mean([e[1] for e in recent])
        ga = np.mean([e[2] for e in recent])
        pts = [e[3] for e in recent]
        return {"gf": gf, "ga": ga, "ppg": float(np.mean(pts)),
                "wr": float(np.mean([p == 3 for p in pts]))}

    def _rest(self, team: str, asof: pd.Timestamp) -> float:
        last = self._last_date.get(team)
        return np.nan if last is None else float((asof - last).days)

    def _append(self, team: str, date: pd.Timestamp, gf: int, ga: int) -> None:
        pts = 3 if gf > ga else 1 if gf == ga else 0
        self._hist[team].append((date, gf, ga, pts))
        self._last_date[team] = date

    def current_form(self, team: str, asof: pd.Timestamp | None = None) -> dict[str, float]:
        """Forma actual de un equipo (para predecir un partido futuro)."""
        asof = asof if asof is not None else (self._last_date.get(team) or pd.Timestamp.today())
        return self._form(team, asof)

    def process(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Añade columnas de forma/descanso (previas al partido) a cada fila."""
        n = len(matches)
        cols = {c: np.empty(n) for c in
                ("home_gf", "home_ga", "home_ppg", "home_wr",
                 "away_gf", "away_ga", "away_ppg", "away_wr",
                 "home_rest", "away_rest")}

        dates = matches["date"].to_numpy()
        homes = matches["home_team"].to_numpy()
        aways = matches["away_team"].to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()

        for i in range(n):
            d = pd.Timestamp(dates[i])
            ht, at = homes[i], aways[i]
            fh, fa = self._form(ht, d), self._form(at, d)
            cols["home_gf"][i], cols["home_ga"][i] = fh["gf"], fh["ga"]
            cols["home_ppg"][i], cols["home_wr"][i] = fh["ppg"], fh["wr"]
            cols["away_gf"][i], cols["away_ga"][i] = fa["gf"], fa["ga"]
            cols["away_ppg"][i], cols["away_wr"][i] = fa["ppg"], fa["wr"]
            cols["home_rest"][i] = self._rest(ht, d)
            cols["away_rest"][i] = self._rest(at, d)
            # registrar el partido DESPUÉS de calcular la forma (sin fuga)
            self._append(ht, d, hs[i], as_[i])
            self._append(at, d, as_[i], hs[i])

        out = matches.copy()
        for c, v in cols.items():
            out[c] = v
        return out


def assemble_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Construye las columnas de `FEATURE_COLS` a partir de un DataFrame que ya
    tiene Elo previo (de `EloModel.process`) y forma (de `FormTracker.process`).
    """
    df = matches
    feats = pd.DataFrame(index=df.index)
    feats["elo_diff"] = df["elo_diff"]
    feats["elo_home"] = df["elo_home"]
    feats["elo_away"] = df["elo_away"]
    feats["is_home"] = (~df["neutral"].astype(bool)).astype(int)
    feats["importance_k"] = df["k"].astype(float)
    feats["form_gf_diff"] = df["home_gf"] - df["away_gf"]
    feats["form_ga_diff"] = df["home_ga"] - df["away_ga"]
    feats["form_ppg_diff"] = df["home_ppg"] - df["away_ppg"]
    feats["form_winrate_diff"] = df["home_wr"] - df["away_wr"]
    feats["rest_diff"] = df["home_rest"] - df["away_rest"]
    return feats[FEATURE_COLS]


def single_match_row(*, elo_home: float, elo_away: float, neutral: bool,
                     importance_k: float, home_form: dict[str, float],
                     away_form: dict[str, float],
                     home_rest: float = np.nan, away_rest: float = np.nan) -> pd.DataFrame:
    """Construye la fila de features de UN partido a predecir (mismo orden que el train)."""
    row = {
        "elo_diff": elo_home - elo_away,
        "elo_home": elo_home,
        "elo_away": elo_away,
        "is_home": 0 if neutral else 1,
        "importance_k": float(importance_k),
        "form_gf_diff": home_form["gf"] - away_form["gf"],
        "form_ga_diff": home_form["ga"] - away_form["ga"],
        "form_ppg_diff": home_form["ppg"] - away_form["ppg"],
        "form_winrate_diff": home_form["wr"] - away_form["wr"],
        "rest_diff": home_rest - away_rest,
    }
    return pd.DataFrame([row])[FEATURE_COLS]
