"""Nivel 1 — Sistema Elo dinámico (método World Football Elo / eloratings.net).

Fórmula de actualización (suma cero entre los dos equipos):

    R'_local = R_local + K · G · (W − We)

con:
    We  = 1 / (1 + 10^(-dr/400))        # esperanza de puntuación del local
    dr  = R_local − R_visitante + ventaja_localía   (0 si campo neutral)
    W   = 1 (gana local) / 0.5 (empate) / 0 (pierde local)
    G   = multiplicador por diferencia de goles
    K   = índice de importancia del torneo (settings.ELO_K_BY_IMPORTANCE)

El recorrido cronológico de TODO el histórico produce:
- el rating actual de cada selección (artefacto `models/elo_ratings.csv`), y
- el Elo **previo** a cada partido (columnas `elo_home`/`elo_away`/`elo_diff`),
  que `features.py` usa como entrada del ML SIN fuga de información.

`blend_with_fifa()` ancla la fuerza actual de las selecciones con los puntos FIFA
(que están en una escala tipo-Elo), corrigiendo derivas por exceso de amistosos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src import data_loader


# --------------------------------------------------------------------------- #
# Componentes de la fórmula
# --------------------------------------------------------------------------- #
def expected_score(rating_home: float, rating_away: float, neutral: bool = False,
                   home_advantage: float = settings.ELO_HOME_ADVANTAGE) -> float:
    """Esperanza de puntuación del local (We), incluida la ventaja de localía."""
    dr = rating_home - rating_away + (0.0 if neutral else home_advantage)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def goal_multiplier(goal_diff: int) -> float:
    """Multiplicador G por margen de goles (eloratings.net):

    |gd| <= 1 -> 1.0 ; |gd| == 2 -> 1.5 ; |gd| >= 3 -> (11 + |gd|) / 8
    """
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def k_factor(importance: str) -> float:
    """K base por importancia del torneo."""
    return data_loader.k_factor(importance)


# --------------------------------------------------------------------------- #
# Modelo Elo
# --------------------------------------------------------------------------- #
class EloModel:
    """Mantiene los ratings Elo y los actualiza partido a partido."""

    def __init__(self, initial: float = settings.ELO_INITIAL,
                 home_advantage: float = settings.ELO_HOME_ADVANTAGE):
        self.ratings: dict[str, float] = {}
        self.initial = initial
        self.home_advantage = home_advantage

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def update_one(self, home: str, away: str, home_score: int, away_score: int,
                   importance: str = "friendly", neutral: bool = False) -> tuple[float, float]:
        """Actualiza el rating de ambos equipos con un resultado.

        Devuelve los ratings PREVIOS (rating_home, rating_away) para poder
        construir features sin fuga. La actualización es de suma cero.
        """
        rh, ra = self.rating(home), self.rating(away)
        we = expected_score(rh, ra, neutral, self.home_advantage)

        if home_score > away_score:
            w = 1.0
        elif home_score == away_score:
            w = 0.5
        else:
            w = 0.0

        delta = k_factor(importance) * goal_multiplier(home_score - away_score) * (w - we)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        return rh, ra

    def process(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Recorre los partidos en orden cronológico actualizando los ratings.

        Devuelve una copia de `matches` con columnas extra del Elo PREVIO al
        partido: `elo_home`, `elo_away`, `elo_diff`. Asume que `matches` ya viene
        ordenado por fecha (como lo entrega `data_loader.load_matches`).
        """
        n = len(matches)
        elo_home = np.empty(n)
        elo_away = np.empty(n)

        homes = matches["home_team"].to_numpy()
        aways = matches["away_team"].to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        imp = matches["importance"].to_numpy()
        neu = matches["neutral"].to_numpy()

        for i in range(n):
            rh, ra = self.update_one(homes[i], aways[i], hs[i], as_[i], imp[i], bool(neu[i]))
            elo_home[i] = rh
            elo_away[i] = ra

        out = matches.copy()
        out["elo_home"] = elo_home
        out["elo_away"] = elo_away
        out["elo_diff"] = elo_home - elo_away
        return out

    def to_frame(self) -> pd.DataFrame:
        """Ratings actuales ordenados de mayor a menor."""
        df = pd.DataFrame(
            {"team": list(self.ratings.keys()), "elo": list(self.ratings.values())}
        )
        return df.sort_values("elo", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ancla FIFA: lleva los puntos FIFA a la escala del Elo y mezcla
# --------------------------------------------------------------------------- #
def blend_with_fifa(elo_ratings: dict[str, float], fifa_points: dict[str, float],
                    w: float = settings.ELO_FIFA_BLEND_W) -> dict[str, float]:
    """Mezcla el Elo calculado con el ranking FIFA como ancla de fuerza ACTUAL.

    Los puntos FIFA se reescalan a la escala del Elo con un ajuste lineal por
    mínimos cuadrados sobre los equipos comunes (elo ≈ a·fifa + b). El resultado
    es `w·elo + (1-w)·fifa_en_escala_elo`. Equipos sin dato FIFA conservan su Elo.
    """
    common = [t for t in elo_ratings if t in fifa_points]
    if len(common) < 5:  # muy pocos para un ajuste fiable -> sin mezcla
        return dict(elo_ratings)

    x = np.array([fifa_points[t] for t in common], dtype=float)
    y = np.array([elo_ratings[t] for t in common], dtype=float)
    a, b = np.polyfit(x, y, 1)

    blended: dict[str, float] = {}
    for team, elo in elo_ratings.items():
        if team in fifa_points:
            fifa_on_elo = a * fifa_points[team] + b
            blended[team] = w * elo + (1.0 - w) * fifa_on_elo
        else:
            blended[team] = elo
    return blended


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def save_ratings(model: EloModel, fifa_points: dict[str, float] | None = None,
                 path=settings.ELO_RATINGS_CSV) -> pd.DataFrame:
    """Guarda los ratings: Elo crudo y, si hay FIFA, la fuerza mezclada."""
    df = model.to_frame()
    if fifa_points:
        blended = blend_with_fifa(model.ratings, fifa_points)
        df["strength"] = df["team"].map(blended)
    else:
        df["strength"] = df["elo"]
    df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def load_ratings(path=settings.ELO_RATINGS_CSV) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)
