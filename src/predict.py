"""Orquestador de predicción (Niveles 1-3) para un partido individual.

Combina:
- Nivel 1 (Elo + ancla FIFA): fuerza actual de cada selección.
- Nivel 2 (ML LightGBM): probabilidades 1X2 (EJE PRINCIPAL) + cuota justa = 1/p.
- Nivel 3 (Poisson + Dixon-Coles): marcador exacto y mercados (EJE SECUNDARIO),
  además de un 1X2 alternativo como cross-check.

`Predictor` reconstruye el estado actual (Elo y forma) recorriendo el histórico una vez
en `__init__` (cacheable en el dashboard) y carga los artefactos entrenados (ml_1x2.pkl,
poisson_params.pkl, fifa_snapshot.csv).
"""
from __future__ import annotations

import numpy as np

from config import settings, wc2026
from src import data_loader, elo, features, fifa, ml_model, poisson, team_ratings


class Predictor:
    def __init__(self, use_fifa_anchor: bool = True):
        self.use_fifa_anchor = use_fifa_anchor

        # Estado a partir del histórico completo (Elo previo y forma).
        matches = data_loader.load_matches()
        self._elo = elo.EloModel()
        self._elo.process(matches)
        self._form = features.FormTracker()
        self._form.process(matches)

        # Ancla FIFA -> fuerza actual mezclada (misma escala que el Elo).
        snap = fifa.load_snapshot()
        fifa_points = fifa.points_map(snap) if snap is not None else {}
        self._strength = (elo.blend_with_fifa(self._elo.ratings, fifa_points)
                          if (use_fifa_anchor and fifa_points) else dict(self._elo.ratings))

        # Señales pre-torneo opcionales (xG/xGA); vacío si no hay CSV.
        self._ratings = team_ratings.load()

        # Ajustes manuales de fuerza por equipo (perilla día-antes: lesiones, rotación...).
        self._adjust: dict[str, float] = {}

        # Artefactos entrenados.
        self._ml = ml_model.load_bundle()
        if self._ml is None:
            raise FileNotFoundError(
                "Falta models/ml_1x2.pkl. Ejecuta primero: python scripts/train.py")
        self._poisson = poisson.load()
        if self._poisson is None:
            raise FileNotFoundError(
                "Faltan los parámetros Poisson. Ejecuta: python scripts/train.py")

    # ----------------------------------------------------------------- #
    def strength(self, team: str) -> float:
        return self._strength.get(team, settings.ELO_INITIAL) + self._adjust.get(team, 0.0)

    def set_adjustment(self, team: str, delta: float) -> None:
        """Ajuste manual de fuerza (puntos Elo) para un equipo. 0 lo elimina."""
        if delta:
            self._adjust[team] = float(delta)
        else:
            self._adjust.pop(team, None)

    def adjustments(self) -> dict[str, float]:
        return dict(self._adjust)

    def strength_map(self) -> dict[str, float]:
        """Fuerza actual (Elo+FIFA + ajustes manuales) de todas las selecciones, en vivo
        (incluye resultados manuales cargados). Útil para alimentar el Montecarlo."""
        return {t: self.strength(t) for t in self._strength}

    def teams(self) -> list[str]:
        return sorted(self._strength)

    # ----------------------------------------------------------------- #
    def predict_1x2(self, home: str, away: str, *, neutral: bool = True,
                    importance: str = "world_cup") -> dict:
        """Probabilidades 1X2 del ML (eje principal)."""
        row = features.single_match_row(
            elo_home=self.strength(home), elo_away=self.strength(away),
            neutral=neutral, importance_k=data_loader.k_factor(importance),
            home_form=self._form.current_form(home),
            away_form=self._form.current_form(away),
            home_rest=np.nan, away_rest=np.nan,
        )
        proba = ml_model.predict_proba(self._ml, row).iloc[0]
        probs = {c: float(proba[c]) for c in ("H", "D", "A")}
        return {"probs": probs}

    # ----------------------------------------------------------------- #
    # Modelo de marcador (Hito B): λ anclada a fuerza (+ xG si hay) + anfitrión + fase,
    # y matriz reconciliada con el 1X2 del ML.
    # ----------------------------------------------------------------- #
    def _score_lambdas(self, home: str, away: str, neutral: bool, stage: str) -> tuple[float, float]:
        slope = settings.GOAL_SLOPE_PER_ELO
        base = settings.GOAL_BASE_TOTAL * settings.GOAL_STAGE_FACTOR.get(stage, 1.0)
        sup = slope * (self.strength(home) - self.strength(away))
        if not neutral:  # ventaja de localía (anfitrión) en términos de goles
            sup += slope * settings.ELO_HOME_ADVANTAGE
        lam_h = (base + sup) / 2.0
        lam_a = (base - sup) / 2.0

        # Mezcla con xG/xGA si hay datos para ambos equipos.
        r = self._ratings
        if r.has_xg:
            xgf_h, xga_h = r.xg_for(home), r.xg_against(home)
            xgf_a, xga_a = r.xg_for(away), r.xg_against(away)
            if None not in (xgf_h, xga_h, xgf_a, xga_a):
                avg = r.league_avg_xg()
                f = settings.GOAL_STAGE_FACTOR.get(stage, 1.0)
                lam_h_xg = xgf_h * xga_a / avg * f
                lam_a_xg = xgf_a * xga_h / avg * f
                w = settings.XG_BLEND_W
                lam_h = (1 - w) * lam_h + w * lam_h_xg
                lam_a = (1 - w) * lam_a + w * lam_a_xg

        lo = settings.GOAL_LAMBDA_MIN
        return max(lam_h, lo), max(lam_a, lo)

    def score_matrix_for(self, home: str, away: str, *, neutral: bool = True,
                         stage: str = "group") -> np.ndarray:
        """Matriz de marcadores Dixon-Coles **reconciliada** con el 1X2 del ML."""
        lam_h, lam_a = self._score_lambdas(home, away, neutral, stage)
        m = poisson.score_matrix(lam_h, lam_a, self._poisson.rho, self._poisson.max_goals)
        ml = self.predict_1x2(home, away, neutral=neutral)["probs"]
        return poisson.reconcile_with_1x2(m, ml)

    def predict_score(self, home: str, away: str, *, neutral: bool = True,
                      stage: str = "group") -> dict:
        """Marcador exacto y mercados (matriz reconciliada con el ML)."""
        lam_h, lam_a = self._score_lambdas(home, away, neutral, stage)
        m = self.score_matrix_for(home, away, neutral=neutral, stage=stage)
        out = {"home": home, "away": away, "neutral": neutral,
               "lambda_home": lam_h, "lambda_away": lam_a, "score_matrix": m}
        out.update(poisson.markets_from_matrix(m))
        return out

    def predict(self, home: str, away: str, *, neutral: bool = True,
                importance: str = "world_cup", stage: str = "group") -> dict:
        """Predicción completa: 1X2 (ML) + marcador (reconciliado)."""
        ml = self.predict_1x2(home, away, neutral=neutral, importance=importance)
        ps = self.predict_score(home, away, neutral=neutral, stage=stage)
        return {
            "home": home, "away": away, "neutral": neutral, "importance": importance,
            "strength_home": self.strength(home), "strength_away": self.strength(away),
            "ml_1x2": ml["probs"],          # EJE PRINCIPAL
            "poisson_1x2": ps["prob_1x2"],  # = ML tras reconciliar (marginales 1X2)
            "most_likely_score": ps["most_likely_score"],
            "top_scores": ps["top_scores"],
            "lambda_home": ps["lambda_home"], "lambda_away": ps["lambda_away"],
            "over_2_5": ps["over_2_5"], "btts_yes": ps["btts_yes"],
            "score_matrix": ps["score_matrix"],
        }


def format_prediction(r: dict) -> str:
    """Resumen legible de una predicción (para CLI)."""
    H, D, A = r["ml_1x2"]["H"], r["ml_1x2"]["D"], r["ml_1x2"]["A"]
    ms = r["most_likely_score"]
    lines = [
        f"{r['home']} vs {r['away']}  ({'neutral' if r['neutral'] else 'localía'})",
        f"  Fuerza: {r['home']} {r['strength_home']:.0f}  |  {r['away']} {r['strength_away']:.0f}",
        f"  ML 1X2:   {r['home']} {H:.1%}  /  Empate {D:.1%}  /  {r['away']} {A:.1%}",
        f"  Poisson 1X2 (cross-check): {r['poisson_1x2']['H']:.1%} / "
        f"{r['poisson_1x2']['D']:.1%} / {r['poisson_1x2']['A']:.1%}",
        f"  Marcador más probable: {ms[0]}-{ms[1]}   "
        f"(λ {r['lambda_home']:.2f}-{r['lambda_away']:.2f})",
        f"  Top marcadores: " + ", ".join(f"{s[0]}-{s[1]} {p:.1%}" for s, p in r["top_scores"][:3]),
        f"  Over 2.5: {r['over_2_5']:.0%}   BTTS: {r['btts_yes']:.0%}",
    ]
    return "\n".join(lines)
