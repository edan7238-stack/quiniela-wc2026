"""Hito C — Optimizador de quiniela.

Elige, para cada partido, la predicción que **maximiza los puntos esperados** según las
reglas del usuario (no el marcador "más probable", que suele ser subóptimo):

    E[pts(s)] = pts_exacto·P(s) + pts_resultado·(P(o) − P(s)) + pts_fallo·(1 − P(o))

con el marcador `s` y su resultado `o(s)`. Para 3/1/0 esto es `2·P(s) + P(o)`.

También cubre los **comodines** (double ×2, triple ×3, all-in 12/5/−6) y su asignación
óptima a los partidos para maximizar el total de puntos esperados.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings, wc2026
from src import poisson


@dataclass(frozen=True)
class Scoring:
    exacto: float
    resultado: float
    fallo: float


DEFAULT = Scoring(settings.QUINIELA_PTS_EXACTO, settings.QUINIELA_PTS_RESULTADO,
                  settings.QUINIELA_PTS_FALLO)
ALLIN = Scoring(12.0, 5.0, -6.0)


def outcome_of(i: int, j: int) -> str:
    return "H" if i > j else ("D" if i == j else "A")


def expected_points(matrix: np.ndarray, i: int, j: int,
                    p_outcome: dict[str, float], scoring: Scoring = DEFAULT) -> float:
    """Puntos esperados de predecir el marcador (i, j)."""
    o = outcome_of(i, j)
    ps = float(matrix[i, j])
    po = float(p_outcome[o])
    return scoring.exacto * ps + scoring.resultado * (po - ps) + scoring.fallo * (1.0 - po)


def best_scoreline(matrix: np.ndarray, p_outcome: dict[str, float] | None = None,
                   scoring: Scoring = DEFAULT) -> dict:
    """Marcador que maximiza los puntos esperados (y su confianza)."""
    if p_outcome is None:
        p_outcome = poisson.outcome_probs(matrix)
    n = matrix.shape[0]
    best = None
    for i in range(n):
        for j in range(n):
            ep = expected_points(matrix, i, j, p_outcome, scoring)
            if best is None or ep > best["ep"]:
                best = {"score": (i, j), "ep": ep, "outcome": outcome_of(i, j),
                        "p_exact": float(matrix[i, j]), "p_outcome": float(p_outcome[outcome_of(i, j)])}
    return best


def most_likely_scoreline(matrix: np.ndarray) -> tuple[int, int]:
    """Marcador modal ('más repetido' = argmax de la matriz)."""
    idx = int(np.argmax(matrix))
    return divmod(idx, matrix.shape[1])


def evaluate_match(matrix: np.ndarray, p_outcome: dict[str, float] | None = None) -> dict:
    """Pick óptimo normal + pick óptimo all-in + marcador modal, para un partido."""
    if p_outcome is None:
        p_outcome = poisson.outcome_probs(matrix)
    normal = best_scoreline(matrix, p_outcome, DEFAULT)
    allin = best_scoreline(matrix, p_outcome, ALLIN)
    return {"normal": normal, "allin": allin,
            "modal": most_likely_scoreline(matrix),
            "p_outcome": p_outcome}


# --------------------------------------------------------------------------- #
# Comodines
# --------------------------------------------------------------------------- #
def allocate_comodines(match_evals: list[dict],
                       counts: dict[str, int] | None = None) -> dict:
    """Asigna comodines a los partidos para maximizar el total de puntos esperados.

    `match_evals`: lista de dicts con `normal["ep"]` y `allin["ep"]` (de `evaluate_match`).
    `counts`: nº disponible de cada comodín (def. 2 double, 2 triple, 2 all-in).
    Cada partido recibe como máximo un comodín (no se apilan).

    Devuelve `{assignments: {idx: tipo}, total_ep, baseline_ep}`.
    El "boost" de cada comodín sobre el pick normal:
        double → +ep ; triple → +2·ep ; all-in → ep_allin − ep
    """
    from scipy.optimize import linear_sum_assignment

    counts = counts or {"triple": 2, "double": 2, "allin": 2}
    normal_ep = np.array([m["normal"]["ep"] for m in match_evals], dtype=float)
    allin_ep = np.array([m["allin"]["ep"] for m in match_evals], dtype=float)
    baseline = float(normal_ep.sum())

    slots: list[str] = []
    for t in ("triple", "double", "allin"):
        slots += [t] * counts.get(t, 0)
    slots = slots[: len(match_evals)]  # no más comodines que partidos
    if not slots:
        return {"assignments": {}, "total_ep": baseline, "baseline_ep": baseline}

    def boost(t: str) -> np.ndarray:
        if t == "double":
            return normal_ep
        if t == "triple":
            return 2.0 * normal_ep
        return allin_ep - normal_ep  # all-in

    cost = -np.vstack([boost(t) for t in slots])     # (n_slots, n_matches), minimizar
    rows, cols = linear_sum_assignment(cost)

    assignments: dict[int, str] = {}
    total_boost = 0.0
    for r, c in zip(rows, cols):
        b = -cost[r, c]
        if b > 0:                       # solo aplicar si suma (boost positivo)
            assignments[int(c)] = slots[r]
            total_boost += b
    return {"assignments": assignments, "total_ep": baseline + total_boost,
            "baseline_ep": baseline}


# --------------------------------------------------------------------------- #
# Fixtures de fase de grupos (con anfitriones como locales)
# --------------------------------------------------------------------------- #
def _official_calendar() -> list[dict] | None:
    """Carga los 72 fixtures del Mundial 2026 desde el dataset Kaggle (que ya viene con el
    **calendario oficial** FIFA: fechas reales, orden home/away real, `neutral` correcto para
    anfitriones). Devuelve la lista ordenada por fecha, o None si no está disponible."""
    import pandas as pd
    from src import data_loader

    # Lectura directa: `load_results_raw` descarta marcadores NaN (los fixtures sin jugar).
    raw = pd.read_csv(data_loader.settings.RESULTS_CSV)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    wc = raw[(raw["tournament"] == "FIFA World Cup") & (raw["date"] >= "2026-06-11")
             & (raw["date"] <= "2026-06-27")].sort_values("date", kind="stable")
    if wc.empty or len(wc) != 72:
        return None
    team2group = {t: g for g, teams in wc2026.GROUPS.items() for t in teams}
    fixtures = []
    for r in wc.itertuples(index=False):
        if r.home_team not in team2group or r.away_team not in team2group:
            return None
        if team2group[r.home_team] != team2group[r.away_team]:
            return None
        neutral = str(r.neutral).strip().upper() in {"TRUE", "1", "T"}
        fixtures.append({"group": team2group[r.home_team], "home": r.home_team,
                         "away": r.away_team, "neutral": neutral,
                         "date": r.date.date().isoformat()})
    return fixtures


def group_fixtures(groups: dict[str, list[str]] | None = None) -> list[dict]:
    """Los 72 partidos de la fase de grupos. El anfitrión juega como local (neutral=False).

    Si `groups` son los oficiales (los 12 del sorteo), usa el **CALENDARIO OFICIAL FIFA**
    del dataset (orden cronológico real, home/away/neutral verificados). Para grupos custom
    cae a un fallback determinista por combinaciones (sin fecha).
    """
    groups = groups if groups is not None else wc2026.GROUPS
    if groups == wc2026.GROUPS:
        official = _official_calendar()
        if official is not None:
            return official
    # Fallback (configs no oficiales o si el calendario aún no está en el dataset).
    fixtures = []
    for gname, teams in groups.items():
        for a in range(len(teams)):
            for b in range(a + 1, len(teams)):
                t1, t2 = teams[a], teams[b]
                if wc2026.is_host(t1):
                    home, away, neutral = t1, t2, False
                elif wc2026.is_host(t2):
                    home, away, neutral = t2, t1, False
                else:
                    home, away, neutral = t1, t2, True
                fixtures.append({"group": gname, "home": home, "away": away, "neutral": neutral})
    return fixtures


# --------------------------------------------------------------------------- #
# Hito E — Pickers de futuros (usan las marginales del Montecarlo)
# --------------------------------------------------------------------------- #
def group_position_picks(df, groups: dict[str, list[str]] | None = None,
                         pts: tuple[float, float, float, float] = (4, 3, 2, 0)):
    """Para cada grupo, asigna equipos a {1º,2º,3º,4º} maximizando Σ pts·P(posición).

    `df` es la salida de `montecarlo.simulate` (necesita P_g1..P_g4).
    Devuelve un DataFrame con el pronóstico de cada grupo y sus puntos esperados.
    """
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    groups = groups if groups is not None else wc2026.GROUPS
    dfi = df.set_index("team")
    ptsv = np.array(pts, dtype=float)
    rows = []
    for g, teams in groups.items():
        Pmat = dfi.loc[teams, ["P_g1", "P_g2", "P_g3", "P_g4"]].to_numpy()  # (4 equipos, 4 pos)
        r, c = linear_sum_assignment(-(Pmat * ptsv))
        pos_team = {int(c[k]): teams[int(r[k])] for k in range(len(r))}
        exp = float(sum(ptsv[c[k]] * Pmat[r[k], c[k]] for k in range(len(r))))
        rows.append({"grupo": g, "1º": pos_team[0], "2º": pos_team[1],
                     "3º": pos_team[2], "4º": pos_team[3], "E_pts": round(exp, 2)})
    return pd.DataFrame(rows)


def best_thirds_picks(df, n: int = None, pts: float = 3.0):
    """Los `n` equipos con mayor prob. de ser 'mejor tercero' (3 pts c/u)."""
    import pandas as pd
    n = n or wc2026.THIRDS_QUALIFY
    top = df.nlargest(n, "P_best_third")[["team", "P_best_third"]].copy()
    top["E_pts"] = (top["P_best_third"] * pts).round(2)
    exp = float(top["E_pts"].sum())
    return top.reset_index(drop=True), round(exp, 2)


def placement_picks(df):
    """Asigna equipos distintos a campeón(10)/subcampeón(6)/3º(4)/4º(2) maximizando Σ pts·P."""
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    positions = [("Campeón", 10, "P_Campeon"), ("Subcampeón", 6, "P_subcampeon"),
                 ("3er puesto", 4, "P_3er_puesto"), ("4º puesto", 2, "P_4to_puesto")]
    teams = df["team"].tolist()
    M = np.array([df[col].to_numpy() for _, _, col in positions])   # (4 pos, N equipos)
    ptsv = np.array([p for _, p, _ in positions], dtype=float)
    r, c = linear_sum_assignment(-(M * ptsv[:, None]))
    rows = []
    for k in range(len(r)):
        name, p, _ = positions[r[k]]
        prob = float(M[r[k], c[k]])
        rows.append({"puesto": name, "pick": teams[c[k]],
                     "prob": round(prob, 3), "E_pts": round(p * prob, 2)})
    order = {"Campeón": 0, "Subcampeón": 1, "3er puesto": 2, "4º puesto": 3}
    return pd.DataFrame(sorted(rows, key=lambda x: order[x["puesto"]]))


def total_goals_pick(extras: dict) -> dict:
    """Estimación del total de goles del torneo (mediana) y rango p10-p90."""
    tg = extras["total_goals"]
    return {"estimacion": int(round(tg["median"])), "media": round(tg["mean"], 1),
            "p10": int(round(tg["p10"])), "p90": int(round(tg["p90"]))}


def recommend_group_stage(predictor, groups: dict[str, list[str]] | None = None) -> dict:
    """Recomendación completa de la fase de grupos: por cada partido el pick óptimo,
    su confianza, el marcador modal y el plan de comodines.

    Devuelve `{matches: [...], comodines: {...}}`.
    """
    import pandas as pd

    fixtures = group_fixtures(groups)
    evals = []
    rows = []
    for fx in fixtures:
        m = predictor.score_matrix_for(fx["home"], fx["away"], neutral=fx["neutral"],
                                       stage="group")
        ev = evaluate_match(m)
        evals.append(ev)
        s = ev["normal"]["score"]
        row = {"grupo": fx["group"]}
        if "date" in fx:
            row["fecha"] = fx["date"]
        row.update({
            "partido": f"{fx['home']} vs {fx['away']}",
            "pick": f"{s[0]}-{s[1]}", "modal": f"{ev['modal'][0]}-{ev['modal'][1]}",
            "resultado": {"H": fx["home"], "D": "Empate", "A": fx["away"]}[ev["normal"]["outcome"]],
            "E_pts": round(ev["normal"]["ep"], 3),
            "P_exacto": round(ev["normal"]["p_exact"], 3),
            "P_resultado": round(ev["normal"]["p_outcome"], 3),
        })
        rows.append(row)
    plan = allocate_comodines(evals)
    df = pd.DataFrame(rows)
    for idx, tipo in plan["assignments"].items():
        df.loc[idx, "comodin"] = tipo
    if "comodin" not in df:
        df["comodin"] = None
    return {"matches": df, "comodines": plan, "fixtures": fixtures}
