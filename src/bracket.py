"""Hito #4 — Eliminatorias por ronda: resolución del cuadro oficial y avance.

Convierte las plazas del **BRACKET OFICIAL 2026** (`config/wc2026`) en cruces concretos de
selección, a partir de unos standings de grupo que pueden ser:
  - **reales** (de los partidos de grupos ya jugados), o
  - **proyectados** (de las marginales del Montecarlo).

Luego avanza ronda a ronda emparejando los ganadores adyacentes (igual que el árbol) y entrega
la **tabla de marcadores óptimos** de una ronda completa (espejo de `quiniela.recommend_group_stage`).

Decisiones (confirmadas por el usuario):
  - Marcador de eliminatorias **a 90'**: un empate es un pick válido; el "avanza" usa
    `P(gana) + ½·P(empate)` (penaltis ≈ moneda al aire).
  - Comodines en eliminatorias: **soportados y opcionales** (por si las reglas los permiten).
  - Sede **neutral** en todas las eliminatorias (la ventaja de anfitrión solo se aplica en grupos;
    para un partido concreto, usa el ajuste manual de fuerza del `Predictor`).
"""
from __future__ import annotations

import pandas as pd

from config import wc2026
from src import quiniela as q

# Rondas del cuadro, de R32 a la final.
KO_ROUNDS = ["R32", "R16", "QF", "SF", "Final"]


# --------------------------------------------------------------------------- #
# Standings de grupo
# --------------------------------------------------------------------------- #
def _team_metrics(matches: pd.DataFrame, teams: list[str]) -> dict[str, tuple[int, int, int]]:
    """(puntos, dif. de goles, goles a favor) de cada equipo dentro de su grupo, contando
    solo los partidos **entre miembros del grupo** con marcador. 3/1/0 por victoria/empate."""
    s = set(teams)
    pts = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    for r in matches.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if h in s and a in s and pd.notna(r.home_score) and pd.notna(r.away_score):
            hs, as_ = int(r.home_score), int(r.away_score)
            gf[h] += hs; gf[a] += as_
            gd[h] += hs - as_; gd[a] += as_ - hs
            if hs > as_:
                pts[h] += 3
            elif hs < as_:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
    return {t: (pts[t], gd[t], gf[t]) for t in teams}


def group_standings(matches: pd.DataFrame,
                    groups: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Clasificación 1º..4º de cada grupo (desempate pts > dif. > goles a favor; final estable
    por el orden del grupo). `matches` necesita columnas home_team/away_team/home_score/away_score.
    Si la fase está incompleta, devuelve la mejor ordenación con lo jugado.
    """
    groups = groups if groups is not None else wc2026.GROUPS
    out: dict[str, list[str]] = {}
    for g, teams in groups.items():
        met = _team_metrics(matches, teams)
        order = sorted(range(len(teams)),
                       key=lambda i: (met[teams[i]][0], met[teams[i]][1], met[teams[i]][2], -i),
                       reverse=True)
        out[g] = [teams[i] for i in order]
    return out


# --------------------------------------------------------------------------- #
# Standings -> (ganadores, subcampeones, terceros, grupos de terceros clasificados)
# --------------------------------------------------------------------------- #
def standings_from_matches(matches: pd.DataFrame, groups: dict[str, list[str]] | None = None):
    """Modo REAL: deriva ganadores/subcampeones/terceros y los 8 grupos cuyos terceros
    clasifican (los 8 mejores por pts > dif. > goles a favor entre los 12 terceros)."""
    groups = groups if groups is not None else wc2026.GROUPS
    standings = group_standings(matches, groups)
    winners = {g: standings[g][0] for g in groups}
    runners = {g: standings[g][1] for g in groups}
    thirds = {g: standings[g][2] for g in groups}
    met = {g: _team_metrics(matches, groups[g])[thirds[g]] for g in groups}
    ranked = sorted(groups, key=lambda g: (met[g][0], met[g][1], met[g][2]), reverse=True)
    qualified = set(ranked[:wc2026.THIRDS_QUALIFY])
    return winners, runners, thirds, qualified


def standings_from_projection(df, groups: dict[str, list[str]] | None = None):
    """Modo PROYECTADO: usa las marginales del Montecarlo (`montecarlo.simulate`). Ganadores/
    subcampeones/terceros = pick óptimo por posición (`quiniela.group_position_picks`); los 8
    grupos clasificados = los de mayor `P_best_third` entre sus terceros proyectados."""
    groups = groups if groups is not None else wc2026.GROUPS
    gp = q.group_position_picks(df, groups)
    winners = dict(zip(gp["grupo"], gp["1º"]))
    runners = dict(zip(gp["grupo"], gp["2º"]))
    thirds = dict(zip(gp["grupo"], gp["3º"]))
    pbt = df.set_index("team")["P_best_third"]
    score = {g: float(pbt.get(thirds[g], 0.0)) for g in thirds}
    qualified = set(sorted(score, key=lambda g: score[g], reverse=True)[:wc2026.THIRDS_QUALIFY])
    return winners, runners, thirds, qualified


# --------------------------------------------------------------------------- #
# Resolución del cuadro y avance
# --------------------------------------------------------------------------- #
def resolve_r32(winners: dict[str, str], runners: dict[str, str], thirds: dict[str, str],
                qualified: set[str]) -> list[tuple[str, str]]:
    """Los 16 cruces de R32 `(local, visitante)` en ORDEN DE CUADRO, resolviendo cada plaza:
    `1X`->ganador del grupo X, `2X`->subcampeón, `3:...`->tercero del grupo que la tabla oficial
    asigna a esa plaza según el conjunto de 8 grupos clasificados."""
    qualified = frozenset(qualified)
    table = wc2026.third_assignment_table()
    if qualified not in table:
        raise ValueError(f"Se requieren exactamente {wc2026.THIRDS_QUALIFY} grupos de terceros "
                         f"clasificados; recibidos {len(qualified)}.")
    pos_to_group = dict(zip(wc2026.third_slot_positions(), table[qualified]))
    flat = wc2026.bracket_r32_flat()
    teams32: list[str] = []
    for p, tok in enumerate(flat):
        if tok[0] == "1":
            teams32.append(winners[tok[1:]])
        elif tok[0] == "2":
            teams32.append(runners[tok[1:]])
        else:
            teams32.append(thirds[pos_to_group[p]])
    return [(teams32[2 * i], teams32[2 * i + 1]) for i in range(len(teams32) // 2)]


def advance(advancers: list[str]) -> list[tuple[str, str]]:
    """Empareja ganadores ADYACENTES (en orden de cuadro) -> cruces de la siguiente ronda."""
    return [(advancers[2 * i], advancers[2 * i + 1]) for i in range(len(advancers) // 2)]


# --------------------------------------------------------------------------- #
# Predicción batch de una ronda
# --------------------------------------------------------------------------- #
def recommend_round(predictor, matchups: list[tuple[str, str]], *, stage: str = "knockout",
                    comodines: dict[str, int] | None = None) -> dict:
    """Tabla de marcadores óptimos (a 90') de una ronda. Espejo de `recommend_group_stage`.

    Devuelve `{matches: df, advancers: [...], comodines?: {...}}`. `advancers` = quién avanza
    según `P(gana) + ½·P(empate)`. Si `comodines` (dict de cuentas) se pasa, asigna comodines.
    """
    rows, evals, advancers = [], [], []
    for home, away in matchups:
        m = predictor.score_matrix_for(home, away, neutral=True, stage=stage)
        ev = q.evaluate_match(m)
        evals.append(ev)
        po = ev["p_outcome"]
        p_home_adv = po["H"] + 0.5 * po["D"]
        advancer = home if p_home_adv >= 0.5 else away
        advancers.append(advancer)
        n_, al = ev["normal"], ev["allin"]
        s = n_["score"]
        res = {"H": home, "D": "Empate (90')", "A": away}[n_["outcome"]]
        rows.append({
            "partido": f"{home} vs {away}",
            "pick": f"{s[0]}-{s[1]}",
            "resultado": res,
            "avanza": advancer,
            "P_avanza": round(max(p_home_adv, 1 - p_home_adv), 3),
            "E_pts": round(n_["ep"], 3),
            "P_exacto": round(n_["p_exact"], 3),
            "P_resultado": round(n_["p_outcome"], 3),
            "modal": f"{ev['modal'][0]}-{ev['modal'][1]}",
            "allin": f"{al['score'][0]}-{al['score'][1]}",
        })
    out = {"matches": pd.DataFrame(rows), "advancers": advancers}
    if comodines is not None:
        plan = q.allocate_comodines(evals, comodines)
        out["comodines"] = plan
        for idx, tipo in plan["assignments"].items():
            out["matches"].loc[idx, "comodin"] = tipo
        if "comodin" not in out["matches"]:
            out["matches"]["comodin"] = None
    return out


def project_rounds(predictor, winners: dict[str, str], runners: dict[str, str],
                   thirds: dict[str, str], qualified: set[str], *,
                   stage: str = "knockout") -> dict[str, list[tuple[str, str]]]:
    """Resuelve R32 desde los standings y **avanza** con el pronóstico del modelo hasta la final.
    Devuelve `{ronda: [(local, visitante)]}` para R32..Final (las rondas posteriores a R32 son
    proyectadas, ya que aún no se han jugado)."""
    rounds: dict[str, list[tuple[str, str]]] = {}
    cur = resolve_r32(winners, runners, thirds, qualified)
    for rnd in KO_ROUNDS:
        rounds[rnd] = cur
        if rnd == "Final":
            break
        rec = recommend_round(predictor, cur, stage=stage)
        cur = advance(rec["advancers"])
    return rounds
