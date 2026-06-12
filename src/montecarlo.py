"""Nivel 4 — Simulación de Montecarlo del Mundial.

Simula el torneo completo `N` veces (vectorizado con numpy) para estimar la probabilidad
de que cada selección supere la fase de grupos, llegue a cada ronda y sea campeona.

Modelo de partido: los goles se derivan de la **fuerza ajustada al rival** (Elo + ancla
FIFA), NO del ataque/defensa Poisson crudo (que sobrevalora a quien golea a selecciones
débiles). Para un partido A-B en sede neutral:

    ventaja   = slope · (fuerza_A − fuerza_B)        # goles de supremacía
    λ_A = (goles_base + ventaja) / 2 ,  λ_B = (goles_base − ventaja) / 2   (λ ≥ λ_min)
    goles ~ Poisson(λ)

`goles_base` se escala por `GOAL_STAGE_FACTOR` según la fase (grupos 1.0, eliminatorias 0.92:
los partidos a eliminación directa suelen ser menos goleadores).

En grupos se acumulan puntos y dif. de goles (desempates puntos > dif. > goles a favor);
en eliminatorias un empate se resuelve por penaltis (moneda al aire). Cuadro: 1º y 2º de
cada grupo + los 8 mejores terceros -> 32 (R32), armado con el **BRACKET OFICIAL 2026**
(cada plaza por posición de grupo; los terceros se asignan a su plaza con la tabla oficial,
ver `config/wc2026`). Configuraciones de grupos no oficiales caen a siembra por rendimiento.
"""
from __future__ import annotations

import functools
from itertools import combinations

import numpy as np
import pandas as pd

from config import settings, wc2026
from src import elo

STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]
_WINNER_STAGE = {16: "R16", 8: "QF", 4: "SF", 2: "Final", 1: "Champion"}


def bracket_seed_order(n: int) -> list[int]:
    """Orden de siembra estándar (1..n) en posiciones del cuadro, de modo que los
    cabezas de serie 1 y 2 solo puedan cruzarse en la final. Ej. n=4 -> [1,4,2,3]."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


@functools.lru_cache(maxsize=1)
def _third_assign_array():
    """`(arr, positions)` para asignar terceros a sus plazas en el cuadro oficial, vectorizado.

    `arr` (2^12, 8) int: indexado por la MÁSCARA de grupos cuyos terceros clasifican (bit i =
    grupo `wc2026.GROUP_NAMES[i]`), da la COLUMNA de grupo (0..11) que ocupa cada plaza de
    tercero; -1 si la máscara no corresponde a 8 terceros válidos (no se consulta nunca).
    `positions` = posiciones 0..31 de esas plazas en el cuadro.
    """
    g2col = {g: i for i, g in enumerate(wc2026.GROUP_NAMES)}
    positions = wc2026.third_slot_positions()
    arr = np.full((1 << wc2026.N_GROUPS, len(positions)), -1, dtype=np.int64)
    for combo, slot_groups in wc2026.third_assignment_table().items():
        mask = 0
        for g in combo:
            mask |= 1 << g2col[g]
        arr[mask] = [g2col[g] for g in slot_groups]
    return arr, np.array(positions)


def _build_official_bracket(winners, runners, thirds, th_order, group_names):
    """Cuadro (N,32) en ORDEN OFICIAL: cada plaza 1X/2X la ocupa el ganador/subcampeón del
    grupo X; las 8 plazas de tercero se asignan según `wc2026.third_assignment_table`.

    `winners/runners/thirds` son (N,12) con columnas en orden `group_names`; `th_order` (N,8)
    son las COLUMNAS de grupo de los 8 mejores terceros de cada simulación.
    """
    g2col = {g: i for i, g in enumerate(group_names)}
    flat = wc2026.bracket_r32_flat()
    N = winners.shape[0]
    cur = np.empty((N, 32), dtype=winners.dtype)
    for p, tok in enumerate(flat):
        if tok[0] == "1":
            cur[:, p] = winners[:, g2col[tok[1:]]]
        elif tok[0] == "2":
            cur[:, p] = runners[:, g2col[tok[1:]]]
        # "3:" (terceros) se rellena abajo.
    assign_arr, positions = _third_assign_array()
    masks = np.zeros(N, dtype=np.int64)
    for k in range(th_order.shape[1]):
        masks |= np.int64(1) << th_order[:, k].astype(np.int64)
    slot_cols = assign_arr[masks]                                 # (N,8) columnas de grupo
    third_teams = np.take_along_axis(thirds, slot_cols, axis=1)   # (N,8) idx de equipo
    for k, p in enumerate(positions):
        cur[:, p] = third_teams[:, k]
    return cur


def load_strength() -> dict[str, float]:
    """Fuerza ajustada (columna `strength` de models/elo_ratings.csv, Elo+FIFA)."""
    df = elo.load_ratings()
    if df is None:
        raise FileNotFoundError("Falta models/elo_ratings.csv. Ejecuta scripts/train.py")
    col = "strength" if "strength" in df.columns else "elo"
    return dict(zip(df["team"], df[col]))


def simulate(n_sims: int = settings.N_SIMULACIONES,
             groups: dict[str, list[str]] | None = None,
             strength: dict[str, float] | None = None,
             slope: float = settings.GOAL_SLOPE_PER_ELO,
             base_total: float = settings.GOAL_BASE_TOTAL,
             lam_min: float = settings.GOAL_LAMBDA_MIN,
             seed: int = 0, return_extras: bool = False):
    """Corre la simulación y devuelve un DataFrame por selección con probabilidades de:
    posición de grupo (P_g1..P_g4), mejor tercero (P_best_third), avance (P_R32..P_Final),
    y placement final (P_Campeon, P_subcampeon, P_3er_puesto, P_4to_puesto).

    Si `return_extras=True`, devuelve `(df, extras)` con `extras["total_goals"]`
    (distribución del total de goles del torneo) y `extras["total_goals_samples"]`.
    """
    groups = groups if groups is not None else wc2026.GROUPS
    strength = strength if strength is not None else load_strength()

    group_names = list(groups)
    teams = sorted({t for g in groups.values() for t in g})
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    mean_s = float(np.mean(list(strength.values()))) if strength else settings.ELO_INITIAL
    S = np.array([strength.get(t, mean_s) for t in teams])
    # Ventaja de anfitrión: solo en fase de grupos (juegan en casa).
    S_group = S.copy()
    host_adv = settings.ELO_HOME_ADVANTAGE
    for t in wc2026.HOSTS:
        if t in idx:
            S_group[idx[t]] += host_adv
    rng = np.random.default_rng(seed)
    N = n_sims

    # Goles base por fase: las eliminatorias son menos goleadoras (GOAL_STAGE_FACTOR).
    base_group = base_total * settings.GOAL_STAGE_FACTOR.get("group", 1.0)
    base_ko = base_total * settings.GOAL_STAGE_FACTOR.get("knockout", 1.0)

    def lam(Sarr, a, b, base):  # tasa de goles de a frente a b según fuerza (sede neutral)
        return np.clip((base + slope * (Sarr[a] - Sarr[b])) / 2.0, lam_min, None)

    counts = {s: np.zeros(n_teams) for s in STAGES}
    pos_counts = np.zeros((4, n_teams))       # 1º/2º/3º/4º de grupo
    best_third_count = np.zeros(n_teams)
    sub_count = np.zeros(n_teams)
    tercero_count = np.zeros(n_teams)
    cuarto_count = np.zeros(n_teams)
    total_goals = np.zeros(N)

    # ---------------- Fase de grupos ----------------
    pairs = list(combinations(range(4), 2))
    win_idx, run_idx, win_key, run_key = [], [], [], []
    third_idx, third_key = [], []

    for gname in group_names:
        gi = np.array([idx[t] for t in groups[gname]])
        P = np.zeros((N, 4)); GD = np.zeros((N, 4)); GF = np.zeros((N, 4))
        for i, j in pairs:
            ga = rng.poisson(lam(S_group, gi[i], gi[j], base_group), N)
            gb = rng.poisson(lam(S_group, gi[j], gi[i], base_group), N)
            total_goals += ga + gb
            tie = ga == gb
            P[:, i] += np.where(ga > gb, 3, np.where(tie, 1, 0))
            P[:, j] += np.where(gb > ga, 3, np.where(tie, 1, 0))
            GD[:, i] += ga - gb; GD[:, j] += gb - ga
            GF[:, i] += ga; GF[:, j] += gb

        key = P * 1e6 + (GD + 100) * 1e3 + GF + rng.random((N, 4)) * 1e-2
        order = np.argsort(-key, axis=1)
        gi_order = gi[order]
        key_order = np.take_along_axis(key, order, axis=1)
        for p in range(4):
            np.add.at(pos_counts[p], gi_order[:, p], 1)

        win_idx.append(gi_order[:, 0]); win_key.append(key_order[:, 0])
        run_idx.append(gi_order[:, 1]); run_key.append(key_order[:, 1])
        third_idx.append(gi_order[:, 2]); third_key.append(key_order[:, 2])

    winners = np.stack(win_idx, axis=1); runners = np.stack(run_idx, axis=1)
    thirds = np.stack(third_idx, axis=1)
    w_key = np.stack(win_key, axis=1); r_key = np.stack(run_key, axis=1)
    t_key = np.stack(third_key, axis=1)

    th_order = np.argsort(-t_key, axis=1)[:, :wc2026.THIRDS_QUALIFY]
    best_thirds = np.take_along_axis(thirds, th_order, axis=1)        # (N,8)
    best_thirds_key = np.take_along_axis(t_key, th_order, axis=1)
    np.add.at(best_third_count, best_thirds.ravel(), 1)

    qualifiers = np.concatenate([winners, runners, best_thirds], axis=1)   # (N,32)
    qual_key = np.concatenate([w_key, r_key, best_thirds_key], axis=1)
    np.add.at(counts["R32"], qualifiers.ravel(), 1)

    # ---------------- Eliminatorias: armar el cuadro ----------------
    if group_names == wc2026.GROUP_NAMES:        # cuadro OFICIAL 2026 (12 grupos A..L)
        cur = _build_official_bracket(winners, runners, thirds, th_order, group_names)
    else:                                        # fallback: siembra por rendimiento
        seed_order = np.argsort(-qual_key, axis=1)
        seeded = np.take_along_axis(qualifiers, seed_order, axis=1)
        bp = [s - 1 for s in bracket_seed_order(32)]
        cur = seeded[:, bp]

    size = 32
    sf_losers = None
    while size > 1:
        a, b = cur[:, 0::2], cur[:, 1::2]
        ga = rng.poisson(lam(S, a, b, base_ko))
        gb = rng.poisson(lam(S, b, a, base_ko))
        total_goals += (ga + gb).sum(axis=1)
        tie = ga == gb
        coin = rng.random(a.shape) < 0.5
        a_wins = np.where(tie, coin, ga > gb)
        winners_k = np.where(a_wins, a, b)
        losers_k = np.where(a_wins, b, a)
        cur = winners_k
        size //= 2
        np.add.at(counts[_WINNER_STAGE[size]], cur.ravel(), 1)
        if size == 2:                 # acabamos de jugar semifinales (4 -> 2)
            sf_losers = losers_k      # (N,2) perdedores de semis
        elif size == 1:               # acabamos de jugar la final (2 -> 1)
            np.add.at(sub_count, losers_k.ravel(), 1)   # subcampeón

    # ---------------- Repechaje por el 3er puesto ----------------
    a3, b3 = sf_losers[:, 0], sf_losers[:, 1]
    ga3 = rng.poisson(lam(S, a3, b3, base_ko)); gb3 = rng.poisson(lam(S, b3, a3, base_ko))
    total_goals += ga3 + gb3
    tie3 = ga3 == gb3
    a3_wins = np.where(tie3, rng.random(N) < 0.5, ga3 > gb3)
    np.add.at(tercero_count, np.where(a3_wins, a3, b3), 1)
    np.add.at(cuarto_count, np.where(a3_wins, b3, a3), 1)

    # ---------------- Resultados ----------------
    df = pd.DataFrame({"team": teams})
    df["P_grupo"] = pos_counts[0] / N          # = P_g1 (gana el grupo)
    df["P_g1"] = pos_counts[0] / N
    df["P_g2"] = pos_counts[1] / N
    df["P_g3"] = pos_counts[2] / N
    df["P_g4"] = pos_counts[3] / N
    df["P_best_third"] = best_third_count / N
    for s, col in zip(STAGES, ["P_R32", "P_R16", "P_QF", "P_SF", "P_Final", "P_Campeon"]):
        df[col] = counts[s] / N
    df["P_subcampeon"] = sub_count / N
    df["P_3er_puesto"] = tercero_count / N
    df["P_4to_puesto"] = cuarto_count / N
    df = df.sort_values("P_Campeon", ascending=False).reset_index(drop=True)

    if return_extras:
        tg = total_goals
        extras = {
            "total_goals": {
                "mean": float(tg.mean()), "median": float(np.median(tg)),
                "std": float(tg.std()), "p10": float(np.percentile(tg, 10)),
                "p90": float(np.percentile(tg, 90)),
            },
            "total_goals_samples": tg,
        }
        return df, extras
    return df
