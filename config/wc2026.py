"""Formato y cuadro del Mundial 2026 (48 equipos, 12 grupos de 4) — SORTEO OFICIAL.

Grupos según el sorteo oficial (ESPN/FIFA). Los nombres son los canónicos del dataset
Kaggle (p. ej. "United States", "South Korea", "Ivory Coast", "Czech Republic"), ya
verificados (cobertura 48/48 contra dataset, FIFA y fuerza).

Clasifican a Dieciseisavos (R32): 1º y 2º de cada grupo (24) + los 8 mejores terceros.

Anfitriones (`HOSTS`): México, Canadá y EE. UU. juegan sus partidos como LOCALES (no en
sede neutral) → ventaja de localía real, señal útil sin alineaciones.
"""
from __future__ import annotations

import functools
from itertools import combinations

# --------------------------------------------------------------------------- #
# Sorteo oficial
# --------------------------------------------------------------------------- #
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

GROUP_NAMES = list(GROUPS.keys())  # A..L
N_GROUPS = len(GROUPS)             # 12
THIRDS_QUALIFY = 8                 # mejores terceros que pasan a R32

# Anfitriones: juegan como locales (no neutral).
HOSTS: set[str] = {"Mexico", "Canada", "United States"}


def is_host(team: str) -> bool:
    return team in HOSTS


def all_participants(groups: dict[str, list[str]] | None = None) -> list[str]:
    groups = groups or GROUPS
    return sorted({t for teams in groups.values() for t in teams})


def group_of(team: str) -> str | None:
    for name, teams in GROUPS.items():
        if team in teams:
            return name
    return None


# --------------------------------------------------------------------------- #
# Cuadro de eliminatorias — BRACKET OFICIAL 2026 (FIFA)
# --------------------------------------------------------------------------- #
# Cada entrada = un partido de Dieciseisavos (R32) en ORDEN DE CUADRO: emparejar los
# ganadores de partidos adyacentes reproduce R16, luego cuartos, semis y la final, igual
# que el árbol oficial. Tokens de plaza:
#   "1X" / "2X" -> 1º / 2º del grupo X.
#   "3:XYZ..."  -> el tercero CLASIFICADO procedente de UNO de esos grupos (la plaza exacta
#                  depende de cuáles 8 terceros pasen; ver `third_assignment_table`).
# El comentario al lado es el ID oficial de partido (P73..P88) del cuadro publicado.
BRACKET_R32: list[tuple[str, str]] = [
    ("1E", "3:ABCDF"),   # P74
    ("1I", "3:CDFGH"),   # P77
    ("2A", "2B"),        # P73
    ("1F", "2C"),        # P75
    ("2K", "2L"),        # P83
    ("1H", "2J"),        # P84
    ("1D", "3:BEFIJ"),   # P81
    ("1G", "3:AEHIJ"),   # P82
    ("1C", "2F"),        # P76
    ("2E", "2I"),        # P78
    ("1A", "3:CEFHI"),   # P79
    ("1L", "3:EHIJK"),   # P80
    ("1J", "2H"),        # P86
    ("2D", "2G"),        # P88
    ("1B", "3:EFGIJ"),   # P85
    ("1K", "3:DEIJL"),   # P87
]
# IDs oficiales (P-número) de cada partido de R32, alineados con BRACKET_R32.
BRACKET_R32_IDS: list[int] = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]


def bracket_r32_flat() -> list[str]:
    """Las 32 plazas de R32 en orden de cuadro (2 por partido, aplanadas)."""
    return [tok for match in BRACKET_R32 for tok in match]


def _third_slots() -> list[tuple[int, frozenset[str]]]:
    """(posición 0..31 en el cuadro, grupos elegibles) de cada plaza de tercero."""
    flat = bracket_r32_flat()
    return [(p, frozenset(tok[2:])) for p, tok in enumerate(flat) if tok.startswith("3:")]


def third_slot_positions() -> list[int]:
    """Posiciones (0..31) de las 8 plazas de tercero, en orden de cuadro."""
    return [p for p, _ in _third_slots()]


def _match_thirds(eligible: list[frozenset[str]], qualified: frozenset[str]) -> list[str] | None:
    """Empareja los 8 grupos cuyos terceros clasifican con las 8 plazas respetando la
    elegibilidad del cuadro (matching bipartito por caminos aumentantes, determinista).
    Devuelve el grupo que ocupa cada plaza (en orden de `eligible`) o None si no hay matching.
    """
    match_group: dict[str, int] = {}   # grupo -> índice de plaza asignada
    def assign(slot: int, seen: set[str]) -> bool:
        for g in sorted(eligible[slot] & qualified):
            if g in seen:
                continue
            seen.add(g)
            if g not in match_group or assign(match_group[g], seen):
                match_group[g] = slot
                return True
        return False
    for s in range(len(eligible)):
        if not assign(s, set()):
            return None
    slot_group: list[str | None] = [None] * len(eligible)
    for g, s in match_group.items():
        slot_group[s] = g
    return [g for g in slot_group]  # type: ignore[misc]


@functools.lru_cache(maxsize=1)
def third_assignment_table() -> dict[frozenset[str], tuple[str, ...]]:
    """Para CADA combinación de 8 grupos cuyos terceros clasifican (C(12,8)=495), qué grupo
    ocupa cada plaza de tercero, en el orden de `third_slot_positions`.

    Asignación **determinista y válida** según los conjuntos elegibles del cuadro oficial
    (no necesariamente el desempate interno exacto de la tabla FIFA entre matchings
    equivalentes; el efecto sobre las probabilidades por selección es de segundo orden).
    """
    eligible = [elig for _, elig in _third_slots()]
    table: dict[frozenset[str], tuple[str, ...]] = {}
    for combo in combinations(GROUP_NAMES, THIRDS_QUALIFY):
        q = frozenset(combo)
        sg = _match_thirds(eligible, q)
        if sg is not None:
            table[q] = tuple(sg)
    return table


# Snapshot FIFA manual de respaldo (solo si la descarga online fallara).
FIFA_FALLBACK: dict[str, float] = {}
