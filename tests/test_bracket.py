"""Tests del Hito #4 — eliminatorias por ronda (`src/bracket.py`)."""
import pandas as pd
import pytest

from config import wc2026
from src import bracket, poisson


GN = list(wc2026.GROUP_NAMES)
# Grupos sintéticos con etiquetas trazables: grupo g -> [g1, g2, g3, g4].
SYNTH_GROUPS = {g: [f"{g}{k}" for k in (1, 2, 3, 4)] for g in GN}


def _round_robin_matches(home_wins_by_rank=True):
    """72 partidos: en cada grupo, el equipo de rango i (g{i}) gana a los de rango mayor 1-0,
    así el standings de cada grupo es [g1, g2, g3, g4] y todos los terceros empatan."""
    rows = []
    for g in GN:
        t = [f"{g}{k}" for k in (1, 2, 3, 4)]
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append({"home_team": t[i], "away_team": t[j],
                             "home_score": 1, "away_score": 0})
    return pd.DataFrame(rows)


class StubPredictor:
    """Predictor mínimo: matriz de marcador por fuerza (sin reconciliar). Más fuerte -> más λ."""
    def __init__(self, strength):
        self.S = strength

    def score_matrix_for(self, home, away, *, neutral=True, stage="knockout"):
        sup = 0.01 * (self.S[home] - self.S[away])
        lh = max((2.6 + sup) / 2, 0.15)
        la = max((2.6 - sup) / 2, 0.15)
        return poisson.score_matrix(lh, la, 0.0, 6)


def _labelled_strength():
    # g1 > g2 > g3 (por rango) y los grupos "tempranos" algo más fuertes -> avances deterministas.
    return {f"{g}{k}": 1500 + (len(GN) - GN.index(g)) * 6 + (4 - k) * 20
            for g in GN for k in (1, 2, 3, 4)}


# --------------------------------------------------------------------------- #
# Standings
# --------------------------------------------------------------------------- #
def test_group_standings_ordena_por_pts_dif_gf():
    groups = {"A": ["T1", "T2", "T3", "T4"]}
    matches = pd.DataFrame([
        {"home_team": "T1", "away_team": "T2", "home_score": 2, "away_score": 0},
        {"home_team": "T1", "away_team": "T3", "home_score": 2, "away_score": 0},
        {"home_team": "T1", "away_team": "T4", "home_score": 2, "away_score": 0},
        {"home_team": "T2", "away_team": "T3", "home_score": 1, "away_score": 0},
        {"home_team": "T2", "away_team": "T4", "home_score": 1, "away_score": 0},
        {"home_team": "T3", "away_team": "T4", "home_score": 1, "away_score": 0},
    ])
    assert bracket.group_standings(matches, groups)["A"] == ["T1", "T2", "T3", "T4"]


def test_standings_from_matches_clasifica_8_terceros():
    w, r, th, qual = bracket.standings_from_matches(_round_robin_matches(), SYNTH_GROUPS)
    assert [w["A"], r["A"], th["A"]] == ["A1", "A2", "A3"]
    assert len(qual) == wc2026.THIRDS_QUALIFY == 8
    # con todos los terceros empatados, clasifican los 8 primeros grupos por orden estable.
    assert qual == set(GN[:8])


# --------------------------------------------------------------------------- #
# Resolución del cuadro
# --------------------------------------------------------------------------- #
def test_resolve_r32_resuelve_plazas_y_sin_choque():
    winners = {g: f"{g}1" for g in GN}
    runners = {g: f"{g}2" for g in GN}
    thirds = {g: f"{g}3" for g in GN}
    qualified = set(GN[:8])
    mm = bracket.resolve_r32(winners, runners, thirds, qualified)
    assert len(mm) == 16
    teams32 = [t for cruce in mm for t in cruce]
    assert len(set(teams32)) == 32                       # 32 equipos distintos
    flat = wc2026.bracket_r32_flat()
    for p, tok in enumerate(flat):
        if tok[0] == "1":
            assert teams32[p] == f"{tok[1:]}1"
        elif tok[0] == "2":
            assert teams32[p] == f"{tok[1:]}2"
        else:                                            # tercero: grupo elegible y clasificado
            grp = teams32[p][:-1]
            assert teams32[p].endswith("3") and grp in set(tok[2:]) and grp in qualified
    for h, a in mm:                                      # ningún cruce del mismo grupo
        assert h[:-1] != a[:-1]


def test_resolve_r32_exige_8_grupos():
    winners = {g: f"{g}1" for g in GN}
    runners = {g: f"{g}2" for g in GN}
    thirds = {g: f"{g}3" for g in GN}
    with pytest.raises(ValueError):
        bracket.resolve_r32(winners, runners, thirds, set(GN[:5]))


def test_advance_empareja_adyacentes():
    adv = [f"m{i}" for i in range(16)]
    pairs = bracket.advance(adv)
    assert len(pairs) == 8
    assert pairs[0] == ("m0", "m1") and pairs[3] == ("m6", "m7")


# --------------------------------------------------------------------------- #
# Predicción batch de una ronda
# --------------------------------------------------------------------------- #
def test_recommend_round_estructura_y_avance():
    sp = StubPredictor(_labelled_strength())
    matchups = [("A1", "B2"), ("C1", "D2")]
    out = bracket.recommend_round(sp, matchups)
    df = out["matches"]
    assert len(df) == 2
    assert {"partido", "pick", "avanza", "P_avanza", "E_pts"}.issubset(df.columns)
    # el más fuerte de cada cruce avanza (P_avanza >= 0.5).
    assert out["advancers"][0] in ("A1", "B2")
    assert (df["P_avanza"] >= 0.5).all()


def test_recommend_round_comodines_opcionales():
    sp = StubPredictor(_labelled_strength())
    matchups = [("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2")]
    out = bracket.recommend_round(sp, matchups, comodines={"triple": 1, "double": 1})
    assert "comodines" in out
    assert "comodin" in out["matches"].columns
    assert out["matches"]["comodin"].notna().sum() == 2       # se asignaron 2 comodines


def test_project_rounds_encadena_hasta_la_final():
    sp = StubPredictor(_labelled_strength())
    w, r, th, qual = bracket.standings_from_matches(_round_robin_matches(), SYNTH_GROUPS)
    rounds = bracket.project_rounds(sp, w, r, th, qual)
    assert [len(rounds[x]) for x in bracket.KO_ROUNDS] == [16, 8, 4, 2, 1]
    # los equipos de cada ronda son subconjunto de la anterior.
    r32 = {t for cruce in rounds["R32"] for t in cruce}
    r16 = {t for cruce in rounds["R16"] for t in cruce}
    assert r16.issubset(r32) and len(r16) == 16
