"""Tests del Nivel 4 (Montecarlo)."""
import numpy as np
import pytest

from config import wc2026
from src import montecarlo as mc


def test_bracket_seed_order_4():
    assert mc.bracket_seed_order(4) == [1, 4, 2, 3]


def test_bracket_seed_order_es_permutacion():
    for n in (8, 16, 32):
        order = mc.bracket_seed_order(n)
        assert sorted(order) == list(range(1, n + 1))


def _strength(teams, overrides=None):
    s = {t: 1700.0 for t in teams}      # todos iguales por defecto
    if overrides:
        s.update(overrides)
    return s


def test_simulate_invariantes_de_probabilidad():
    teams = wc2026.all_participants()
    df = mc.simulate(n_sims=400, strength=_strength(teams), seed=1)
    assert len(df) == 48
    # Cada simulación produce exactamente: 32 clasifican, 16 a R16, ... 1 campeón.
    assert df["P_R32"].sum() == pytest.approx(32.0)
    assert df["P_R16"].sum() == pytest.approx(16.0)
    assert df["P_QF"].sum() == pytest.approx(8.0)
    assert df["P_SF"].sum() == pytest.approx(4.0)
    assert df["P_Final"].sum() == pytest.approx(2.0)
    assert df["P_Campeon"].sum() == pytest.approx(1.0)
    assert df["P_grupo"].sum() == pytest.approx(12.0)   # 12 ganadores de grupo
    # posiciones de grupo, mejores terceros y placements
    for col in ["P_g1", "P_g2", "P_g3", "P_g4"]:
        assert df[col].sum() == pytest.approx(12.0)
    assert df["P_best_third"].sum() == pytest.approx(8.0)
    for col in ["P_subcampeon", "P_3er_puesto", "P_4to_puesto"]:
        assert df[col].sum() == pytest.approx(1.0)
    for col in df.columns[1:]:
        assert ((df[col] >= 0) & (df[col] <= 1)).all()


def test_simulate_total_goles_extras():
    teams = wc2026.all_participants()
    df, extras = mc.simulate(n_sims=300, strength=_strength(teams), seed=3, return_extras=True)
    tg = extras["total_goals"]
    # 104 partidos con ~2.7 goles de media -> total razonable (200-360).
    assert 200 < tg["mean"] < 360
    assert tg["p10"] <= tg["median"] <= tg["p90"]


def test_simulate_factor_fase_reduce_goles_knockout(monkeypatch):
    from config import settings
    teams = wc2026.all_participants()
    s = _strength(teams)
    # Mismo seed: la fase de grupos sale idéntica (factor 1.0); solo cambian las eliminatorias.
    monkeypatch.setattr(settings, "GOAL_STAGE_FACTOR", {"group": 1.0, "knockout": 1.0})
    _, ex_plano = mc.simulate(n_sims=800, strength=s, seed=11, return_extras=True)
    monkeypatch.setattr(settings, "GOAL_STAGE_FACTOR", {"group": 1.0, "knockout": 0.92})
    _, ex_factor = mc.simulate(n_sims=800, strength=s, seed=11, return_extras=True)
    assert ex_factor["total_goals"]["mean"] < ex_plano["total_goals"]["mean"]


def test_simulate_el_mas_fuerte_es_favorito():
    teams = wc2026.all_participants()
    # Spain mucho más fuerte -> debe liderar la prob. de ser campeón.
    df = mc.simulate(n_sims=600, strength=_strength(teams, {"Spain": 2400.0}), seed=2)
    assert df.iloc[0]["team"] == "Spain"
    assert df.iloc[0]["P_Campeon"] > df["P_Campeon"].median()


# --------------------------------------------------------------------------- #
# Cuadro oficial 2026
# --------------------------------------------------------------------------- #
def test_bracket_oficial_estructura():
    flat = wc2026.bracket_r32_flat()
    assert len(flat) == 32
    ones = sorted(t[1:] for t in flat if t[0] == "1")
    twos = sorted(t[1:] for t in flat if t[0] == "2")
    thirds = [t for t in flat if t.startswith("3:")]
    assert ones == list(wc2026.GROUP_NAMES)   # 12 ganadores, todos los grupos
    assert twos == list(wc2026.GROUP_NAMES)   # 12 subcampeones, todos los grupos
    assert len(thirds) == wc2026.THIRDS_QUALIFY == 8


def test_bracket_r32_sin_choque_mismo_grupo():
    # Ningún partido de R32 puede enfrentar a dos plazas del mismo grupo (ni un 1º/2º
    # con un tercero de su propio grupo).
    for a, b in wc2026.BRACKET_R32:
        ga = {a[1:]} if a[0] in "12" else set(a[2:])
        gb = {b[1:]} if b[0] in "12" else set(b[2:])
        if a[0] in "12" and b[0] in "12":
            assert a[1:] != b[1:]
        if a[0] in "12":
            assert a[1:] not in gb
        if b[0] in "12":
            assert b[1:] not in ga


def test_terceros_tabla_495_valida():
    tab = wc2026.third_assignment_table()
    assert len(tab) == 495                                  # C(12,8)
    eligibles = [e for _, e in wc2026._third_slots()]
    for combo, assign in tab.items():
        assert set(assign) == set(combo)                   # biyección con la combinación
        assert len(set(assign)) == 8
        for g, elig in zip(assign, eligibles):
            assert g in elig                               # cada grupo en plaza elegible


def test_build_official_bracket_coloca_plazas():
    import numpy as np
    gn = list(wc2026.GROUP_NAMES)
    # Codificación trazable: ganador de grupo g -> g ; subcampeón -> 100+g ; tercero -> 200+g.
    winners = np.array([[i for i in range(12)]])
    runners = np.array([[100 + i for i in range(12)]])
    thirds = np.array([[200 + i for i in range(12)]])
    th_order = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])         # clasifican terceros de A..H
    cur = mc._build_official_bracket(winners, runners, thirds, th_order, gn)
    assert cur.shape == (1, 32)
    flat = wc2026.bracket_r32_flat()
    for p, tok in enumerate(flat):
        val = int(cur[0, p])
        if tok[0] == "1":
            assert val == gn.index(tok[1:])                # ganador del grupo correcto
        elif tok[0] == "2":
            assert val == 100 + gn.index(tok[1:])          # subcampeón del grupo correcto
        else:
            g = val - 200                                  # es un tercero...
            assert 0 <= g < 12 and gn[g] in set(tok[2:])   # ...de un grupo elegible
