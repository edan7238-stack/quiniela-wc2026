"""Tests del cargador de señales pre-torneo (Hito A2 agregado + Hito A3 por-partido)."""
import pandas as pd
import pytest

from config import settings
from src import team_ratings as tr

# Conjunto pequeño de "selecciones" para no cargar el histórico Kaggle en los tests.
TEAMS = {"Spain", "France", "Germany", "Italy", "Bosnia and Herzegovina"}


def _pm(rows) -> pd.DataFrame:
    """Construye un DataFrame por-partido desde tuplas (date, home, away, hxg, axg)."""
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                       "home_xg", "away_xg"])


# --------------------------------------------------------------------------- #
# Detección de formato
# --------------------------------------------------------------------------- #
def test_detecta_formato_por_partido():
    df = _pm([("2024-09-01", "Spain", "France", 1.5, 0.8)])
    assert tr._is_per_match(df) is True


def test_detecta_formato_agregado():
    df = pd.DataFrame({"team": ["Spain"], "xg_for": [1.6], "xg_against": [0.9]})
    assert tr._is_per_match(df) is False


def test_detecta_columnas_con_alias_y_mayusculas():
    # Cabeceras alternativas (Home/Away, xG_home/xG_away, Date) deben detectarse.
    df = pd.DataFrame(columns=["Date", "Home", "Away", "xG_home", "xG_away"])
    assert tr._is_per_match(df) is True


# --------------------------------------------------------------------------- #
# Filtro a partidos entre selecciones (descarta clubes)
# --------------------------------------------------------------------------- #
def test_filtra_solo_partidos_entre_selecciones(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([
        ("2024-09-01", "Spain", "France", 1.5, 0.8),      # selección vs selección -> entra
        ("2024-09-05", "Real Madrid", "Barcelona", 2.1, 1.3),  # club vs club -> fuera
        ("2024-09-08", "Spain", "Barcelona", 3.0, 0.2),   # mixto -> fuera
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS)
    assert set(out["team"]) == {"Spain", "France"}


def test_perspectiva_local_visitante_correcta(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([("2024-09-01", "Spain", "France", 2.0, 0.5)])
    out = tr._aggregate_per_match(df, teams=TEAMS).set_index("team")
    assert out.loc["Spain", "xg_for"] == pytest.approx(2.0)
    assert out.loc["Spain", "xg_against"] == pytest.approx(0.5)
    assert out.loc["France", "xg_for"] == pytest.approx(0.5)
    assert out.loc["France", "xg_against"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Alias de nombres de selección
# --------------------------------------------------------------------------- #
def test_alias_de_nombre_de_seleccion(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([("2024-09-01", "Bosnia & Herzegovina", "France", 1.0, 1.0)])
    out = tr._aggregate_per_match(df, teams=TEAMS)
    assert "Bosnia and Herzegovina" in set(out["team"])


def test_norm_team_mapea_variantes():
    assert tr._norm_team("D.R. Congo") == "DR Congo"
    assert tr._norm_team("Chinese Taipei") == "Taiwan"
    assert tr._norm_team("Ireland") == "Republic of Ireland"


# --------------------------------------------------------------------------- #
# Ventana de recencia + decaimiento temporal
# --------------------------------------------------------------------------- #
def test_descarta_partidos_anteriores_al_corte(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([
        ("2017-06-01", "Spain", "France", 5.0, 0.0),   # < CORTE_RECIENTE (2018) -> fuera
        ("2024-09-01", "Spain", "France", 1.0, 1.0),
        ("2024-10-01", "Spain", "Italy", 1.0, 1.0),
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS).set_index("team")
    # Si el de 2017 contara, Spain xg_for subiría muchísimo; debe quedar en ~1.0.
    assert out.loc["Spain", "xg_for"] == pytest.approx(1.0, abs=1e-6)
    assert out.loc["Spain", "matches"] == 2


def test_decaimiento_pondera_lo_reciente(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    # Spain: partido reciente con xG alto (3.0) y antiguo con xG bajo (1.0).
    df = _pm([
        ("2023-08-01", "Spain", "France", 1.0, 1.0),   # antiguo
        ("2025-08-01", "Spain", "Italy", 3.0, 1.0),    # reciente (~ref)
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS).set_index("team")
    media_simple = 2.0
    val = out.loc["Spain", "xg_for"]
    assert media_simple < val < 3.0   # el reciente pesa más, pero el antiguo aún cuenta


def test_decaimiento_cero_es_media_simple(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    monkeypatch.setattr(settings, "XG_DECAY_PER_DAY", 0.0)  # sin decaimiento
    df = _pm([
        ("2023-08-01", "Spain", "France", 1.0, 1.0),
        ("2025-08-01", "Spain", "Italy", 3.0, 1.0),
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS).set_index("team")
    assert out.loc["Spain", "xg_for"] == pytest.approx(2.0)  # (1+3)/2


# --------------------------------------------------------------------------- #
# Guardarraíl de mínimo de partidos
# --------------------------------------------------------------------------- #
def test_minimo_de_partidos_descarta_seleccion(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 3)
    df = _pm([
        ("2024-09-01", "Spain", "France", 1.0, 1.0),
        ("2024-10-01", "Spain", "Italy", 1.0, 1.0),
        ("2024-11-01", "Spain", "Germany", 1.0, 1.0),
        ("2024-09-05", "France", "Italy", 1.0, 1.0),  # France solo 2 partidos -> fuera
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS)
    teams_out = set(out["team"])
    assert "Spain" in teams_out          # 3 partidos
    assert "France" not in teams_out     # 2 partidos (< mínimo)


def test_sin_partidos_validos_devuelve_none(monkeypatch):
    df = _pm([("2024-09-01", "Real Madrid", "Barcelona", 2.0, 1.0)])  # solo clubes
    assert tr._aggregate_per_match(df, teams=TEAMS) is None


def test_filas_con_xg_nulo_se_ignoran(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([
        ("2024-09-01", "Spain", "France", None, 1.0),   # xG local nulo -> fuera
        ("2024-10-01", "Spain", "Italy", 2.0, 0.5),
    ])
    out = tr._aggregate_per_match(df, teams=TEAMS).set_index("team")
    assert out.loc["Spain", "matches"] == 1
    assert out.loc["Spain", "xg_for"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# national_xg_long (long-form sin agregar, para backtests point-in-time)
# --------------------------------------------------------------------------- #
def test_national_xg_long_formato_y_filtro():
    df = _pm([
        ("2024-09-01", "Spain", "France", 2.0, 0.5),
        ("2024-10-01", "Real Madrid", "Barcelona", 3.0, 1.0),  # club -> fuera
    ])
    long = tr.national_xg_long(df, teams=TEAMS)
    assert list(long.columns) == ["date", "team", "xgf", "xga"]
    # 1 partido entre selecciones -> 2 filas (una por equipo), sin clubes.
    assert len(long) == 2
    assert set(long["team"]) == {"Spain", "France"}
    spain = long[long["team"] == "Spain"].iloc[0]
    assert spain["xgf"] == pytest.approx(2.0) and spain["xga"] == pytest.approx(0.5)


def test_national_xg_long_no_agrega_ni_decae():
    # Dos partidos de Spain -> dos filas (NO promedia): es materia prima point-in-time.
    df = _pm([
        ("2023-08-01", "Spain", "France", 1.0, 1.0),
        ("2025-08-01", "Spain", "Italy", 3.0, 1.0),
    ])
    long = tr.national_xg_long(df, teams=TEAMS)
    assert sorted(long[long["team"] == "Spain"]["xgf"]) == [1.0, 3.0]


def test_national_xg_long_corte_recencia():
    df = _pm([
        ("2017-06-01", "Spain", "France", 5.0, 0.0),  # < CORTE_RECIENTE -> fuera
        ("2024-09-01", "Spain", "France", 1.0, 1.0),
    ])
    long = tr.national_xg_long(df, teams=TEAMS)
    assert (long["date"].dt.year >= settings.CORTE_RECIENTE).all()
    assert len(long) == 2  # solo el partido de 2024 (x2 equipos)


# --------------------------------------------------------------------------- #
# Formato AGREGADO (compatibilidad hacia atrás)
# --------------------------------------------------------------------------- #
def test_formato_agregado_sigue_funcionando():
    df = pd.DataFrame({"equipo": ["Spain", "France"], "xgf": [1.8, 1.2],
                       "xga": [0.7, 1.1], "valor": [900, 800]})
    out = tr._aggregate_per_team(df, "test.csv").set_index("team")
    assert out.loc["Spain", "xg_for"] == pytest.approx(1.8)
    assert out.loc["France", "xg_against"] == pytest.approx(1.1)
    assert out.loc["Spain", "squad_value"] == pytest.approx(900)


# --------------------------------------------------------------------------- #
# TeamRatings (API consumidora)
# --------------------------------------------------------------------------- #
def test_teamratings_api_por_partido(monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    df = _pm([("2024-09-01", "Spain", "France", 2.0, 0.5)])
    agg = tr._aggregate_per_match(df, teams=TEAMS)
    r = tr.TeamRatings(agg, source="per_match")
    assert r.has_xg is True
    assert r.source == "per_match"
    assert r.xg_for("Spain") == pytest.approx(2.0)
    assert isinstance(r.xg_for("Spain"), float)
    assert r.xg_for("Marte") is None              # desconocido -> None (cae a fuerza)
    assert r.n_matches("Spain") == 1
    assert "Spain" in r.teams_with_xg()
    assert r.league_avg_xg() == pytest.approx((2.0 + 0.5) / 2)


def test_teamratings_vacio():
    r = tr.TeamRatings(None)
    assert r.has_xg is False
    assert r.source == "empty"
    assert r.xg_for("Spain") is None
    assert r.n_matches("Spain") == 0
    # league_avg cae al valor por defecto (mitad del total base).
    assert r.league_avg_xg() == pytest.approx(settings.GOAL_BASE_TOTAL / 2.0)


# --------------------------------------------------------------------------- #
# Dispatch de load_xg sobre CSV temporal
# --------------------------------------------------------------------------- #
def test_load_xg_dispatch_por_partido(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "XG_MIN_MATCHES", 1)
    p = tmp_path / "xg.csv"
    _pm([
        ("2024-09-01", "Spain", "France", 1.5, 0.8),
        ("2024-10-01", "France", "Spain", 1.0, 1.2),
    ]).to_csv(p, index=False)
    out = tr.load_xg(path=p)
    assert out is not None
    assert set(out["team"]) == {"Spain", "France"}
    assert "xg_for" in out.columns and "xg_against" in out.columns


def test_load_xg_inexistente_devuelve_none(tmp_path):
    assert tr.load_xg(path=tmp_path / "no_existe.csv") is None


# --------------------------------------------------------------------------- #
# Integración con los datos REALES del usuario (si están presentes)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not settings.XG_CSV.exists(), reason="data/xg.csv no presente")
def test_integracion_datos_reales():
    r = tr.load()
    assert r.source == "per_match"
    assert r.has_xg is True
    # La media de xG por partido de selecciones debe ser razonable (~1.0-1.6).
    assert 0.9 < r.league_avg_xg() < 1.7
    # Selección europea bien cubierta -> tiene xG; anfitrión sin clasificación -> no.
    assert r.xg_for("Spain") is not None
    assert r.n_matches("Spain") >= settings.XG_MIN_MATCHES
    assert r.xg_for("Mexico") is None   # anfitrión: sin partidos de clasificación
