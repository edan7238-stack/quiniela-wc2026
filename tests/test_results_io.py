"""Tests del núcleo de ingreso de resultados (`src/results_io.py`)."""
import pandas as pd
import pytest

from config import settings
from src import results_io


@pytest.fixture
def tmp_csv(tmp_path, monkeypatch):
    p = tmp_path / "wc2026_results.csv"
    monkeypatch.setattr(settings, "WC2026_RESULTS_CSV", p)
    return p


def test_normalize_team_alias_y_exacto():
    assert results_io.normalize_team("USA")[0] == "United States"
    assert results_io.normalize_team("Korea Republic")[0] == "South Korea"
    assert results_io.normalize_team("Brazil")[0] == "Brazil"


def test_normalize_team_desconocido_sugiere():
    name, sug = results_io.normalize_team("Atlantis")
    assert name is None and isinstance(sug, list)


def test_add_result_escribe_y_anfitrion_local(tmp_csv):
    r = results_io.add_wc_result("2026-06-11", "Mexico", "South Africa", 2, 1)
    assert r["ok"] and r["written"]
    assert r["row"]["neutral"] is False         # México es anfitrión -> juega de local
    df = pd.read_csv(tmp_csv)
    assert len(df) == 1
    assert df.iloc[0].home_team == "Mexico" and df.iloc[0].away_team == "South Africa"
    assert df.iloc[0].home_score == 2 and df.iloc[0].away_score == 1


def test_add_result_neutral_si_no_hay_anfitrion(tmp_csv):
    r = results_io.add_wc_result("2026-06-13", "Brazil", "Morocco", 1, 0)
    assert r["ok"] and r["row"]["neutral"] is True


def test_add_result_normaliza_alias_al_escribir(tmp_csv):
    r = results_io.add_wc_result("2026-06-12", "USA", "Paraguay", 1, 1)
    assert r["ok"] and r["row"]["home_team"] == "United States"


def test_add_result_dedup_ignora_orden(tmp_csv):
    assert results_io.add_wc_result("2026-06-11", "Mexico", "South Africa", 2, 1)["ok"]
    dup = results_io.add_wc_result("2026-06-11", "South Africa", "Mexico", 1, 2)  # orden inverso
    assert dup["ok"] is False and dup.get("duplicate")
    assert len(pd.read_csv(tmp_csv)) == 1


def test_add_result_equipo_invalido_no_escribe(tmp_csv):
    r = results_io.add_wc_result("2026-06-11", "Narnia", "Brazil", 1, 0)
    assert r["ok"] is False and "suggestions" in r
    assert not tmp_csv.exists()


def test_add_result_dry_run_no_escribe(tmp_csv):
    r = results_io.add_wc_result("2026-06-11", "Spain", "Brazil", 3, 0, dry_run=True)
    assert r["ok"] and r["written"] is False
    assert not tmp_csv.exists()
