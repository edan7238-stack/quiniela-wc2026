"""Registro de resultados del Mundial — núcleo compartido (agente + dashboard).

Centraliza lo que hoy hace a mano la página "Ingreso de datos": normalizar los nombres a los
canónicos del dataset, fijar la localía del anfitrión, deduplicar y anexar a
`data/wc2026_results.csv`. Lo usa el agente (`scripts/agent_resultados.py`) y puede reutilizarlo
el dashboard.
"""
from __future__ import annotations

import difflib

import pandas as pd

from config import settings, wc2026
from src import team_ratings

COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score",
           "tournament", "city", "country", "neutral"]


def valid_teams() -> list[str]:
    """Los 48 participantes (nombres canónicos del dataset Kaggle)."""
    return wc2026.all_participants()


def normalize_team(name: str) -> tuple[str | None, list[str]]:
    """Resuelve un nombre a su canónico de los 48. Devuelve `(nombre|None, sugerencias)`.

    Aplica los alias conocidos (`USA`->`United States`, etc.) y, si no hay match exacto,
    intenta sin distinguir mayúsculas y por similitud (sugerencias para que el agente corrija).
    """
    cand = team_ratings._norm_team(name)
    teams = valid_teams()
    tset = set(teams)
    if cand in tset:
        return cand, []
    lower = {t.lower(): t for t in teams}
    if cand.lower() in lower:
        return lower[cand.lower()], []
    return None, difflib.get_close_matches(cand, teams, n=3, cutoff=0.5)


def _load() -> pd.DataFrame:
    p = settings.WC2026_RESULTS_CSV
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame(columns=COLUMNS)


def add_wc_result(date, home, away, home_score, away_score, *,
                  tournament: str = "FIFA World Cup", dry_run: bool = False) -> dict:
    """Registra un resultado del Mundial. Devuelve `{ok, message, ...}` (nunca lanza).

    - Normaliza `home`/`away` a canónicos (con sugerencias si fallan).
    - `neutral = not (anfitrión local o visitante)` — los anfitriones juegan en casa.
    - Deduplica por (fecha, par de equipos) contra el CSV existente.
    - Si `dry_run`, no escribe (solo informa la fila que escribiría).
    """
    try:
        d = pd.to_datetime(date).strftime("%Y-%m-%d")
    except Exception:
        return {"ok": False, "message": f"Fecha inválida: {date!r} (usa YYYY-MM-DD)."}

    h, h_sug = normalize_team(str(home))
    if h is None:
        return {"ok": False, "message": f"Equipo local no reconocido: {home!r}.",
                "suggestions": h_sug}
    a, a_sug = normalize_team(str(away))
    if a is None:
        return {"ok": False, "message": f"Equipo visitante no reconocido: {away!r}.",
                "suggestions": a_sug}
    if h == a:
        return {"ok": False, "message": "Local y visitante no pueden ser el mismo equipo."}
    try:
        hs, as_ = int(home_score), int(away_score)
        if hs < 0 or as_ < 0:
            raise ValueError
    except Exception:
        return {"ok": False, "message": f"Marcador inválido: {home_score}-{away_score}."}

    df = _load()
    if not df.empty and {"date", "home_team", "away_team"}.issubset(df.columns):
        dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        seen = {(dt, frozenset((str(ht), str(at))))
                for dt, ht, at in zip(dates, df["home_team"], df["away_team"])}
        if (d, frozenset((h, a))) in seen:
            return {"ok": False, "duplicate": True,
                    "message": f"Ya existe un resultado para {h} vs {a} el {d}."}

    neutral = not (wc2026.is_host(h) or wc2026.is_host(a))
    row = {"date": d, "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
           "tournament": tournament, "city": "", "country": "", "neutral": neutral}

    if dry_run:
        return {"ok": True, "written": False, "row": row,
                "message": f"[dry-run] {h} {hs}-{as_} {a} ({d})"}

    out = pd.concat([df[COLUMNS] if set(COLUMNS).issubset(df.columns) else df,
                     pd.DataFrame([row])], ignore_index=True)
    settings.WC2026_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(settings.WC2026_RESULTS_CSV, index=False, encoding="utf-8")
    return {"ok": True, "written": True, "row": row,
            "message": f"Registrado: {h} {hs}-{as_} {a} ({d})."}
