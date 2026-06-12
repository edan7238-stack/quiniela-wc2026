"""Nivel 1 (ancla) — Extracción del Ranking Mundial FIFA.

La página oficial (https://inside.fifa.com/es/fifa-world-ranking/men) es un sitio
Next.js que carga el ranking por su API interna. Esta API se descubrió inspeccionando
el SSR (`__NEXT_DATA__`) y el endpoint estable es:

    https://inside.fifa.com/api/ranking-overview?locale=en&dateId=<id>

donde `<id>` proviene de `allAvailableDates`. Las ediciones más recientes migraron a
un sistema "live" distinto que este endpoint devuelve vacío; por eso seleccionamos
automáticamente la edición **más reciente que SÍ trae datos** (recorriendo de la más
nueva a la más antigua).

El ranking FIFA moderno (desde 2018) es un sistema tipo Elo, así que sus puntos sirven
como **ancla de la fuerza actual** de las selecciones, complementando el Elo propio.

Si la red o FIFA fallan, `load_snapshot()` recurre al CSV previo o al snapshot manual
de `config/wc2026.py` (fallback).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd
import requests

from config import settings

RANKING_PAGE = "https://inside.fifa.com/{locale}/fifa-world-ranking/men"
RANKING_API = "https://inside.fifa.com/api/ranking-overview?locale={locale}&dateId={date_id}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html",
}

# FIFA (locale=en) -> nombre canónico del dataset Kaggle.
# Solo los que difieren; el resto pasa por su nombre tal cual.
FIFA_TO_DATASET: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "China PR": "China",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "USA": "United States",
    "Congo DR": "DR Congo",
    "The Gambia": "Gambia",
    "Brunei Darussalam": "Brunei",
    "Hong Kong, China": "Hong Kong",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "US Virgin Islands": "United States Virgin Islands",
}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


def normalize_fifa_name(name: str) -> str:
    """Mapea un nombre de selección de FIFA al nombre canónico del dataset."""
    return FIFA_TO_DATASET.get(name, name)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def available_dates(locale: str = "en", session: requests.Session | None = None) -> list[dict]:
    """Devuelve las ediciones disponibles (`{date, id}`) ordenadas de más nueva a
    más antigua, leídas del SSR (`__NEXT_DATA__`) de la página de ranking.
    """
    s = session or _session()
    html = s.get(RANKING_PAGE.format(locale=locale), timeout=30).text
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise RuntimeError("No se encontró __NEXT_DATA__ en la página de FIFA.")
    data = json.loads(m.group(1))
    dates = data["props"]["pageProps"]["pageData"]["ranking"]["allAvailableDates"]
    return sorted(dates, key=lambda d: d["date"], reverse=True)


def fetch_ranking(locale: str = "en", max_attempts: int = 15) -> pd.DataFrame:
    """Descarga la edición más reciente del ranking FIFA que contenga datos.

    Recorre las ediciones disponibles de la más nueva a la más antigua y devuelve
    la primera no vacía. Lanza RuntimeError si ninguna de las `max_attempts`
    primeras trae datos.

    Columnas: rank, team, points, confederation, code, fifa_name, ranking_date.
    """
    s = _session()
    dates = available_dates(locale=locale, session=s)

    for entry in dates[:max_attempts]:
        url = RANKING_API.format(locale=locale, date_id=entry["id"])
        try:
            payload = s.get(url, timeout=30).json()
        except Exception:
            continue
        rankings = payload.get("rankings") or []
        if not rankings:
            continue

        rows = []
        for r in rankings:
            item = r.get("rankingItem", {})
            name = item.get("name", "")
            rows.append(
                {
                    "rank": item.get("rank"),
                    "team": normalize_fifa_name(name),
                    "points": item.get("totalPoints"),
                    "confederation": (r.get("tag") or {}).get("text"),
                    "code": item.get("countryCode"),
                    "fifa_name": name,
                    "ranking_date": entry["date"],
                }
            )
        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
        return df

    raise RuntimeError(
        f"Ninguna de las {max_attempts} ediciones FIFA más recientes devolvió datos."
    )


def save_snapshot(df: pd.DataFrame, path=settings.FIFA_SNAPSHOT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def load_snapshot(path=settings.FIFA_SNAPSHOT_CSV) -> pd.DataFrame | None:
    """Lee el snapshot FIFA guardado. Devuelve None si no existe (el llamador
    decide el fallback al snapshot manual de config)."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def points_map(df: pd.DataFrame | None = None) -> dict[str, float]:
    """Diccionario {selección_canónica -> puntos FIFA} para usar como ancla."""
    if df is None:
        df = load_snapshot()
    if df is None or df.empty:
        return {}
    return dict(zip(df["team"], df["points"]))


def refresh(locale: str = "en") -> pd.DataFrame:
    """Descarga y guarda el snapshot. Devuelve el DataFrame."""
    df = fetch_ranking(locale=locale)
    save_snapshot(df)
    return df
