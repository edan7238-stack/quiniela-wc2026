"""Configuración central del modelo probabilístico del Mundial 2026.

Todas las rutas se derivan de la ubicación de este archivo, así que el proyecto
es portable: funciona sin importar desde dónde se ejecute Python.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
MODELS_DIR: Path = ROOT_DIR / "models"

# Dataset Kaggle "International football results from 1872 to present"
KAGGLE_ZIP: Path = Path.home() / "Downloads" / "archive.zip"
RESULTS_CSV: Path = RAW_DIR / "results.csv"
SHOOTOUTS_CSV: Path = RAW_DIR / "shootouts.csv"
GOALSCORERS_CSV: Path = RAW_DIR / "goalscorers.csv"
FORMER_NAMES_CSV: Path = RAW_DIR / "former_names.csv"

# Datos de entrada manual (se rellenan durante el torneo desde el dashboard)
WC2026_RESULTS_CSV: Path = DATA_DIR / "wc2026_results.csv"

# Datos pre-torneo aportados por el usuario (xG/xGA por selección, etc.)
XG_CSV: Path = DATA_DIR / "xg.csv"

# Artefactos generados
ELO_RATINGS_CSV: Path = MODELS_DIR / "elo_ratings.csv"
FIFA_SNAPSHOT_CSV: Path = MODELS_DIR / "fifa_snapshot.csv"
ML_MODEL_PKL: Path = MODELS_DIR / "ml_1x2.pkl"
POISSON_PARAMS_PKL: Path = MODELS_DIR / "poisson_params.pkl"

# --------------------------------------------------------------------------- #
# Nivel 1 — Sistema Elo (método World Football Elo / eloratings.net)
# --------------------------------------------------------------------------- #
ELO_INITIAL: float = 1500.0          # rating de partida para un equipo nuevo
ELO_HOME_ADVANTAGE: float = 65.0     # puntos extra al local (salvo campo neutral)

# K base por importancia del torneo. Se mapea desde la columna `tournament`.
# Cuanto más importante el partido, más se mueve el rating.
ELO_K_BY_IMPORTANCE: dict[str, float] = {
    "world_cup": 60.0,        # Mundial (fase final)
    "continental": 50.0,      # Eurocopa, Copa América, etc.
    "world_cup_qual": 40.0,   # Clasificatorias mundialistas
    "continental_qual": 40.0,
    "confederations": 40.0,
    "nations_league": 40.0,
    "friendly": 20.0,         # Amistosos (poco peso)
    "other": 30.0,
}

# Mezcla de fuerza actual = w*Elo_calculado + (1-w)*Elo_equivalente_FIFA.
# El FIFA ancla la fuerza de las 48 selecciones y corrige derivas del Elo por amistosos.
ELO_FIFA_BLEND_W: float = 0.70       # peso del Elo propio frente al FIFA

# --------------------------------------------------------------------------- #
# Nivel 2 — Clasificador ML (recencia y entrenamiento)
# --------------------------------------------------------------------------- #
# FILTRO DURO DE RECENCIA: el ML solo entrena con partidos desde este año.
CORTE_RECIENTE: int = 2018

# Ventana (en años) sobre la que se calculan las features de "forma reciente".
FORMA_VENTANA_ANIOS: int = 3
FORMA_MIN_PARTIDOS: int = 5           # mínimo de partidos para que la forma sea fiable

# Fracción final de la ventana reciente reservada para validación temporal.
VALIDACION_HOLDOUT_FRAC: float = 0.20

RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Nivel 3 — Poisson bivariado + Dixon-Coles
# --------------------------------------------------------------------------- #
POISSON_MAX_GOALS: int = 10          # tamaño de la matriz de marcadores (0..N)
DIXON_COLES_XI: float = 0.0018       # tasa de decaimiento temporal (φ ~ medio año)

# --------------------------------------------------------------------------- #
# Nivel 4 — Simulación de Montecarlo
# --------------------------------------------------------------------------- #
N_SIMULACIONES: int = 20_000

# Modelo de goles del Montecarlo a partir de la FUERZA ajustada (Elo+FIFA), NO del
# ataque/defensa Poisson crudo (que sobrevalora a quien golea a rivales débiles).
# Calibrado sobre la ventana reciente: goal_diff ~ slope·elo_diff, y goles totales medios.
GOAL_SLOPE_PER_ELO: float = 0.0053   # goles de ventaja por punto de fuerza (1 gol ~ 189)
GOAL_BASE_TOTAL: float = 2.73        # goles totales esperados en un partido medio
GOAL_LAMBDA_MIN: float = 0.15        # cota inferior de λ para evitar valores no positivos

# --------------------------------------------------------------------------- #
# Quiniela — reglas de puntuación (del usuario) y pesos del modelo de marcador
# --------------------------------------------------------------------------- #
# Puntos por partido: marcador exacto / resultado correcto (1X2) / fallo.
QUINIELA_PTS_EXACTO: int = 3
QUINIELA_PTS_RESULTADO: int = 1
QUINIELA_PTS_FALLO: int = 0

# Peso del xG/xGA frente a la fuerza (Elo+FIFA) en la λ del marcador.
# CALIBRADO (scripts/calibrate_xg.py, backtest point-in-time, 574 partidos con xG en ambas
# selecciones): el óptimo en puntos/partido es w≈0.3; w>=0.5 degrada (el xG de selecciones
# europeas viene inflado por golear a rivales débiles). 0 = solo fuerza ; 1 = solo xG.
XG_BLEND_W: float = 0.3

# --- Modo por-partido del cargador de xG (src/team_ratings.py) ---
# Si el CSV de xG viene por PARTIDO (muchas ligas/clubes), el cargador filtra a los
# partidos entre selecciones (ambos equipos en data_loader.all_teams()), aplica el
# corte de recencia (CORTE_RECIENTE) y pondera por DECAIMIENTO temporal antes de
# promediar el xG a favor / en contra de cada selección.
# Decaimiento diario: peso = exp(-XG_DECAY_PER_DAY · días_de_antigüedad). Semivida ~3.2 años
# (ln2/0.0006 ≈ 1155 d): decaimiento SUAVE — como el xG de selecciones es escaso, conservar
# casi todo el historial (2-3 años) calibró mejor que decaer rápido (ver scripts/calibrate_xg.py).
XG_DECAY_PER_DAY: float = 0.0006
# Mínimo de partidos (sin ponderar) para fiarnos del xG de una selección; por debajo,
# esa selección no recibe señal xG y su marcador se queda solo con la fuerza.
XG_MIN_MATCHES: int = 3

# Ajuste de goles por fase (las eliminatorias suelen ser menos goleadoras).
GOAL_STAGE_FACTOR: dict[str, float] = {"group": 1.0, "knockout": 0.92}
