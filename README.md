# ⚽ Modelo Probabilístico — Mundial 2026 (Quiniela)

Software en Python para **predecir el resultado 1X2 y el marcador exacto** de cada partido
del Mundial 2026, optimizado para **maximizar los puntos de una quiniela** (no para apostar).

El sistema no busca "adivinar el marcador exacto" (imposible: el exacto tiene un techo de
~12-18% por partido). Para cada predicción elige la opción que **maximiza tus puntos
esperados** según las reglas de la quiniela, y reporta la confianza real de cada pick.

---

## 🚀 Inicio rápido

### Requisito previo: dataset de Kaggle

El modelo necesita el historial de partidos internacionales. Descárgalo gratis:

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com) (es gratis)
2. Ve a este dataset: **[International football results from 1872 to 2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)**
3. Haz clic en **Download** → obtendrás un archivo llamado `archive.zip`
4. Copia ese archivo a la **raíz del proyecto** (junto a este README)

---

### 🍎 Mac / Linux — un solo comando

```bash
# 1. Clona el repo
git clone https://github.com/edan7238-stack/quiniela-wc2026.git
cd quiniela-wc2026

# 2. Coloca archive.zip aquí (ver instrucciones arriba)

# 3. Da permisos al script y ejecútalo
chmod +x run_dashboard.sh
./run_dashboard.sh
```

El script se encarga automáticamente de:
- Verificar que tienes Python 3.11+
- Instalar las dependencias
- Extraer el dataset
- Descargar el ranking FIFA
- Entrenar el modelo
- Abrir el dashboard en tu navegador en http://localhost:8501

> **¿No tienes Python 3.11+?** Descárgalo en https://www.python.org/downloads/

---

### 🪟 Windows — doble clic

Coloca `archive.zip` en la raíz del proyecto y haz doble clic en `run_dashboard.bat`.

---

### ⚙️ Instalación manual (avanzado)

```bash
pip install -r requirements.txt          # Python 3.11+ (probado en 3.13)
python scripts/setup_data.py             # extrae el dataset Kaggle (archive.zip)
python scripts/fetch_fifa.py             # descarga el ranking FIFA
python scripts/train.py                  # entrena Elo + ML + Poisson
streamlit run app/dashboard.py           # interfaz principal
```

---

## Arquitectura en cascada

```
[Histórico Kaggle (filtro reciente)] + [Ranking FIFA] + [xG/xGA opcional] + [Resultados manuales]
        │
  NIVEL 1  Elo dinámico + ancla FIFA        →  fuerza actual de cada selección
        │
  NIVEL 2  Clasificador ML (LightGBM)       →  probabilidades 1X2  (EJE PRINCIPAL)
        │
  NIVEL 3  Poisson + Dixon-Coles            →  matriz de marcadores (reconciliada con el ML)
        │
  NIVEL 4  Montecarlo (20k-50k sims)        →  avance, posiciones, campeón, total de goles
        │
  OPTIMIZADOR DE QUINIELA                   →  marcador óptimo 2·P(s)+P(o), comodines, futuros
```

---

## Dashboard (5 páginas)

1. **🎯 Quiniela** — el núcleo:
   - *Fase de grupos*: los 72 partidos con el marcador óptimo, marcador modal, puntos
     esperados y el **plan de comodines** (dónde poner double/triple/all-in).
   - *Futuros*: posiciones de grupo (1º/2º/3º), mejores terceros, campeón/subcampeón/3º/4º
     y total de goles del torneo. Campos manuales para goleador/balón de oro/equipo sorpresa.
   - *Eliminatorias*: marcador óptimo de cualquier cruce + **ajuste manual de fuerza**.
2. **🔮 Predicción de partido** — 1X2 + marcador para un cruce.
3. **📊 Ranking Elo + FIFA**.
4. **🎲 Simulación Montecarlo**.
5. **📝 Ingreso de datos** — añadir resultados de cada jornada (recalcula Elo/forma).

---

## La idea clave (puntuación 3/1/0)

Predecir el marcador `s` (que implica un resultado `o`):

```
E[pts(s)] = 3·P(s) + 1·(P(o) − P(s)) = 2·P(s) + P(o)   → se elige el s que lo maximiza
```

No es lo mismo que el "marcador más probable" (modal): el óptimo evita la trampa del empate
1-1 cuando un equipo es claro favorito. Los comodines (double ×2, triple ×3, all-in 12/5/−6)
se asignan a los partidos que más suben el total de puntos esperados.

---

## Reglas de la quiniela soportadas

- Por partido: exacto 3 / resultado 1 / fallo 0. Comodines double/triple/all-in (2 c/u en grupos).
- Posiciones de grupo (4/3/2) y mejores terceros (3 c/u).
- Futuros: campeón 10 / subcampeón 6 / 3º 4 / 4º 2 / total de goles 4.
- Goleador, balón de oro y equipo sorpresa son a nivel jugador/subjetivos → **entrada manual**.

---

## xG/xGA (opcional)

Si colocas un CSV en `data/xg.csv`, el cargador flexible (`src/team_ratings.py`) autodetecta
el formato y mezcla el xG/xGA con la fuerza (Elo+FIFA) en el modelo de marcador (`XG_BLEND_W`).

Sin el archivo, el modelo funciona solo con la fuerza.

---

## Scripts

| Script                        | Qué hace                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `scripts/setup_data.py`       | Extrae el dataset Kaggle a `data/raw/`.                               |
| `scripts/setup_xg.py`         | Prepara `data/xg.csv` desde el CSV por-partido con xG del usuario.    |
| `scripts/fetch_fifa.py`       | Descarga el ranking FIFA → `models/fifa_snapshot.csv`.                |
| `scripts/train.py`            | Entrena Elo (+FIFA), ML 1X2 y Poisson.                                |
| `scripts/simulate.py`         | Montecarlo del torneo → favoritos.                                    |
| `scripts/backtest.py`         | Backtest fuera de muestra: acierto exacto/1X2 y puntos/partido.       |
| `scripts/calibrate_xg.py`     | Calibra `XG_BLEND_W`/`XG_DECAY_PER_DAY` por backtest point-in-time.   |
| `scripts/export_excel.py`     | Vuelca las predicciones de grupos a Excel (determinista, sin IA).     |
| `scripts/agent_resultados.py` | Agente (API Anthropic) que ingresa resultados desde capturas/enlaces. |

---

## Rendimiento (backtest fuera de muestra, 1616 partidos 2024-2026)

| Métrica                 | Pick óptimo | Modal |
| ----------------------- | ----------- | ----- |
| Acierto marcador exacto | 13.4%       | 13.9% |
| Acierto resultado 1X2   | 58.0%       | —     |
| Puntos/partido (3/1/0)  | **0.848**   | 0.809 |

El optimizador saca **+0.04 pts/partido** sobre elegir el marcador más probable.

---

## Estructura

```
config/   settings.py · wc2026.py (grupos OFICIALES del sorteo + anfitriones)
data/     raw/ (Kaggle) · wc2026_results.csv (manual) · xg.csv (opcional, lo aportas tú)
models/   elo_ratings.csv · ml_1x2.pkl · poisson_params.pkl · fifa_snapshot.csv
src/      data_loader · elo · fifa · features · ml_model · poisson · team_ratings ·
          montecarlo · predict · quiniela
scripts/  setup_data · fetch_fifa · train · simulate · backtest
app/      dashboard.py (Streamlit)
tests/    pytest (54 tests)
```

---

## Tests

```bash
python -m pytest tests/ -q
```

---

## Notas

- Los grupos son los del **sorteo oficial** (`config/wc2026.py`); los anfitriones (México,
  EE. UU., Canadá) juegan como locales (ventaja de localía).
- Eliminatorias: el marcador se modela a 90 minutos.
- El ranking FIFA online puede cambiar de formato; si la descarga falla se conserva el
  snapshot previo o el fallback manual de `config/wc2026.py`.
- Ningún modelo garantiza aciertos; el objetivo es maximizar puntos esperados, no adivinar.
