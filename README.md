# ⚽ Modelo Probabilístico — Mundial 2026 (Quiniela)

Software en Python para **predecir el resultado 1X2 y el marcador exacto** de cada partido
del Mundial 2026, optimizado para **maximizar los puntos de una quiniela** (no para apostar).

El sistema no busca "adivinar el marcador exacto" (imposible: el exacto tiene un techo de
~12-18% por partido). Para cada predicción elige la opción que **maximiza tus puntos
esperados** según las reglas de la quiniela, y reporta la confianza real de cada pick.

---

## 🚀 Inicio rápido

### 🍎 Mac / Linux

**Requisitos previos:**
- Python 3.11 o superior → descárgalo en https://www.python.org/downloads/ si no lo tienes

**Pasos:**

```bash
# 1. Clona el repositorio
git clone https://github.com/edan7238-stack/quiniela-wc2026.git
cd quiniela-wc2026

# 2. Da permisos al script y ejecútalo
chmod +x run_dashboard.sh
./run_dashboard.sh
```

¡Listo! El script se encarga automáticamente de todo:

- ✅ Verificar Python 3.11+
- ✅ Instalar las dependencias
- ✅ Extraer el dataset de partidos (`archive.zip`)
- ✅ Extraer los datos xG/xGA (`archive(1).zip`)
- ✅ Descargar el ranking FIFA
- ✅ Entrenar el modelo
- ✅ Abrir el dashboard en tu navegador → http://localhost:8501

> La primera vez tarda 2-3 minutos. Las siguientes veces abre directo.

> Para cerrar el servidor: **Ctrl+C** en la terminal.

---

### 🪟 Windows

Haz doble clic en `run_dashboard.bat`.

---

### ⚙️ Instalación manual (usuarios avanzados)

```bash
pip install -r requirements.txt
python scripts/setup_data.py       # extrae archive.zip → data/raw/
python scripts/fetch_fifa.py       # descarga el ranking FIFA
python scripts/train.py            # entrena Elo + ML + Poisson
streamlit run app/dashboard.py     # abre el dashboard
```

---

## Arquitectura en cascada

```
[Histórico de partidos] + [Ranking FIFA] + [xG/xGA] + [Resultados manuales]
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
- Goleador, balón de oro y equipo sorpresa → **entrada manual**.

---

## Scripts

| Script                        | Qué hace                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `scripts/setup_data.py`       | Extrae el dataset de partidos a `data/raw/`.                          |
| `scripts/setup_xg.py`         | Prepara `data/xg.csv` desde el CSV con xG del usuario.               |
| `scripts/fetch_fifa.py`       | Descarga el ranking FIFA → `models/fifa_snapshot.csv`.                |
| `scripts/train.py`            | Entrena Elo (+FIFA), ML 1X2 y Poisson.                                |
| `scripts/simulate.py`         | Montecarlo del torneo → favoritos.                                    |
| `scripts/backtest.py`         | Backtest fuera de muestra: acierto exacto/1X2 y puntos/partido.       |
| `scripts/calibrate_xg.py`     | Calibra `XG_BLEND_W`/`XG_DECAY_PER_DAY` por backtest point-in-time.  |
| `scripts/export_excel.py`     | Vuelca las predicciones de grupos a Excel.                            |
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
data/     raw/ (partidos) · wc2026_results.csv (manual) · xg.csv
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
