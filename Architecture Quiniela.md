# Architecture Quiniela — Modelo Probabilístico WC 2026

> Documento de continuidad. Cítalo en un chat nuevo para seguir trabajando.
> Proyecto: `C:\Modelo probabilístico WC 2026` · Python 3.13 (Win) · `pytest`: **101 verdes**.

## 1. Objetivo
Predecir 1X2 y marcador exacto de cada partido del Mundial 2026 para **maximizar puntos en
una quiniela** (NO apuestas — el módulo +EV/momios fue eliminado). El exacto tiene techo
~12-18%; el sistema elige lo que **maximiza puntos esperados**, no "adivina".

## 2. Reglas de la quiniela (del usuario)
- Por partido: **exacto 3 / resultado 1 / fallo 0**.
- Comodines (solo grupos, 2 usos c/u): `double` ×2, `triple` ×3, `all-in` (12/5/−6).
- Posiciones de grupo: 1º 4 / 2º 3 / 3º 2. Mejores terceros: 3 c/u.
- Futuros: campeón 10 / subcampeón 6 / 3º 4 / 4º 2 / total de goles 4 / goleador 6 /
  balón de oro 5 / equipo sorpresa 4.
- Grupos se predicen **de golpe**; eliminatorias **ronda por ronda**.

## 3. Fórmulas núcleo
- **Pick óptimo por partido** (3/1/0): elegir `s` que maximiza `E[pts(s)] = 2·P(s) + P(o(s))`.
- **All-in**: `E = 7·P(s) + 11·P(o) − 6`.
- **Reconciliación**: la matriz Poisson se reescala para que sus marginales 1X2 = las del ML
  (`P'(s)=P_pois(s)·P_ML(o)/P_pois(o)`).
- **λ del marcador** (anclada a fuerza): `sup=slope·(fuerzaH−fuerzaA)`; si local, `+slope·65`;
  `λH=(base+sup)/2`, `λA=(base−sup)/2`; mezcla con xG si hay (`XG_BLEND_W`).
- **Metodología 2 motores**: marcadores por partido = matriz analítica (NO Montecarlo);
  Montecarlo (20k-50k) solo para futuros/avance (probabilidades por frecuencia, no "escenario
  más probable"). Optimización **por componente** (puntos aditivos).

## 4. Arquitectura en cascada
```
[Kaggle hist. (≥2018)] + [FIFA] + [xG opcional] + [resultados manuales]
  → N1 Elo+FIFA (fuerza) → N2 ML LightGBM (1X2) → N3 Poisson+Dixon-Coles (matriz, reconciliada)
  → N4 Montecarlo (futuros) → OPTIMIZADOR quiniela (pick, comodines, futuros)
```

## 5. Mapa de archivos (todo construido y testeado)
- `config/settings.py` — parámetros. Claves: `CORTE_RECIENTE=2018`, `ELO_FIFA_BLEND_W=0.70`,
  `ELO_HOME_ADVANTAGE=65`, `GOAL_SLOPE_PER_ELO=0.0053`, `GOAL_BASE_TOTAL=2.73`,
  `GOAL_LAMBDA_MIN=0.15`, `GOAL_STAGE_FACTOR={group:1, knockout:0.92}`, `XG_BLEND_W=0.3`
  (calibrado), `XG_DECAY_PER_DAY=0.0006` (semivida ~3.2 a, calibrado), `XG_MIN_MATCHES=3`,
  `N_SIMULACIONES=20000`, `DIXON_COLES_XI=0.0018`, `POISSON_MAX_GOALS=10`,
  `QUINIELA_PTS_EXACTO/RESULTADO/FALLO=3/1/0`. (Sin EV/Kelly/ODDS.)
- `config/wc2026.py` — **GROUPS oficiales del sorteo** (A..L), `HOSTS={Mexico,Canada,United
  States}`, `is_host`, `all_participants`, `group_of`, `THIRDS_QUALIFY=8`. **Cuadro OFICIAL
  2026**: `BRACKET_R32` (16 partidos en orden de cuadro, tokens `1X/2X/3:XYZ`), `BRACKET_R32_IDS`
  (P73..P88), `bracket_r32_flat`, `third_slot_positions`, `third_assignment_table` (las 495
  combinaciones de 8 terceros → su plaza, matching válido determinista).
- `src/data_loader.py` — `load_matches(recent_only,since_year,include_manual)`, `all_teams`,
  `tournament_importance`, `k_factor`, `canonical_name`. Nombres canónicos = dataset Kaggle.
- `src/elo.py` — `EloModel(.process,.update_one,.ratings)`, `blend_with_fifa`, `save/load_ratings`.
- `src/fifa.py` — `refresh/fetch_ranking/load_snapshot/points_map`, `FIFA_TO_DATASET`
  (API interna inside.fifa.com; toma la edición más reciente con datos).
- `src/features.py` — `FormTracker(.process,.current_form)`, `assemble_features`,
  `single_match_row`, `FEATURE_COLS` (elo_diff, forma 3 años, etc.; SIN mercado).
- `src/ml_model.py` — `train(calibrate='none')`, `build_training_frame`, `predict_proba`,
  `load_bundle`, `_temporal_split`, `CLASSES=['A','D','H']`. LightGBM crudo (mejor ECE).
- `src/poisson.py` — `fit`, `DixonColesParams`, `score_matrix`, `outcome_probs`,
  `reconcile_with_1x2`, `markets_from_matrix`, `predict_match`.
- `src/team_ratings.py` — `load()`, `load_xg()`, `TeamRatings(.has_xg,.xg_for,.xg_against,
  .squad_value,.league_avg_xg,.n_matches,.teams_with_xg,.source)` + `national_xg_long()`
  (long-form sin agregar, para backtests point-in-time). **Autodetecta 2 formatos**: agregado
  por equipo (Hito A2) y **por-partido (Hito A3, hecho)**: filtra a partidos selección-vs-
  selección (ambos en `all_teams()` → descarta clubes solo), corte `CORTE_RECIENTE`,
  **decaimiento temporal** (`XG_DECAY_PER_DAY`) y promedio ponderado xG a favor/en contra;
  guardarraíl `XG_MIN_MATCHES`. Mapea nombres con `_TEAM_ALIASES`.
- `src/montecarlo.py` — `simulate(n_sims,groups,strength,...,return_extras)` →
  cols `P_grupo,P_g1..g4,P_best_third,P_R32..P_Campeon,P_subcampeon,P_3er_puesto,P_4to_puesto`
  + extras `total_goals`. **Eliminatorias con el CUADRO OFICIAL 2026** (`_build_official_bracket`
  + `_third_assign_array`); grupos no oficiales → fallback `bracket_seed_order`. `load_strength`.
  Ventaja anfitrión en grupos. Goles base escalados por `GOAL_STAGE_FACTOR` (grupos 1.0, knockouts 0.92).
- `src/predict.py` — `Predictor`: `.predict_1x2`, `.score_matrix_for(h,a,neutral,stage)`
  (reconciliada), `.predict_score`, `.predict`, `.strength/.strength_map/.teams`,
  `.set_adjustment(team,delta)` (perilla manual día-antes).
- `src/quiniela.py` — `Scoring/DEFAULT/ALLIN`, `expected_points`, `best_scoreline`,
  `most_likely_scoreline`, `evaluate_match`, `allocate_comodines`, `group_fixtures` (usa el
  **CALENDARIO OFICIAL FIFA** del dataset: 72 partidos en orden cronológico real con `fecha`),
  `recommend_group_stage(predictor)` (tabla con col. `fecha`), `group_position_picks`,
  `best_thirds_picks`, `placement_picks`, `total_goals_pick`.
- `src/results_io.py` — ingreso de resultados (compartido agente+dashboard): `valid_teams`,
  `normalize_team` (alias→canónico, con sugerencias), `add_wc_result` (normaliza, localía de
  anfitrión, deduplica, anexa a `data/wc2026_results.csv`).
- `src/bracket.py` — **eliminatorias por ronda (Hito #4)**: `group_standings`,
  `standings_from_matches` (real) / `standings_from_projection` (Montecarlo) → ganadores/
  subcampeones/terceros + 8 grupos clasificados; `resolve_r32` (resuelve tokens del cuadro a
  equipos), `advance` (empareja adyacentes), `recommend_round(predictor,matchups,comodines=None)`
  (tabla batch, marcador a 90', `avanza`=P(gana)+½·empate), `project_rounds` (encadena R32→Final).
- `app/dashboard.py` — 5 páginas: **🎯 Quiniela** (grupos/futuros/**eliminatorias por ronda**:
  selector de ronda + modo Proyectado/Real con el cuadro oficial; + cruce manual + ajuste),
  Predicción, Ranking, Montecarlo, Ingreso de datos.
- `scripts/` — `setup_data.py`, `setup_xg.py` (genera `data/xg.csv` slim desde el CSV
  por-partido del usuario), `fetch_fifa.py`, `train.py` (Elo+ML+Poisson), `simulate.py`,
  `backtest.py`, `calibrate_xg.py` (calibra `XG_BLEND_W`/`XG_DECAY_PER_DAY` point-in-time),
  `export_excel.py` (predicciones de grupos → Excel, **determinista, sin IA**; mapeo `EXCEL_MAP`),
  `agent_resultados.py` (**agente API Anthropic**, `claude-sonnet-4-6`, visión + `web_fetch`:
  lee marcadores de capturas/enlaces → `results_io.add_wc_result`; flags `--dry-run`/`--retrain`).
- `tests/` — test_elo, test_features, test_ml, test_poisson, test_quiniela, test_montecarlo
  (+ cuadro oficial 2026: estructura, sin choque mismo grupo, tabla de 495 terceros, colocación),
  test_bracket (standings, resolución R32, avance, tabla batch por ronda, comodines),
  test_team_ratings (22: formatos, filtro selecciones, alias, decaimiento, mínimos, long-form
  point-in-time, integración real), test_results_io (normalización/alias/dedup/host-neutral),
  test_export_excel (tabla + mapeo a celdas conservando plantilla).
- `models/` — `elo_ratings.csv`, `ml_1x2.pkl`, `poisson_params.pkl`, `fifa_snapshot.csv`.

## 6. Comandos
```bash
pip install -r requirements.txt
python scripts/setup_data.py            # extrae Kaggle (ya hecho)
python scripts/setup_xg.py              # genera data/xg.csv desde ~/Downloads/xG.csv (ya hecho)
python scripts/fetch_fifa.py            # ranking FIFA (ya hecho)
python scripts/train.py                 # Elo+ML+Poisson (ya hecho)
python scripts/simulate.py              # favoritos del Montecarlo
python scripts/backtest.py              # honestidad (fuerza pura)
python scripts/calibrate_xg.py          # calibra XG_BLEND_W/decay (point-in-time)
python scripts/export_excel.py --excel tu.xlsx   # predicciones de grupos -> Excel (sin IA)
python scripts/agent_resultados.py --image c.png # agente: resultados de capturas/enlaces (API)
streamlit run app/dashboard.py          # interfaz
python -m pytest tests/ -q              # 101 tests
```

## 7. Rendimiento actual (backtest fuera de muestra, 1616 partidos 2024-2026)
Exacto pick óptimo 13.4% (modal 13.9%) · 1X2 58.0% · **0.848 pts/partido** vs modal 0.809
(+0.04). ML: log-loss 0.86 / acc 0.60 / ECE 0.03. Montecarlo: España ~22% campeón, total
goles ~277 (con factor de fase en knockouts). Grupos: ~64 pts base → ~78 con comodines.

## 8. PASOS PARA COMPLETAR (TODO)
**Prioridad ALTA — integrar xG:**
1. ✅ **HECHO (Hito A3).** `src/team_ratings.py` modo **por-partido**: autodetecta columnas,
   filtra a partidos selección-vs-selección (ambos en `all_teams()`), corte `CORTE_RECIENTE`
   + **decaimiento temporal**, y agrega `xg_for/xg_against` ponderados por equipo. Datos en
   `data/xg.csv` (16.688 partidos del CSV del usuario → **34/48 participantes con xG**; los 3
   anfitriones — sin clasificación — y varias selecciones africanas/OFC vienen sin xG en este
   dataset → caen a **fuerza pura** vía el guardarraíl `None`, comportamiento correcto).
   `TeamRatings.has_xg=True`: la mezcla xG del marcador (`predict._score_lambdas`) **ya está activa**.
2. ✅ **HECHO.** `scripts/calibrate_xg.py` (backtest **point-in-time** sobre la misma ventana
   de test del ML: el xG de cada partido se agrega solo con partidos anteriores a su fecha;
   574/1616 partidos con xG en ambas selecciones). Resultado: óptimo en `XG_BLEND_W=0.3`
   (`w>=0.5` degradaba — confirmada la inflación europea) y `XG_DECAY_PER_DAY=0.0006`
   (decaimiento suave; datos escasos). En total: 0.848→0.853 pts/partido (+0.005); sobre el
   subconjunto con xG, 0.841→0.854 (exacto 12.4%→12.9%). Como la matriz se reconcilia con el
   1X2 del ML, la ganancia es en MARCADOR EXACTO. Mejora posible (BAJA): mezcla ponderada por `n_matches`.

**Prioridad MEDIA:**
3. ✅ **HECHO.** **Bracket oficial 2026** en `config/wc2026.py` (`BRACKET_R32`, IDs P73..P88,
   `third_assignment_table` con las 495 combinaciones de terceros, matching válido determinista)
   y usado en `montecarlo` (`_build_official_bracket`): las eliminatorias respetan el cuadro real
   (cada plaza por posición de grupo; terceros a su plaza). Favoritos sanos (España ~22% campeón).
   Mejora "futuros" y "hasta qué ronda llega". Grupos no oficiales → fallback `bracket_seed_order`.
4. ✅ **HECHO.** **Helper de eliminatorias por ronda** (`src/bracket.py`): resuelve el cuadro
   oficial a equipos (standings reales o proyectados del Montecarlo), encadena R32→Final y da la
   **tabla batch de marcadores óptimos** por ronda (marcador a 90', `avanza`=P(gana)+½·empate,
   comodines opcionales). Dashboard tab3: selector de ronda + modo Proyectado/Real. Tests en
   `tests/test_bracket.py`. (Refinamiento futuro: fijar rondas ya jugadas con resultados KO reales,
   hoy R16+ son siempre proyectadas.)
5. ✅ **HECHO.** **Ajuste de fase** del total de goles en Montecarlo: `lam` usa la base de cada
   fase (grupos `base_total`, knockouts `base_total·0.92` vía `GOAL_STAGE_FACTOR`). Total de goles
   del torneo ~284→~277. ET/penales no modelados (marcador a 90').

**Prioridad BAJA / opcional:**
6. Heurística opcional para goleador/balón/sorpresa (hoy manual; sin datos de jugadores).
7. Reglas (parcial): **marcador knockout a 90'** (confirmado por el usuario → un empate es pick
   válido). Comodines en eliminatorias: regla aún incierta, pero el helper los soporta como
   **opcional** (`recommend_round(..., comodines=...)`) por si se permiten.
8. Persistir/cachear forma+Elo del `Predictor` para acelerar arranque del dashboard.

## 9. Flujo durante el torneo
Tras cada jornada: añadir resultados (dashboard → Ingreso de datos o `data/wc2026_results.csv`)
→ "Recalcular" (Elo/forma/Montecarlo en vivo) → `python scripts/train.py` para refrescar
ML/Poisson → re-predecir la siguiente ronda en la página Quiniela.

## 10. Decisiones/supuestos clave
- Sin APIs (todo CSV + manual). Sin apuestas. Recencia dura ≥2018 para el ML.
- Fuerza = Elo (todo el histórico) mezclado 70/30 con FIFA (ancla). Anfitriones = locales.
- xG como **señal mezclada** (escaso en selecciones), no único motor.
- Goleador/balón/sorpresa = manual. Optimización por componente (no "escenario más probable").
