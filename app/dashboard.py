"""Dashboard Streamlit — interfaz principal del modelo probabilístico WC 2026.

Páginas:
  1. Predicción de partido  — 1X2 (ML) + marcador (Poisson) + panel de mercado.
  2. Detección de valor +EV — cuotas manuales -> apuestas de valor + Kelly.
  3. Ranking Elo + FIFA      — fuerza de las selecciones.
  4. Simulación Montecarlo   — prob. de avance / campeón.
  5. Ingreso de datos        — añadir resultados de jornada y cuotas (manual).

Ejecutar:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings, wc2026
from src import bracket, data_loader, elo, fifa, montecarlo, quiniela as q
from src.predict import Predictor, format_prediction

st.set_page_config(page_title="Modelo WC 2026", page_icon="⚽", layout="wide")


# --------------------------------------------------------------------------- #
# Carga cacheada
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando modelo (Elo + forma + artefactos)...")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def get_ratings() -> pd.DataFrame | None:
    return elo.load_ratings()


@st.cache_data
def get_fifa() -> pd.DataFrame | None:
    return fifa.load_snapshot()


def artifacts_ready() -> bool:
    return settings.ML_MODEL_PKL.exists() and settings.POISSON_PARAMS_PKL.exists() \
        and settings.ELO_RATINGS_CSV.exists()


def clear_caches():
    st.cache_resource.clear()
    st.cache_data.clear()


# --------------------------------------------------------------------------- #
# Páginas
# --------------------------------------------------------------------------- #
def page_match(P: Predictor):
    st.header("🔮 Predicción de partido")
    teams = P.teams()
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Local / Equipo 1", teams, index=teams.index("Spain") if "Spain" in teams else 0)
    away = c2.selectbox("Visitante / Equipo 2", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)
    neutral = c3.checkbox("Sede neutral", value=True)
    importance = c3.selectbox("Tipo", ["world_cup", "continental", "world_cup_qual", "friendly"], index=0)

    if home == away:
        st.warning("Elige dos selecciones distintas.")
        return

    r = P.predict(home, away, neutral=neutral, importance=importance)
    ml = r["ml_1x2"]

    st.subheader("Probabilidades 1X2 (modelo ML — eje principal)")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Gana {home}", f"{ml['H']:.1%}")
    m2.metric("Empate", f"{ml['D']:.1%}")
    m3.metric(f"Gana {away}", f"{ml['A']:.1%}")

    colL, colR = st.columns(2)
    with colL:
        fig = go.Figure(go.Bar(
            x=[f"{home}", "Empate", f"{away}"],
            y=[ml["H"], ml["D"], ml["A"]],
            marker_color=["#2563eb", "#9ca3af", "#dc2626"],
            text=[f"{v:.1%}" for v in (ml["H"], ml["D"], ml["A"])], textposition="outside"))
        fig.update_layout(title="1X2 (ML)", yaxis_tickformat=".0%", height=320,
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Fuerza (Elo+FIFA): **{home}** {r['strength_home']:.0f}  ·  "
                   f"**{away}** {r['strength_away']:.0f}")

    with colR:
        # Pick recomendado para la quiniela: el marcador que MAXIMIZA puntos
        # esperados (3/1/0), no el más probable. Reusa la matriz ya reconciliada.
        pick = q.best_scoreline(r["score_matrix"])
        pi, pj = pick["score"]
        st.success(
            f"✅ **Pick recomendado (quiniela):** {home} **{pi} - {pj}** {away}\n\n"
            f"Maximiza puntos esperados ({pick['ep']:.2f} pts) · "
            f"P(marcador exacto) {pick['p_exact']:.1%} · "
            f"P(acierto de resultado) {pick['p_outcome']:.0%}")

        ms = r["most_likely_score"]
        st.markdown(f"**Marcador más probable (modal):** {home} **{ms[0]} - {ms[1]}** {away}  "
                    f"(λ {r['lambda_home']:.2f} - {r['lambda_away']:.2f})")
        st.markdown("**Top marcadores más probables (Poisson + Dixon-Coles):**")
        st.caption("Probabilidad de cada **marcador exacto** — NO de quién gana. "
                   "Suman poco porque hay muchos marcadores posibles; para el 1X2 "
                   "(quién gana) usa el panel de la izquierda.")
        st.table(pd.DataFrame(
            [{"Marcador": f"{s[0]}-{s[1]}", "P(exacto)": f"{p:.1%}"} for s, p in r["top_scores"]]))
        st.caption(f"Over 2.5: {r['over_2_5']:.0%}  ·  BTTS: {r['btts_yes']:.0%}  ·  "
                   f"Poisson 1X2 (cross-check): {r['poisson_1x2']['H']:.0%}/"
                   f"{r['poisson_1x2']['D']:.0%}/{r['poisson_1x2']['A']:.0%}")


def page_ranking(P: Predictor):
    st.header("📊 Ranking — Fuerza (Elo + FIFA)")
    ratings = get_ratings()
    if ratings is None:
        st.warning("No hay ratings. Ejecuta `python scripts/train.py`.")
        return
    fifa_df = get_fifa()
    if fifa_df is not None:
        ratings = ratings.merge(fifa_df[["team", "rank", "points"]]
                                .rename(columns={"rank": "rank_FIFA", "points": "pts_FIFA"}),
                                on="team", how="left")
    top = st.slider("Mostrar top N", 10, 60, 30)
    show = ratings.head(top)
    fig = px.bar(show, x="strength" if "strength" in show else "elo", y="team",
                 orientation="h", height=max(400, top * 18),
                 labels={"strength": "Fuerza (Elo+FIFA)", "team": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=30, l=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show, use_container_width=True, hide_index=True)


def page_montecarlo(P: Predictor):
    st.header("🎲 Simulación de Montecarlo del Mundial")
    st.caption(f"Grupos actuales (placeholder editable en `config/wc2026.py`). "
               f"Usa la fuerza en vivo (incluye resultados manuales cargados).")

    with st.expander("Ver grupos"):
        gd = pd.DataFrame({g: teams for g, teams in wc2026.GROUPS.items()}).T
        gd.columns = [f"Equipo {i+1}" for i in range(gd.shape[1])]
        st.dataframe(gd, use_container_width=True)

    c1, c2 = st.columns([1, 3])
    n = c1.select_slider("Nº de simulaciones", [2000, 5000, 10000, 20000, 50000], value=20000)
    if c1.button("Simular", type="primary"):
        with st.spinner(f"Simulando {n:,} mundiales..."):
            df = montecarlo.simulate(n_sims=n, strength=P.strength_map())
        st.session_state["mc"] = df
    if "mc" in st.session_state:
        df = st.session_state["mc"]
        fig = px.bar(df.head(20), x="P_Campeon", y="team", orientation="h",
                     labels={"P_Campeon": "Prob. de ser campeón", "team": ""},
                     height=560, text=df.head(20)["P_Campeon"].map(lambda v: f"{v:.1%}"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat=".0%",
                          margin=dict(t=30, l=10))
        st.plotly_chart(fig, use_container_width=True)
        nice = df.copy()
        for col in ["P_grupo", "P_R32", "P_R16", "P_QF", "P_SF", "P_Final", "P_Campeon"]:
            nice[col] = (nice[col] * 100).round(1).astype(str) + "%"
        st.dataframe(nice, use_container_width=True, hide_index=True)


def page_quiniela(P: Predictor):
    st.header("🎯 Quiniela — recomendaciones óptimas")
    st.caption("Elige las predicciones que **maximizan tus puntos esperados** (no el marcador "
               "'más probable'). Reglas: exacto 3 / resultado 1 / fallo 0; comodines "
               "double ×2, triple ×3, all-in (12/5/−6). El marcador exacto tiene un techo "
               "(~10-18%): el modelo optimiza puntos, no garantiza aciertos.")
    tab1, tab2, tab3 = st.tabs(["⚽ Fase de grupos", "🏆 Futuros", "🔻 Eliminatorias (por ronda)"])

    # ---- Fase de grupos (de golpe) ----
    with tab1:
        st.markdown("Los **72 partidos** de grupos con el marcador óptimo y el plan de comodines.")
        if st.button("Calcular recomendación de grupos", type="primary"):
            with st.spinner("Optimizando 72 partidos..."):
                st.session_state["q_grupos"] = q.recommend_group_stage(P)
        if "q_grupos" in st.session_state:
            rec = st.session_state["q_grupos"]
            df, plan = rec["matches"], rec["comodines"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Puntos esperados (base)", f"{plan['baseline_ep']:.1f}")
            c2.metric("Con comodines", f"{plan['total_ep']:.1f}")
            c3.metric("Partidos", len(df))
            comod = {df.loc[i, "partido"]: t for i, t in plan["assignments"].items()}
            if comod:
                st.markdown("**Plan de comodines:** " +
                            " · ".join(f"`{t}` → {p}" for p, t in comod.items()))
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("`pick` = marcador óptimo · `modal` = marcador más probable "
                       "(a veces difieren: el óptimo evita la trampa del empate).")

    # ---- Futuros (Montecarlo) ----
    with tab2:
        st.markdown("Pronósticos pre-torneo con el Montecarlo (posiciones, terceros, placements, goles).")
        n = st.select_slider("Nº de simulaciones", [5000, 10000, 20000, 50000], value=20000)
        if st.button("Simular futuros", type="primary"):
            with st.spinner(f"Simulando {n:,} mundiales..."):
                df, extras = montecarlo.simulate(n_sims=n, strength=P.strength_map(),
                                                 return_extras=True)
                st.session_state["q_fut"] = (df, extras)
        if "q_fut" in st.session_state:
            mc_df, extras = st.session_state["q_fut"]
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Posiciones de grupo** (1º/2º/3º)")
                gp = q.group_position_picks(mc_df)
                st.dataframe(gp, use_container_width=True, hide_index=True)
                st.caption(f"Puntos esperados posiciones: {gp['E_pts'].sum():.1f}")
            with colB:
                st.markdown("**Campeón / Subcampeón / 3º / 4º**")
                st.dataframe(q.placement_picks(mc_df), use_container_width=True, hide_index=True)
                top, exp = q.best_thirds_picks(mc_df)
                st.markdown("**Mejores terceros (pick)**")
                st.dataframe(top, use_container_width=True, hide_index=True)
                st.caption(f"Puntos esperados mejores terceros: {exp}")
            tg = q.total_goals_pick(extras)
            st.metric("Total de goles del torneo (estimación)", tg["estimacion"],
                      help=f"Rango p10-p90: {tg['p10']}-{tg['p90']}")
            st.markdown("**Manuales (sin datos de jugadores → los eliges tú):**")
            m1, m2, m3 = st.columns(3)
            m1.text_input("Goleador del torneo")
            m2.text_input("Balón de oro")
            m3.text_input("Equipo sorpresa")

    # ---- Eliminatorias por ronda (cuadro oficial 2026) ----
    with tab3:
        st.markdown("Tabla de **marcadores óptimos de una ronda** del **cuadro oficial 2026** "
                    "(a 90'). Las llaves se resuelven solas; en R16+ los cruces son proyectados.")
        c1, c2, c3 = st.columns([2, 1, 1])
        modo = c1.radio("Cruces desde", ["Proyectado (Montecarlo)", "Resultados reales de grupos"],
                        key="ko_modo")
        ronda = c2.selectbox("Ronda", bracket.KO_ROUNDS, key="ko_ronda")
        usar_com = c3.checkbox("Comodines (tentativo)", value=False, key="ko_comod",
                               help="Asigna comodines a la ronda por si las reglas los permiten "
                                    "también en eliminatorias.")

        standings = None
        if modo.startswith("Proyectado"):
            if "q_fut" in st.session_state:
                standings = bracket.standings_from_projection(st.session_state["q_fut"][0])
            else:
                st.info("Primero corre **Simular futuros** en la pestaña 🏆 Futuros.")
        else:
            matches = data_loader.load_matches()
            wc = matches[(matches["tournament"] == "FIFA World Cup") & (matches["year"] == 2026)]
            wc = wc.dropna(subset=["home_score", "away_score"])
            if wc.empty:
                st.warning("Aún no hay resultados del Mundial 2026 cargados (página 📝 Ingreso de "
                           "datos): se usaría el orden del sorteo. Mejor usa el modo **Proyectado**.")
            standings = bracket.standings_from_matches(wc)

        if standings is not None and st.button("Calcular ronda", type="primary", key="ko_calc"):
            with st.spinner(f"Resolviendo {ronda} y rondas previas..."):
                w, r, th, qual = standings
                rounds = bracket.project_rounds(P, w, r, th, qual)
                comod = {"triple": 2, "double": 2, "allin": 2} if usar_com else None
                rec = bracket.recommend_round(P, rounds[ronda], comodines=comod)
                st.session_state["ko_rec"] = (ronda, len(rounds[ronda]), rec)

        if "ko_rec" in st.session_state:
            rnd_done, n_cruces, rec = st.session_state["ko_rec"]
            st.caption(f"Ronda **{rnd_done}** — {n_cruces} cruces. `avanza` = quién pasa "
                       "(P(gana)+½·empate); marcador a 90' (un empate es pick válido).")
            st.dataframe(rec["matches"], use_container_width=True, hide_index=True)
            if "comodines" in rec:
                plan = rec["comodines"]
                st.caption(f"Comodines (tentativo): E[pts] {plan['baseline_ep']:.1f} "
                           f"→ {plan['total_ep']:.1f}")

        with st.expander("🎯 Marcador óptimo de un cruce manual cualquiera"):
            teams = P.teams()
            d1, d2, d3 = st.columns([2, 2, 1])
            h = d1.selectbox("Equipo 1 (local)", teams, key="qh")
            a = d2.selectbox("Equipo 2", teams, index=1, key="qa")
            neutral = d3.checkbox("Sede neutral", value=True, key="qn")
            stage = d3.selectbox("Fase", ["knockout", "group"], key="qs")
            if h != a:
                m = P.score_matrix_for(h, a, neutral=neutral, stage=stage)
                ev = q.evaluate_match(m)
                n_, al = ev["normal"], ev["allin"]
                res = {"H": h, "D": "Empate", "A": a}[n_["outcome"]]
                st.success(f"**Pick óptimo: {h} {n_['score'][0]}-{n_['score'][1]} {a}**  "
                           f"(E[pts]={n_['ep']:.2f} · P_exacto={n_['p_exact']:.0%} · "
                           f"P_resultado={n_['p_outcome']:.0%} → {res})")
                st.caption(f"Modal: {ev['modal'][0]}-{ev['modal'][1]}  ·  all-in óptimo: "
                           f"{al['score'][0]}-{al['score'][1]} (E[pts]={al['ep']:.2f})")

        with st.expander("⚙️ Ajuste manual de fuerza (día-antes: lesiones, rotación)"):
            st.caption("Suma/resta puntos de fuerza a un equipo (≈ +65 = ventaja de local).")
            adj_team = st.selectbox("Equipo", teams, key="adjteam")
            adj_val = st.slider("Ajuste (puntos)", -200, 200, int(P.adjustments().get(adj_team, 0)), 5)
            if st.button("Aplicar ajuste"):
                P.set_adjustment(adj_team, adj_val)
                st.session_state.pop("q_grupos", None)   # invalidar cálculos previos
                st.session_state.pop("q_fut", None)
                st.success(f"Ajuste aplicado a {adj_team}: {adj_val:+d}. Recalcula las pestañas.")
            if P.adjustments():
                st.write("Ajustes activos:", P.adjustments())


def page_data():
    st.header("📝 Ingreso de datos (manual)")
    teams = data_loader.all_teams()

    st.subheader("Añadir resultado de una jornada del Mundial")
    st.caption("Se guarda en `data/wc2026_results.csv` y actualiza Elo/forma al recalcular.")
    with st.form("res"):
        c = st.columns(5)
        f_date = c[0].date_input("Fecha")
        f_home = c[1].selectbox("Local", teams, key="rh")
        f_away = c[2].selectbox("Visitante", teams, key="ra")
        f_hs = c[3].number_input("Goles local", 0, 30, 0)
        f_as = c[4].number_input("Goles visit.", 0, 30, 0)
        f_tour = st.text_input("Torneo", "FIFA World Cup")
        if st.form_submit_button("Guardar resultado"):
            row = {"date": f_date, "home_team": f_home, "away_team": f_away,
                   "home_score": f_hs, "away_score": f_as, "tournament": f_tour,
                   "city": "", "country": "", "neutral": True}
            _append_csv(settings.WC2026_RESULTS_CSV, row)
            st.success(f"Guardado: {f_home} {f_hs}-{f_as} {f_away}. "
                       "Pulsa *Recalcular* para actualizar el modelo.")

    st.divider()
    if st.button("🔄 Recalcular (limpiar caché y releer datos)"):
        clear_caches()
        st.success("Caché limpiada. El modelo se recalculará con los datos nuevos.")
        st.rerun()
    st.caption("Nota: el ML y el Poisson son artefactos; para reentrenarlos con los nuevos "
               "resultados ejecuta `python scripts/train.py`. El Elo, la forma y el "
               "Montecarlo sí se actualizan al recalcular.")


def _append_csv(path: Path, row: dict):
    df_new = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.sidebar.title("⚽ Modelo WC 2026")
    st.sidebar.caption("Predicción 1X2 + marcador exacto para la quiniela")
    page = st.sidebar.radio("Navegación", [
        "🎯 Quiniela",
        "🔮 Predicción de partido",
        "📊 Ranking Elo + FIFA",
        "🎲 Simulación Montecarlo",
        "📝 Ingreso de datos",
    ])
    st.sidebar.divider()

    if not artifacts_ready():
        st.error("Faltan artefactos del modelo. Ejecuta primero:\n\n"
                 "```\npython scripts/setup_data.py\npython scripts/fetch_fifa.py\n"
                 "python scripts/train.py\n```")
        return

    if page == "📝 Ingreso de datos":
        page_data()
        return

    P = get_predictor()
    if page == "🎯 Quiniela":
        page_quiniela(P)
    elif page == "🔮 Predicción de partido":
        page_match(P)
    elif page == "📊 Ranking Elo + FIFA":
        page_ranking(P)
    elif page == "🎲 Simulación Montecarlo":
        page_montecarlo(P)


if __name__ == "__main__":
    main()
