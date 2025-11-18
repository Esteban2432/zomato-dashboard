import os
import requests
import streamlit as st
import pandas as pd
import altair as alt

# ============================
# Configuración del backend
# ============================
# Prioriza el valor en st.secrets (para Streamlit Cloud)
# Si no existe, usa la variable de entorno (para desarrollo local)
# Y si tampoco existe, usa el localhost como fallback
if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

BACKEND_URL = BACKEND_URL.rstrip("/")  # limpia posibles / al final

# ============================
# Configuración de la interfaz
# ============================
st.set_page_config(page_title="Zomato — Mercado Gastronómico", layout="wide")
st.title("🍽️ Zomato — Mercado Gastronómico")
st.caption(f"🔗 Backend conectado a: {BACKEND_URL}")

# ------------ Helpers ------------
def build_params(locations, rest_types, cuisines, online_order, book_table, rate_range, cost_range, bins=None):
    params = []
    for v in (locations or []):  params.append(("locations", v))
    for v in (rest_types or []): params.append(("rest_types", v))
    for v in (cuisines or []):   params.append(("cuisines", v))

    if online_order is not None: params.append(("online_order", str(bool(online_order))))
    if book_table  is not None:  params.append(("book_table", str(bool(book_table))))
    if rate_range:
        params += [("rate_min", rate_range[0]), ("rate_max", rate_range[1])]
    if cost_range:
        params += [("cost_min", cost_range[0]), ("cost_max", cost_range[1])]
    if bins is not None:
        params.append(("bins", bins))
    return params

def get_json(path, params=None):
    r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ------------ Sidebar: Filtros ------------
with st.sidebar:
    st.header("🔎 Filtros")

    f = get_json("/filters")
    loc = st.multiselect("Ubicación", options=f["locations"])
    rtype = st.multiselect("Tipo de restaurante", options=f["rest_types"])
    cuis = st.multiselect("Cocinas", options=f["cuisines"])

    online = st.selectbox(
        "Online order",
        options=[None, True, False],
        format_func=lambda x: {None:"(Todos)", True:"Sí", False:"No"}[x]
    )
    book   = st.selectbox(
        "Book table",
        options=[None, True, False],
        format_func=lambda x: {None:"(Todos)", True:"Sí", False:"No"}[x]
    )

    rate_rng = st.slider("Rango de rating", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.1)
    st.caption("ℹ️ Filtra restaurantes por su calificación promedio (0 a 5).")

    cost_rng = st.slider("Rango de costo (para 2)", min_value=0.0, max_value=4000.0, value=(0.0, 0.0), step=50.0)
    st.caption("ℹ️ Define el rango de costos (para 2 personas). Si lo dejas en 0–0, no filtra por costo.")

    bins = st.slider("Bins histograma", 5, 40, 20)
    st.caption("ℹ️ Número de columnas del histograma en la Vista 1.")

params = build_params(loc, rtype, cuis, online, book, rate_rng, cost_rng, bins)

tabs = st.tabs([
    "⭐ Calificaciones",
    "💵 Costos vs Rating",
    "🍳 Cocinas y Tipos",
    "📍 Benchmark",
    "🛒 Adopción de Servicios",
    "💬 Opinión y Popularidad"
])

with st.sidebar:
    st.markdown("### 📄 Documentación")

    manual_path = os.path.join(os.path.dirname(__file__), "Manual_Usuario.pdf")

    try:
        with open(manual_path, "rb") as file:
            st.download_button(
                label="Descargar Manual de Usuario",
                data=file,
                file_name="Manual_Usuario_Zomato.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.warning("No se encontró el archivo del Manual de Usuario en el servidor.")


# ------------ Vista 1 ------------
with tabs[0]:
    st.subheader("📊 Visión General")

    try:
        k = get_json("/kpis", params=params)
    except Exception as e:
        st.error(f"Error obteniendo KPIs: {e}")
        st.stop()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Restaurantes", k["total_restaurants"])
    c2.metric("Rating promedio", k["rating_avg"])
    c3.metric("Costo promedio (2)", k["cost_for_two_avg"])
    c4.metric("% Online order", k["pct_online_order"])
    c5.metric("% Book table", k["pct_book_table"])
    c6.metric("Cocinas únicas", k["unique_cuisines"])

    st.markdown("### Distribución de ratings")
    try:
        h = get_json("/ratings_distribution", params=params)
        if h["bins"]:
            dfh = pd.DataFrame({"bin": h["bins"][:-1], "count": h["counts"]})
            chart = alt.Chart(dfh).mark_bar().encode(
                x=alt.X("bin:Q", bin=alt.Bin(step=(5/len(dfh))), title="Rating"),
                y=alt.Y("count:Q", title="Frecuencia"),
                tooltip=["bin","count"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No hay datos para el histograma con los filtros actuales.")
    except Exception as e:
        st.error(f"Error en histograma: {e}")

# ------------ Vista 2 ------------
with tabs[1]:
    st.subheader("💵 Costos vs Rating")

    st.markdown("""
> **Descripción de la vista:**  
> Esta sección analiza la relación entre el **costo promedio para dos personas** y la **calificación (rating)** de los restaurantes.  
> Permite identificar si los locales más costosos obtienen mejores puntuaciones, y destacar las zonas con **mejor relación calidad-precio**.
>
> Usa los controles de abajo para ajustar:
> - **Percentil de costo:** excluye los precios extremos para una vista más clara.  
> - **Límite de puntos:** controla cuántos restaurantes se grafican para mantener el rendimiento.
""")

    # Controles adicionales
    st.markdown("##### Parámetros de análisis")
    pclip = st.slider(
        "Excluir outliers por percentil de costo", 
        0.0, 0.99, 0.95, 0.01, help="Valores mayores al percentil se excluyen del análisis."
    )
    st.caption(f"ℹ️ Se excluye el {round((1 - pclip) * 100, 1)}% de los restaurantes más caros para evitar que distorsionen la vista.")

    scatter_max = st.slider(
        "Límite de puntos en el gráfico", 
        200, 5000, 1200, 100, help="Máximo de puntos mostrados en el scatter para mantener el rendimiento."
    )
    st.caption("ℹ️ Este límite solo afecta la visualización del gráfico; los KPIs se calculan con todos los datos filtrados.")

    # --- Llamada al backend ---
    try:
        ps = get_json("/price_stats", params=params + [
            ("pclip", pclip),
            ("scatter_max", scatter_max)
        ])
    except Exception as e:
        st.error(f"Error obteniendo estadísticas de precios: {e}")
        st.stop()

    if not ps.get("available", False):
        st.info(ps.get("reason", "Datos no disponibles."))
    else:
        s = ps["summary"]
        st.markdown("### Resumen de costos")

        # KPIs principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Costo promedio (para 2 personas)", s.get("mean"))
        c2.metric("📊 Mediana del costo", s.get("median"))
        c3.metric("📈 Rango de costos", f"{s.get('min')} – {s.get('max')}")
        c4.metric("🔗 Correlación entre costo y rating", s.get("corr_cost_rating"))

        # --- Gráfico principal: Scatter + línea de tendencia ---
        scatter_df = pd.DataFrame(ps["scatter"])
        if not scatter_df.empty:
            base = alt.Chart(scatter_df).encode(
                x=alt.X("cost:Q", title="Costo para 2"),
                y=alt.Y("rating:Q", title="Rating"),
                tooltip=[
                    alt.Tooltip("cost:Q", title="Costo para 2"),
                    alt.Tooltip("rating:Q", title="Rating")
                ]
            )
            scatter = base.mark_circle(size=60, opacity=0.3, color="#66b3ff")
            trend = base.transform_regression("cost", "rating").mark_line(color="red")
            st.altair_chart((scatter + trend).properties(height=380), use_container_width=True)

            st.caption("""
🟦 Cada punto representa un restaurante.  
La línea roja indica la **tendencia general**: si sube, los locales más caros tienden a tener mejor calificación;  
si es plana, el costo no influye significativamente en el rating.
""")
        else:
            st.info("No hay suficientes datos para el gráfico.")

        # --- Tablas de insights por ubicación ---
        st.markdown("### Insights por ubicación")
        col_a, col_b = st.columns(2)
        col_a.caption("🏙️ Top ubicaciones más **caras**")
        col_a.dataframe(pd.DataFrame(ps["expensive_locations"]))
        col_b.caption("💡 Mejores **relaciones valor** (rating/costo)")
        col_b.dataframe(pd.DataFrame(ps["best_value_locations"]))

        # --- Información adicional ---
        st.caption(f"Columna de costo utilizada: `{s.get('cost_col')}`")
        if s.get("pclip_used") is not None:
            st.caption(f"Outliers excluidos a partir del percentil: **{s.get('pclip_used')}**")

# ------------ Vista 3 ------------
with tabs[2]:
    st.subheader("🍳 Cocinas y Tipos")

    st.markdown("""
> **Descripción de la vista:**  
> Explora qué cocinas y tipos de restaurante son más frecuentes, cuáles obtienen **mejores calificaciones** y cómo se relacionan entre sí.
> Usa los filtros de la izquierda y los parámetros de abajo para ajustar los resultados.
""")

    # Controles propios de la vista
    colp = st.columns(3)
    with colp[0]:
        top_n = st.slider("Top N", 5, 20, 10, 1)
        st.caption("ℹ️ Cantidad de categorías a mostrar en los gráficos (e.g., Top 10 cocinas por conteo o rating).")
    with colp[1]:
        min_n = st.slider("Mínimo de restaurantes por grupo", 1, 20, 5, 1, help="Evita destacar grupos con muy pocos locales.")
        st.caption(f"ℹ️ Solo se incluyen cocinas/tipos con al menos **{min_n}** restaurantes (mejora la representatividad).")
    with colp[2]:
        top_cuisines_heat = st.slider("Cocinas a mostrar en heatmap", 5, 30, 15, 1)
        st.caption("ℹ️ Limita la matriz a las cocinas más frecuentes para que el heatmap sea legible.")

    # --- Llamadas al backend ---
    try:
        cs = get_json("/cuisine_stats", params=params + [("top_n", top_n), ("min_n", min_n)])
        rs = get_json("/resttype_stats", params=params + [("top_n", top_n), ("min_n", min_n)])
        mx = get_json("/cuisine_resttype_matrix", params=params + [("min_n", min_n), ("top_cuisines", top_cuisines_heat)])
    except Exception as e:
        st.error(f"Error obteniendo estadísticas de cocinas/tipos: {e}")
        st.stop()

    c1, c2 = st.columns(2)

    # Top cocinas por conteo
    with c1:
        st.markdown("### Top cocinas por **cantidad**")
        df_count = pd.DataFrame(cs["by_count"])
        if not df_count.empty:
            chart = alt.Chart(df_count).mark_bar().encode(
                x=alt.X("count:Q", title="Restaurantes"),
                y=alt.Y("cuisine:N", sort="-x", title="Cocina"),
                tooltip=["cuisine", "count", "mean_rate", "mean_cost"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Sin datos para cocinas (conteo).")

    # Top cocinas por rating medio (con mínimo n)
    with c2:
        st.markdown(f"### Top cocinas por **rating promedio** (mín. n = {min_n})")
        df_rate = pd.DataFrame(cs["by_rating"])
        if not df_rate.empty:
            chart = alt.Chart(df_rate).mark_bar().encode(
                x=alt.X("mean_rate:Q", title="Rating promedio"),
                y=alt.Y("cuisine:N", sort="-x", title="Cocina"),
                tooltip=["cuisine", "count", "mean_rate", "mean_cost"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Sin datos para cocinas (rating).")

    st.markdown("### Tipos de restaurante con mejor desempeño")
    df_rt = pd.DataFrame(rs)
    if not df_rt.empty:
        chart = alt.Chart(df_rt).mark_bar().encode(
            x=alt.X("mean_rate:Q", title="Rating promedio"),
            y=alt.Y("rest_type:N", sort="-x", title="Tipo de restaurante"),
            color=alt.value("#88c0ff"),
            tooltip=["rest_type", "n", "mean_rate", "mean_cost"]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Sin datos para tipos de restaurante.")

    st.markdown("### Mapa de calor: Cocina × Tipo (rating promedio)")
    df_mx = pd.DataFrame(mx)
    if not df_mx.empty:
        heat = alt.Chart(df_mx).mark_rect().encode(
            x=alt.X("rest_type:N", title="Tipo de restaurante"),
            y=alt.Y("cuisine:N", title="Cocina"),
            color=alt.Color("mean_rate:Q", title="Rating prom."),
            tooltip=["cuisine", "rest_type", "n", "mean_rate"]
        ).properties(height=420)
        st.altair_chart(heat, use_container_width=True)
        st.caption("Cada celda muestra el **rating promedio** para la combinación Cocina × Tipo (solo combinaciones con mínimo n seleccionado).")
    else:
        st.info("No hay suficientes combinaciones Cocina × Tipo para el heatmap.")

# ------------ Vista 4 ------------
with tabs[3]:
    st.subheader("📍 Benchmark de Ubicaciones")

    st.markdown("""
> **Descripción de la vista:**  
> Compara las ubicaciones (zonas o barrios) según el **número de restaurantes**, su **costo promedio** y el **rating medio**.  
> Permite identificar áreas con mayor oferta, zonas premium y ubicaciones con mejor equilibrio calidad/precio.
""")

    # Parámetros
    c1, c2 = st.columns(2)
    min_n = c1.slider("Mínimo de restaurantes por ubicación", 1, 20, 5, 1)
    top_n = c2.slider("Número de ubicaciones a mostrar (Top N)", 5, 30, 15, 1)
    st.caption(f"ℹ️ Solo se muestran ubicaciones con al menos **{min_n}** restaurantes para garantizar resultados representativos.")
    st.caption(f"ℹ️ Se listan las **Top {top_n}** ubicaciones según la métrica por defecto (cantidad).")

    try:
        lb = get_json("/location_benchmark", params=params + [("min_n", min_n), ("top_n", top_n)])
    except Exception as e:
        st.error(f"Error obteniendo datos del benchmark: {e}")
        st.stop()

    df_loc = pd.DataFrame(lb["locations"])

    if not df_loc.empty:
        # KPIs globales
        st.markdown("### Métricas agregadas por ubicación")
        st.dataframe(df_loc)

        # Gráfico 1: Cantidad de restaurantes
        st.markdown("### 🏙️ Top ubicaciones por cantidad de restaurantes")
        chart1 = alt.Chart(df_loc).mark_bar().encode(
            x=alt.X("n:Q", title="Número de restaurantes"),
            y=alt.Y("location:N", sort="-x", title="Ubicación"),
            tooltip=["location", "n", "mean_cost", "mean_rate"]
        ).properties(height=320)
        st.altair_chart(chart1, use_container_width=True)

        # Gráfico 2: Scatter costo vs rating
        st.markdown("### 💰 Relación costo–rating por ubicación")
        df_scat = pd.DataFrame(lb["scatter"])
        chart2 = alt.Chart(df_scat).mark_circle(size=100).encode(
            x=alt.X("mean_cost:Q", title="Costo promedio (para 2)"),
            y=alt.Y("mean_rate:Q", title="Rating promedio"),
            size=alt.Size("n:Q", title="Cantidad de restaurantes"),
            color=alt.Color("location:N", legend=None),
            tooltip=["location", "mean_cost", "mean_rate", "n"]
        ).properties(height=400)
        st.altair_chart(chart2, use_container_width=True)

        st.caption("""
Cada burbuja representa una **ubicación**:
- Eje X: costo promedio  
- Eje Y: rating promedio  
- Tamaño: cantidad de restaurantes  
Las ubicaciones más arriba y a la izquierda ofrecen **mejor calidad a menor precio**.
""")
    else:
        st.info("No hay suficientes datos para mostrar el benchmark.")

# ------------ Vista 5 ------------
with tabs[4]:
    st.subheader("🛒 Adopción de Servicios (Online / Book)")

    st.markdown("""
> **Descripción de la vista:**  
> Analiza cómo los restaurantes adoptan servicios digitales como **pedidos online** y **reservas de mesa**,  
> y cómo esta adopción se relaciona con su **popularidad y calificación promedio**.
""")

    # ---- Controles ----
    top_n = st.slider("Top N (ubicaciones y tipos)", 5, 200, 10, 1)
    st.caption(f"ℹ️ Se muestran las **Top {top_n}** categorías (ubicaciones y tipos) por tamaño del grupo.")

    min_n = st.slider("Mínimo de restaurantes por grupo", 1, 20, 5, 1)
    st.caption(f"ℹ️ Los grupos con menos de **{min_n}** se marcan como pequeños (no se eliminan).")

    show_all = st.checkbox("Mostrar **todos** los grupos (ignorar Top N)", value=False)
    highlight_small = st.checkbox("Resaltar grupos pequeños (< mínimo)", value=True)
    st.caption("ℹ️ Si activas *Mostrar todos*, se ignora el límite Top N y se muestran todos los grupos disponibles.")

    top_n_param = 0 if show_all else top_n  # 0 => backend no recorta

    # ---- Llamada al backend ----
    try:
        av = get_json("/availability_stats", params=params + [("top_n", top_n_param), ("min_n", min_n)])
    except Exception as e:
        st.error(f"Error obteniendo estadísticas de adopción: {e}")
        st.stop()

    # ---- KPIs generales ----
    k = av["kpis"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📱 % Online Order", f"{k['pct_online']}%")
    c2.metric("📅 % Book Table", f"{k['pct_book']}%")
    c3.metric("🧩 Ambos servicios", f"{k['pct_both']}%")
    c4.metric("⭐ Rating (con online / sin)", f"{k['avg_rating_online']} / {k['avg_rating_no_online']}")

    # ---- Adopción por ubicación ----
    st.markdown("### 🏙️ Adopción por ubicación")
    df_loc = pd.DataFrame(av["by_location"])
    if not df_loc.empty:
        if "is_small" not in df_loc.columns:
            df_loc["is_small"] = df_loc["n"] < min_n  # respaldo si backend antiguo

        if not highlight_small:
            df_loc = df_loc[df_loc["n"] >= min_n]

        opacity_expr = alt.condition(
            alt.datum.is_small == True,
            alt.value(0.25 if highlight_small else 0.85),
            alt.value(1.0)
        )

        chart = alt.Chart(df_loc).mark_bar().encode(
            y=alt.Y("location:N", sort="-x", title="Ubicación"),
            x=alt.X("pct_online:Q", title="% Online Order"),
            opacity=opacity_expr,
            color=alt.condition(alt.datum.is_small == True, alt.value("#A5D6A7"), alt.value("#4CAF50")),
            tooltip=["location", "n", "pct_online", "pct_book", "mean_rate", "is_small"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hay datos suficientes por ubicación.")

    # ---- Adopción por tipo de restaurante ----
    st.markdown("### 🍽️ Adopción por tipo de restaurante")
    df_type = pd.DataFrame(av["by_resttype"])
    if not df_type.empty:
        if "is_small" not in df_type.columns:
            df_type["is_small"] = df_type["n"] < min_n

        if not highlight_small:
            df_type = df_type[df_type["n"] >= min_n]

        opacity_expr2 = alt.condition(
            alt.datum.is_small == True,
            alt.value(0.25 if highlight_small else 0.85),
            alt.value(1.0)
        )

        chart2 = alt.Chart(df_type).mark_bar().encode(
            y=alt.Y("rest_type:N", sort="-x", title="Tipo de restaurante"),
            x=alt.X("pct_book:Q", title="% Book Table"),
            opacity=opacity_expr2,
            color=alt.condition(alt.datum.is_small == True, alt.value("#FFCC80"), alt.value("#FF9800")),
            tooltip=["rest_type", "n", "pct_online", "pct_book", "mean_rate", "is_small"]
        ).properties(height=300)
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No hay datos suficientes por tipo de restaurante.")

    # ---- Matriz Online vs Book ----
    st.markdown("### 🔄 Combinación de servicios (matriz 2×2)")
    df_matrix = pd.DataFrame(av["matrix"])
    if not df_matrix.empty:
        heat = alt.Chart(df_matrix).mark_rect().encode(
            x=alt.X("book_table:N", title="Reserva de mesa"),
            y=alt.Y("online_order:N", title="Pedidos online"),
            color=alt.Color("mean_rate:Q", title="Rating promedio", scale=alt.Scale(scheme="greens")),
            tooltip=["online_order", "book_table", "n", "mean_rate"]
        ).properties(height=280)
        st.altair_chart(heat, use_container_width=True)
        st.caption("Cada celda muestra el **rating promedio** para la combinación Online × Book. El tamaño representa la cantidad de restaurantes.")
    else:
        st.info("No hay suficientes combinaciones para la matriz.")


# ------------ Vista 6 ------------
with tabs[5]:
    st.subheader("💬 Opinión Pública y Popularidad")

    st.markdown("""
> **Descripción de la vista:**  
> Examina cómo la **popularidad** (cantidad de votos) se relaciona con la **reputación** (calificación promedio).  
> Permite identificar si los restaurantes más conocidos son también los mejor calificados.
""")

    try:
        pop = get_json("/popularity_stats")
    except Exception as e:
        st.error(f"Error obteniendo estadísticas de popularidad: {e}")
        st.stop()

    # ---- KPIs principales ----
    k1, k2, k3 = st.columns(3)
    k1.metric("🏪 Total restaurantes", f"{pop['total']:,}")
    k2.metric("📊 Promedio de votos", f"{pop['avg_votes']:.2f}")
    corr_text = "N/A" if pop["corr_votes_rate"] is None else f"{pop['corr_votes_rate']:.3f}"
    k3.metric("🔗 Correlación votos–rating", corr_text)

    # Interpretación
    if pop["corr_votes_rate"] is not None:
        if pop["corr_votes_rate"] > 0.5:
            st.success("📈 Fuerte correlación positiva: los restaurantes con más votos tienden a tener mejores ratings.")
        elif pop["corr_votes_rate"] > 0.2:
            st.info("🟨 Correlación moderada: cierta relación entre popularidad y rating, pero con excepciones.")
        elif pop["corr_votes_rate"] >= 0:
            st.warning("📉 Correlación débil: la cantidad de votos no necesariamente refleja la calidad percibida.")
        else:
            st.error("❌ Correlación negativa: los más votados no siempre son los mejor calificados.")

    st.divider()

    # ---- Gráfico de dispersión ----
    st.markdown("### 🔹 Popularidad vs Calificación")

    df_scatter = pd.DataFrame(pop["scatter"])
    if not df_scatter.empty:
        chart = (
            alt.Chart(df_scatter)
            .mark_circle(size=70, opacity=0.6)
            .encode(
                x=alt.X("votes:Q", title="Cantidad de votos (popularidad)"),
                y=alt.Y("rate:Q", title="Calificación promedio"),
                color=alt.Color("rest_type:N", title="Tipo de restaurante"),
                size=alt.Size("approx_cost_for_two_people:Q", title="Costo promedio (para 2)", scale=alt.Scale(range=[30, 300])),
                tooltip=["name", "votes", "rate", "rest_type", "location", "approx_cost_for_two_people"]
            )
            .interactive()
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hay suficientes datos para mostrar la relación votos–rating.")

    # ---- Ranking de los más votados ----
    st.markdown("### 🏆 Top 10 Restaurantes más votados")
    df_top = pd.DataFrame(pop["top_voted"])
    if not df_top.empty:
        st.dataframe(df_top, use_container_width=True, hide_index=True)
    else:
        st.info("No se encontraron restaurantes con votos registrados.")
