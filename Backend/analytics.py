import numpy as np
import pandas as pd

def _none_if_nan(x):
    try:
        return None if pd.isna(x) else float(x)
    except Exception:
        return None

# ---------- KPIs ----------
def get_kpis(df: pd.DataFrame) -> dict:
    total = int(len(df))

    rating_avg = _none_if_nan(round(pd.to_numeric(df.get("rate"), errors="coerce").mean(), 2))
    cost_avg   = _none_if_nan(round(pd.to_numeric(df.get("approx_cost_for_two_people"), errors="coerce").mean(), 0))

    def _pct_true(col):
        if col not in df.columns:
            return None
        s = df[col].astype("boolean")
        return None if s.notna().sum() == 0 else float((s.mean() * 100).round(1))

    pct_online = _pct_true("online_order")
    pct_book   = _pct_true("book_table")

    cuisines_div = None
    if "cuisines" in df.columns:
        cuisines_div = int(
            df["cuisines"].dropna().astype(str).str.split(",").explode().str.strip().nunique()
        ) if len(df) else 0

    return {
        "total_restaurants": total,
        "rating_avg": rating_avg,
        "cost_for_two_avg": cost_avg,
        "pct_online_order": pct_online,
        "pct_book_table": pct_book,
        "unique_cuisines": cuisines_div,
    }

# ---------- Filtros ----------
def get_filters(df: pd.DataFrame) -> dict:
    def _uniq(col):
        return sorted([x for x in df[col].dropna().astype(str).str.strip().unique() if x])

    return {
        "locations": _uniq("location") if "location" in df.columns else [],
        "rest_types": _uniq("rest_type") if "rest_type" in df.columns else [],
        "cuisines": sorted(
            df["cuisines"].dropna().astype(str).str.split(",").explode().str.strip().unique()
        ) if "cuisines" in df.columns else [],
    }

# ---------- Histograma ----------
def ratings_histogram(df: pd.DataFrame, bins: int = 20) -> dict:
    r = pd.to_numeric(df.get("rate"), errors="coerce").dropna()
    if r.empty:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(r, bins=bins, range=(0, 5))
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }

# ---------- Vista 2: Costos vs Rating ----------
def get_price_stats(df: pd.DataFrame, top_n: int = 5, pclip: float = 0.0, scatter_max: int = 1200) -> dict:
    cost_col = "approx_cost_for_two_people" if "approx_cost_for_two_people" in df.columns else None
    if not cost_col:
        return {
            "available": False,
            "reason": "No cost column.",
            "summary": {},
            "scatter": [],
            "expensive_locations": [],
            "best_value_locations": [],
        }

    # Cast numérico
    df = df.assign(
        _cost=pd.to_numeric(df[cost_col], errors="coerce"),
        _rate=pd.to_numeric(df.get("rate"), errors="coerce")
    ).dropna(subset=["_cost"])

    # 1) Clipping por percentil (outliers)
    pclip_used = None
    if isinstance(pclip, (int, float)) and 0 < float(pclip) < 1:
        lim = df["_cost"].quantile(float(pclip))
        df = df[df["_cost"] <= lim]
        pclip_used = float(pclip)

    # Series limpias
    cost_series   = df["_cost"]
    rating_series = df["_rate"]

    # 2) Correlación
    corr = None
    c = pd.concat([cost_series, rating_series], axis=1).dropna()
    if len(c) >= 2:
        corr = float(c.corr().iloc[0, 1].round(3))

    summary = {
        "count": int(cost_series.shape[0]),
        "mean": float(round(cost_series.mean(), 2)) if not cost_series.empty else None,
        "median": float(round(cost_series.median(), 2)) if not cost_series.empty else None,
        "min": float(cost_series.min()) if not cost_series.empty else None,
        "max": float(cost_series.max()) if not cost_series.empty else None,
        "corr_cost_rating": corr,
        "cost_col": cost_col,
    }

    # 3) Scatter (muestra limitada)
    scatter_df = pd.DataFrame({"cost": cost_series, "rating": rating_series}).dropna()
    if len(scatter_df) > scatter_max:
        scatter_df = scatter_df.sample(scatter_max, random_state=42)
    scatter = scatter_df.round(2).to_dict(orient="records")

    # Insights por ubicación (igual que antes)
    expensive_locations, best_value_locations = [], []
    if "location" in df.columns:
        by_loc = df.groupby("location", dropna=True).agg(
            mean_cost=("_cost", "mean"),
            mean_rate=("_rate", "mean"),
            n=("location", "size"),
        ).reset_index()
        expensive = by_loc.sort_values("mean_cost", ascending=False).head(top_n)
        expensive_locations = [
            {"location": r["location"],
             "mean_cost": float(round(r["mean_cost"], 2)) if pd.notna(r["mean_cost"]) else None,
             "mean_rate": float(round(r["mean_rate"], 2)) if pd.notna(r["mean_rate"]) else None,
             "n": int(r["n"])}
            for _, r in expensive.iterrows()
        ]
        by_loc = by_loc.assign(
            value_score=by_loc.apply(
                lambda x: (x["mean_rate"]/x["mean_cost"]) if pd.notna(x["mean_rate"]) and pd.notna(x["mean_cost"]) and x["mean_cost"]>0 else np.nan,
                axis=1
            )
        )
        value = by_loc.dropna(subset=["value_score"]).sort_values("value_score", ascending=False).head(top_n)
        best_value_locations = [
            {"location": r["location"], "value_score": float(round(r["value_score"], 3)),
             "mean_cost": float(round(r["mean_cost"], 2)) if pd.notna(r["mean_cost"]) else None,
             "mean_rate": float(round(r["mean_rate"], 2)) if pd.notna(r["mean_rate"]) else None,
             "n": int(r["n"])}
            for _, r in value.iterrows()
        ]

    # Nota: añadimos un campo auxiliar en summary SIN romper el schema (es opcional para el front)
    summary["pclip_used"] = pclip_used

    return {
        "available": True,
        "summary": summary,
        "scatter": scatter,
        "expensive_locations": expensive_locations,
        "best_value_locations": best_value_locations,
    }

# ---------- Vista 1: Top cocinas / tipos ----------
def top_cuisines(df: pd.DataFrame, top_n: int = 10):
    if "cuisines" not in df.columns or df.empty:
        return []
    s = (
        df["cuisines"].dropna().astype(str)
        .str.split(",").explode().str.strip()
    )
    g = s.to_frame(name="cuisine").assign(
        rate=pd.to_numeric(df.loc[s.index, "rate"], errors="coerce"),
    )
    agg = g.groupby("cuisine", dropna=True).agg(
        count=("cuisine", "size"),
        mean_rate=("rate", "mean"),
    ).reset_index()
    agg["mean_rate"] = agg["mean_rate"].round(2)
    agg = agg.sort_values(["count", "mean_rate"], ascending=[False, False]).head(top_n)
    return agg.to_dict(orient="records")

def top_rest_types(df: pd.DataFrame, top_n: int = 10):
    if "rest_type" not in df.columns or df.empty:
        return []
    g = df.groupby("rest_type", dropna=True).agg(
        count=("rest_type", "size"),
        mean_rate=("rate", "mean"),
    ).reset_index()
    g["mean_rate"] = g["mean_rate"].round(2)
    g = g.sort_values(["count", "mean_rate"], ascending=[False, False]).head(top_n)
    return g.to_dict(orient="records")


# ---------- Vista 3: Cocinas y Tipos ----------

def cuisine_stats(df: pd.DataFrame, top_n: int = 10, min_n: int = 5) -> dict:
    """
    Devuelve top de cocinas por conteo, por rating medio (con mínimo n) y por costo medio.
    """
    if df.empty or "cuisines" not in df.columns:
        return {"by_count": [], "by_rating": [], "by_cost": []}

    # expandir cocinas
    tmp = (
        df.assign(
            _rate=pd.to_numeric(df.get("rate"), errors="coerce"),
            _cost=pd.to_numeric(df.get("approx_cost_for_two_people"), errors="coerce"),
        )["cuisines"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .to_frame(name="cuisine")
    )
    tmp["_rate"] = pd.to_numeric(df.loc[tmp.index, "rate"], errors="coerce")
    tmp["_cost"] = pd.to_numeric(df.loc[tmp.index, "approx_cost_for_two_people"], errors="coerce")

    g = tmp.groupby("cuisine", dropna=True).agg(
        count=("cuisine", "size"),
        mean_rate=("_rate", "mean"),
        mean_cost=("_cost", "mean"),
    ).reset_index()
    g["mean_rate"] = g["mean_rate"].round(2)
    g["mean_cost"] = g["mean_cost"].round(2)

    by_count  = g.sort_values("count", ascending=False).head(top_n)
    by_rating = g[g["count"] >= int(min_n)].sort_values("mean_rate", ascending=False).head(top_n)
    by_cost   = g.sort_values("mean_cost", ascending=False).head(top_n)

    return {
        "by_count": by_count.to_dict(orient="records"),
        "by_rating": by_rating.to_dict(orient="records"),
        "by_cost": by_cost.to_dict(orient="records"),
    }


def resttype_stats(df: pd.DataFrame, top_n: int = 10, min_n: int = 5) -> list[dict]:
    """
    Rating y costo medios por tipo de restaurante.
    """
    if df.empty or "rest_type" not in df.columns:
        return []

    g = df.assign(
        _rate=pd.to_numeric(df.get("rate"), errors="coerce"),
        _cost=pd.to_numeric(df.get("approx_cost_for_two_people"), errors="coerce"),
    ).groupby("rest_type", dropna=True).agg(
        n=("rest_type", "size"),
        mean_rate=("_rate", "mean"),
        mean_cost=("_cost", "mean"),
    ).reset_index()

    g = g[g["n"] >= int(min_n)]
    g["mean_rate"] = g["mean_rate"].round(2)
    g["mean_cost"] = g["mean_cost"].round(2)
    g = g.sort_values(["mean_rate", "n"], ascending=[False, False]).head(top_n)
    return g.to_dict(orient="records")


def cuisine_resttype_matrix(df: pd.DataFrame, min_n: int = 3, top_cuisines: int = 15) -> list[dict]:
    """
    Matriz (cocina x tipo) con rating medio y n. Se limita a las cocinas más frecuentes.
    """
    if df.empty or "cuisines" not in df.columns or "rest_type" not in df.columns:
        return []

    base = df.assign(
        _rate=pd.to_numeric(df.get("rate"), errors="coerce")
    ).dropna(subset=["cuisines", "rest_type"])

    # top cocinas por conteo
    c_counts = (
        base["cuisines"].astype(str).str.split(",").explode().str.strip().value_counts()
    )
    top_cuis = set(c_counts.head(int(top_cuisines)).index)

    expl = (
        base.assign(cuisine=base["cuisines"].astype(str).str.split(","))
        .explode("cuisine")
        .assign(cuisine=lambda d: d["cuisine"].str.strip())
    )
    expl = expl[expl["cuisine"].isin(top_cuis)]

    g = expl.groupby(["cuisine", "rest_type"], dropna=True).agg(
        n=("rest_type", "size"),
        mean_rate=("_rate", "mean"),
    ).reset_index()

    g = g[g["n"] >= int(min_n)]
    g["mean_rate"] = g["mean_rate"].round(2)
    return g.to_dict(orient="records")


# ---------- Vista 4: Benchmark por ubicación ----------

def location_benchmark(df: pd.DataFrame, min_n: int = 5, top_n: int = 15) -> dict:
    """
    Calcula métricas agregadas por ubicación: cantidad, costo promedio, rating promedio.
    """
    if df.empty or "location" not in df.columns:
        return {"locations": []}

    g = df.assign(
        _rate=pd.to_numeric(df.get("rate"), errors="coerce"),
        _cost=pd.to_numeric(df.get("approx_cost_for_two_people"), errors="coerce"),
    ).groupby("location", dropna=True).agg(
        n=("location", "size"),
        mean_cost=("_cost", "mean"),
        mean_rate=("_rate", "mean")
    ).reset_index()

    g = g[g["n"] >= int(min_n)]
    g["mean_cost"] = g["mean_cost"].round(2)
    g["mean_rate"] = g["mean_rate"].round(2)

    # Ordenar por cantidad (por defecto)
    top = g.sort_values("n", ascending=False).head(int(top_n))

    # Costo vs Rating para scatter
    scatter = top[["location", "mean_cost", "mean_rate", "n"]].to_dict(orient="records")

    return {
        "locations": top.to_dict(orient="records"),
        "scatter": scatter
    }

# ---------- Vista 5: Adopción de servicios (Online / Book) ----------
def availability_stats(df: pd.DataFrame, top_n: int = 10, min_n: int = 5) -> dict:
    """
    Analiza adopción de servicios digitales (online_order, book_table).
    Devuelve KPIs, tasas por ubicación y tipo, y una matriz 2x2.

    Notas:
    - NO elimina grupos con n < min_n; los marca con is_small para que el front los resalte.
    - Si top_n <= 0, NO se recorta la lista (se devuelven todos los grupos).
    """
    if df.empty:
        return {"kpis": {}, "by_location": [], "by_resttype": [], "matrix": []}

    d = df.copy()

    # Asegurar columnas
    for col in ["online_order", "book_table"]:
        if col not in d.columns:
            d[col] = False
        else:
            if d[col].dtype != bool:
                d[col] = d[col].apply(lambda x: str(x).strip().lower() in ("yes", "true", "1", "si", "sí"))
            d[col] = d[col].fillna(False).astype(bool)

    d["_rate"] = pd.to_numeric(d.get("rate"), errors="coerce")

    total = len(d)
    online = int(d["online_order"].sum())
    book = int(d["book_table"].sum())
    both = int((d["online_order"] & d["book_table"]).sum())

    kpis = {
        "total_restaurants": total,
        "pct_online": round(100 * online / total, 1) if total else 0.0,
        "pct_book": round(100 * book / total, 1) if total else 0.0,
        "pct_both": round(100 * both / total, 1) if total else 0.0,
        "avg_rating_online": round(d.loc[d["online_order"], "_rate"].mean(), 2) if online else None,
        "avg_rating_no_online": round(d.loc[~d["online_order"], "_rate"].mean(), 2) if (total - online) > 0 else None,
    }

    # --- Ubicación ---
    if "location" in d.columns:
        by_loc = (
            d.groupby("location", dropna=True)
             .agg(n=("location", "size"),
                  pct_online=("online_order", "mean"),
                  pct_book=("book_table", "mean"),
                  mean_rate=("_rate", "mean"))
             .reset_index()
        )
        by_loc["pct_online"] = (100 * by_loc["pct_online"]).round(1)
        by_loc["pct_book"]   = (100 * by_loc["pct_book"]).round(1)
        by_loc["mean_rate"]  = by_loc["mean_rate"].round(2)
        by_loc["is_small"]   = by_loc["n"] < int(min_n)
        by_loc = by_loc.sort_values(["n", "mean_rate"], ascending=[False, False])
        if top_n and int(top_n) > 0:
            by_loc = by_loc.head(int(top_n))
        by_location = by_loc.to_dict(orient="records")
    else:
        by_location = []

    # --- Tipo de restaurante ---
    if "rest_type" in d.columns:
        by_type = (
            d.groupby("rest_type", dropna=True)
             .agg(n=("rest_type", "size"),
                  pct_online=("online_order", "mean"),
                  pct_book=("book_table", "mean"),
                  mean_rate=("_rate", "mean"))
             .reset_index()
        )
        by_type["pct_online"] = (100 * by_type["pct_online"]).round(1)
        by_type["pct_book"]   = (100 * by_type["pct_book"]).round(1)
        by_type["mean_rate"]  = by_type["mean_rate"].round(2)
        by_type["is_small"]   = by_type["n"] < int(min_n)
        by_type = by_type.sort_values(["n", "mean_rate"], ascending=[False, False])
        if top_n and int(top_n) > 0:
            by_type = by_type.head(int(top_n))
        by_resttype = by_type.to_dict(orient="records")
    else:
        by_resttype = []

    # --- Matriz 2x2 ---
    g = (
        d.groupby(["online_order", "book_table"], dropna=True)
         .agg(n=("online_order", "size"), mean_rate=("_rate", "mean"))
         .reset_index()
    )
    g["mean_rate"]    = g["mean_rate"].round(2)
    g["online_order"] = g["online_order"].map({True: "Sí", False: "No"})
    g["book_table"]   = g["book_table"].map({True: "Sí", False: "No"})
    matrix = g.to_dict(orient="records")

    return {"kpis": kpis, "by_location": by_location, "by_resttype": by_resttype, "matrix": matrix}


# ---------- Vista 6: Opinión Pública y Popularidad ----------
def popularity_stats(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Analiza la relación entre popularidad (votes) y calificación (rate).
    Devuelve KPIs, correlación y puntos para gráfico de dispersión.
    """
    if df.empty or "votes" not in df.columns or "rate" not in df.columns:
        return {
            "total": 0,
            "avg_votes": 0,
            "corr_votes_rate": None,
            "top_voted": [],
            "scatter": []
        }

    d = df.copy()
    d["votes"] = pd.to_numeric(d["votes"], errors="coerce")
    d["rate"] = pd.to_numeric(d["rate"], errors="coerce")
    d["approx_cost_for_two_people"] = pd.to_numeric(d.get("approx_cost_for_two_people"), errors="coerce")

    d = d.dropna(subset=["votes", "rate"])
    total = len(d)
    avg_votes = round(d["votes"].mean(), 2)
    corr = round(d["votes"].corr(d["rate"]), 3) if total > 2 else None

    # Top N más votados
    top_voted_df = (
        d.sort_values("votes", ascending=False)
         .head(top_n)
         .loc[:, ["name", "votes", "rate", "rest_type", "location"]]
    )
    top_voted = top_voted_df.to_dict(orient="records")

    # Puntos para scatter
    scatter_df = d.loc[:, ["name", "votes", "rate", "rest_type", "location", "approx_cost_for_two_people"]]
    scatter = scatter_df.to_dict(orient="records")

    return {
        "total": int(total),
        "avg_votes": float(avg_votes),
        "corr_votes_rate": corr,
        "top_voted": top_voted,
        "scatter": scatter
    }
