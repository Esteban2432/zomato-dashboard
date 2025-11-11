from typing import Optional, List
import ast
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .settings import ALLOWED_ORIGINS
from .data_loader import get_df
from .analytics import (
    get_kpis, get_filters, ratings_histogram, get_price_stats,
    top_cuisines, top_rest_types,
    cuisine_stats, resttype_stats, cuisine_resttype_matrix,
    location_benchmark, availability_stats, popularity_stats
)

from .schemas import (
    KPIResponse, FiltersResponse, HistogramResponse, PriceStatsResponse,
    TopCuisineItem, TopRestTypeItem,
    CuisineStatsResponse, RestTypeAggItem, MatrixItem,
    LocationBenchmarkResponse, AvailabilityStatsResponse,
    PopularityStatsResponse
)


app = FastAPI(title="Zomato API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Helpers de parsing/filtrado ----------
def _as_list(values):
    """
    Acepta:
      - ['a','b'] (normal)
      - ["['a','b']"] (stringificado) -> lo parsea
    """
    if not values:
        return values
    if len(values) == 1 and isinstance(values[0], str) and values[0].startswith("["):
        try:
            parsed = ast.literal_eval(values[0])
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return values

def _apply_filters(d: pd.DataFrame,
                   locations, rest_types, cuisines,
                   online_order, book_table,
                   rate_min, rate_max, cost_min, cost_max):
    locations = _as_list(locations)
    rest_types = _as_list(rest_types)
    cuisines  = _as_list(cuisines)

    if locations:
        d = d[d["location"].isin(locations)]
    if rest_types:
        d = d[d["rest_type"].isin(rest_types)]
    if cuisines:
        # un restaurante puede tener varias cocinas separadas por coma
        d = d[d["cuisines"].fillna("").apply(
            lambda x: any(
                c.strip().lower() in [p.strip().lower() for p in str(x).split(",")]
                for c in cuisines
            )
        )]

    if online_order is not None:
        d = d[d["online_order"] == online_order]
    if book_table is not None:
        d = d[d["book_table"] == book_table]

    if rate_min is not None:
        d = d[pd.to_numeric(d["rate"], errors="coerce") >= float(rate_min)]
    if rate_max is not None and float(rate_max) > 0:
        d = d[pd.to_numeric(d["rate"], errors="coerce") <= float(rate_max)]

    cost_col = "approx_cost_for_two_people" if "approx_cost_for_two_people" in d.columns else None
    if cost_col:
        if cost_min is not None and float(cost_min) > 0:
            d = d[pd.to_numeric(d[cost_col], errors="coerce") >= float(cost_min)]
        if cost_max is not None and float(cost_max) > 0:
            d = d[pd.to_numeric(d[cost_col], errors="coerce") <= float(cost_max)]

    return d

# --------- Rutas ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def meta():
    df = get_df()
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "columns": list(df.columns)}

@app.get("/filters", response_model=FiltersResponse)
def filters():
    return get_filters(get_df())

@app.get("/kpis", response_model=KPIResponse)
def kpis(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return get_kpis(df)

@app.get("/ratings_distribution", response_model=HistogramResponse)
def ratings_distribution(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    bins: int = 20,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return ratings_histogram(df, bins=bins)

@app.get("/price_stats", response_model=PriceStatsResponse)
def price_stats(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    pclip: Optional[float] = Query(0.0, ge=0.0, le=0.999),   # nuevo
    scatter_max: Optional[int] = Query(1200, ge=100, le=10000),  # nuevo
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return get_price_stats(df, top_n=5, pclip=pclip or 0.0, scatter_max=scatter_max or 1200)


@app.get("/top_cuisines", response_model=list[TopCuisineItem])
def top_cuisines_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    top_n: int = 10,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return top_cuisines(df, top_n=top_n)

@app.get("/top_rest_types", response_model=list[TopRestTypeItem])
def top_rest_types_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    top_n: int = 10,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return top_rest_types(df, top_n=top_n)



@app.get("/cuisine_stats", response_model=CuisineStatsResponse)
def cuisine_stats_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    top_n: int = 10,
    min_n: int = 5,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return cuisine_stats(df, top_n=top_n, min_n=min_n)


@app.get("/resttype_stats", response_model=list[RestTypeAggItem])
def resttype_stats_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    top_n: int = 10,
    min_n: int = 5,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return resttype_stats(df, top_n=top_n, min_n=min_n)


@app.get("/cuisine_resttype_matrix", response_model=list[MatrixItem])
def cuisine_resttype_matrix_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    min_n: int = 3,
    top_cuisines: int = 15,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return cuisine_resttype_matrix(df, min_n=min_n, top_cuisines=top_cuisines)


@app.get("/location_benchmark", response_model=LocationBenchmarkResponse)
def location_benchmark_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    min_n: int = 5,
    top_n: int = 15,
):
    df = _apply_filters(get_df(), locations, rest_types, cuisines,
                        online_order, book_table, rate_min, rate_max, cost_min, cost_max)
    return location_benchmark(df, min_n=min_n, top_n=top_n)



@app.get("/availability_stats", response_model=AvailabilityStatsResponse)
def availability_stats_ep(
    locations: Optional[List[str]] = Query(None),
    rest_types: Optional[List[str]] = Query(None),
    cuisines: Optional[List[str]] = Query(None),
    online_order: Optional[bool] = None,
    book_table: Optional[bool] = None,
    rate_min: Optional[float] = None,
    rate_max: Optional[float] = None,
    cost_min: Optional[float] = None,
    cost_max: Optional[float] = None,
    top_n: int = 10,
    min_n: int = 5,
):
    df = _apply_filters(
        get_df(), locations, rest_types, cuisines,
        online_order, book_table, rate_min, rate_max, cost_min, cost_max
    )
    return availability_stats(df, top_n=top_n, min_n=min_n)



@app.get("/popularity_stats", response_model=PopularityStatsResponse)
def popularity_stats_ep():
    df = get_df()
    return popularity_stats(df)