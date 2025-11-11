from typing import Optional, List
from pydantic import BaseModel

# ---- Base KPIs / Filtros / Histograma ----
class KPIResponse(BaseModel):
    total_restaurants: int
    rating_avg: Optional[float] = None
    cost_for_two_avg: Optional[float] = None
    pct_online_order: Optional[float] = None
    pct_book_table: Optional[float] = None
    unique_cuisines: Optional[int] = None

class FiltersResponse(BaseModel):
    locations: List[str]
    rest_types: List[str]
    cuisines: List[str]

class HistogramResponse(BaseModel):
    bins: list[float]
    counts: list[int]

# ---- Price Stats ----
class LocationCostStat(BaseModel):
    location: str
    mean_cost: Optional[float] = None
    mean_rate: Optional[float] = None
    n: int

class LocationValueStat(BaseModel):
    location: str
    value_score: float
    mean_cost: Optional[float] = None
    mean_rate: Optional[float] = None
    n: int

class PriceSummary(BaseModel):
    count: int
    mean: Optional[float]
    median: Optional[float]
    min: Optional[float]
    max: Optional[float]
    corr_cost_rating: Optional[float]
    cost_col: Optional[str]

class ScatterPoint(BaseModel):
    cost: float
    rating: float

class PriceStatsResponse(BaseModel):
    available: bool
    reason: Optional[str] = None
    summary: PriceSummary
    scatter: list[ScatterPoint]
    expensive_locations: list[LocationCostStat]
    best_value_locations: list[LocationValueStat]

# ---- Top lists ----
class TopCuisineItem(BaseModel):
    cuisine: str
    count: int
    mean_rate: Optional[float] = None

class TopRestTypeItem(BaseModel):
    rest_type: str
    count: int
    mean_rate: Optional[float] = None


class CuisineAggItem(BaseModel):
    cuisine: str
    count: int
    mean_rate: Optional[float] = None
    mean_cost: Optional[float] = None

class CuisineStatsResponse(BaseModel):
    by_count: list[CuisineAggItem]
    by_rating: list[CuisineAggItem]
    by_cost: list[CuisineAggItem]

class RestTypeAggItem(BaseModel):
    rest_type: str
    n: int
    mean_rate: Optional[float] = None
    mean_cost: Optional[float] = None

class MatrixItem(BaseModel):
    cuisine: str
    rest_type: str
    n: int
    mean_rate: Optional[float] = None


class LocationItem(BaseModel):
    location: str
    n: int
    mean_cost: Optional[float] = None
    mean_rate: Optional[float] = None

class LocationBenchmarkResponse(BaseModel):
    locations: list[LocationItem]
    scatter: list[dict]


class AvailabilityKPI(BaseModel):
    total_restaurants: int
    pct_online: float
    pct_book: float
    pct_both: float
    avg_rating_online: Optional[float] = None
    avg_rating_no_online: Optional[float] = None

class AvailabilityAggItem(BaseModel):
    n: int
    pct_online: float
    pct_book: float
    mean_rate: Optional[float] = None
    location: Optional[str] = None
    rest_type: Optional[str] = None
    is_small: Optional[bool] = None 

class AvailabilityMatrixItem(BaseModel):
    online_order: str
    book_table: str
    n: int
    mean_rate: Optional[float] = None

class AvailabilityStatsResponse(BaseModel):
    kpis: AvailabilityKPI
    by_location: list[AvailabilityAggItem]
    by_resttype: list[AvailabilityAggItem]
    matrix: list[AvailabilityMatrixItem]


# ---- Vista 6: Opinión Pública y Popularidad ----
class ScatterPointPopularity(BaseModel):
    name: Optional[str] = None
    votes: Optional[float] = None
    rate: Optional[float] = None
    rest_type: Optional[str] = None
    location: Optional[str] = None
    approx_cost_for_two_people: Optional[float] = None

class TopVotedItem(BaseModel):
    name: str
    votes: float
    rate: float
    rest_type: Optional[str] = None
    location: Optional[str] = None

class PopularityStatsResponse(BaseModel):
    total: int
    avg_votes: float
    corr_votes_rate: Optional[float] = None
    top_voted: list[TopVotedItem]
    scatter: list[ScatterPointPopularity]
