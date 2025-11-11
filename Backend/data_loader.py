import pandas as pd
from .settings import DATASET_PATH

_df_cache = None

def _to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("yes", "true", "1", "si", "sí"):
        return True
    if s in ("no", "false", "0", "none", "nan"):
        return False
    return None

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres y tipos desde zomato_clean.csv a los usados por la API.
    Esperado en el CSV: name, address, online_order, book_table, location,
    restaurant_type, cuisines, category, city, votes, rating, cost_for_two
    """
    rename_map = {
        "restaurant_type": "rest_type",
        "rating": "rate",
        "cost_for_two": "approx_cost_for_two_people",
    }
    df = df.rename(columns=rename_map)

    # Booleanos
    if "online_order" in df.columns:
        df["online_order"] = df["online_order"].map(_to_bool)
    if "book_table" in df.columns:
        df["book_table"] = df["book_table"].map(_to_bool)

    # Numéricos
    for c in ("rate", "votes", "approx_cost_for_two_people"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categóricos/strings limpios
    for c in ("name", "location", "rest_type", "cuisines", "category", "city", "address"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return _standardize_columns(df)

def get_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = load_df()
    # devolvemos copia para no mutar la caché
    return _df_cache.copy()
