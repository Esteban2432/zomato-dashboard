from pathlib import Path

# Directorio base del proyecto (carpeta Zomato/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta del dataset que vamos a usar
DATASET_PATH = BASE_DIR / "Dataset" / "zomato_clean.csv"

# Orígenes permitidos para el Frontend (Streamlit)
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
