from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

DATASET_PATH = DATA_DIR / "data.csv"
REMOTE_DATASET_URL = "https://github.com/Kunal127069/GridSecure/releases/download/v1.0.0/electricity_theft_dataset_with_clusters_V2.csv"

BEST_MODEL_PATH = MODELS_DIR / "gridsecure_best_model.pkl"
PIPELINE_PATH = MODELS_DIR / "gridsecure_pipeline.joblib"
MODEL_COMPARISON_PATH = DOCS_DIR / "Model_Comparison_Table.csv"
EDA_REPORT_PATH = DOCS_DIR / "EDA_Report.csv"

METADATA_COLS = ["CONS_NO", "Locality", "City", "State"]
CATEGORICAL_COLS = ["Urban_Rural", "Consumer_Type"]
TARGET_COL = "Theft_Flag"

RANDOM_STATE = 42
TEST_SIZE = 0.20
