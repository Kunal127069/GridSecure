import io
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from src.data_utils import load_dataset, get_consumer_by_id, get_demo_dataframe, get_fallback_consumer_record
from src.inference import TheftInferenceEngine
from src.config import MODEL_COMPARISON_PATH

inference_engine = None
cached_df = None


def _ensure_cached_df():
    global cached_df
    if cached_df is None or cached_df.empty:
        try:
            cached_df = load_dataset()
        except Exception as e:
            print("Dataset load fallback error:", e)
            cached_df = get_demo_dataframe()
    return cached_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, cached_df
    try:
        inference_engine = TheftInferenceEngine()
    except Exception as e:
        print("Engine init error:", e)

    _ensure_cached_df()
    yield


app = FastAPI(
    title="GridSecure API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsumerRequest(BaseModel):
    CONS_NO: str = "CONS_1001"
    Locality: str = "Zone-A"
    Consumer_Type: str = "Residential"
    Urban_Rural: str = "Urban"
    Avg_Consumption: float = 18.5
    Median_Consumption: float = 16.0
    Max_Consumption: float = 95.0
    Min_Consumption: float = 0.0
    Std_Consumption: float = 15.2
    Zero_Consumption_Days: int = 22
    Sudden_Drop_Days: int = 5
    Behavioural_Anomaly_Score: float = 0.78
    model_name: Optional[str] = "Random Forest"


@app.get("/")
def home():
    return {"message": "GridSecure API is online", "docs": "/docs"}


@app.get("/health")
def health():
    ready = inference_engine is not None and bool(inference_engine.models)
    available_models = list(inference_engine.models.keys()) if ready else []
    return {
        "status": "healthy" if ready else "degraded",
        "model_ready": ready,
        "available_models": available_models
    }


@app.post("/predict")
def predict(request: ConsumerRequest):
    if inference_engine is None or not inference_engine.models:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")
    return inference_engine.predict_single(request.dict(), model_name=request.model_name)


@app.get("/sample-consumers")
def sample_consumers(limit: int = 10):
    df = _ensure_cached_df()
    if df is not None and not df.empty and "CONS_NO" in df.columns:
        sample_ids = df["CONS_NO"].head(limit).tolist()
    else:
        sample_ids = ["CONS_1001", "CONS_1002", "CONS_1003", "CONS_DEMO_1001", "0387DD8A07E07FDA6271170F86AD9151"]
    return {"consumer_ids": sample_ids}


@app.get("/consumer/{consumer_id}")
def get_consumer(consumer_id: str, model_name: Optional[str] = Query("Random Forest")):
    df = _ensure_cached_df()
    record = get_consumer_by_id(consumer_id, df)
    if record is None:
        record = get_fallback_consumer_record(consumer_id)

    if inference_engine is not None and bool(inference_engine.models):
        pred_result = inference_engine.predict_single(record, model_name=model_name)
    else:
        pred_result = {
            "prediction": int(record.get("Theft_Flag", 0)),
            "probability": float(record.get("Behavioural_Anomaly_Score", 0.5)),
            "model": model_name,
            "drivers": [
                {"feature": "Zero Consumption Days", "contribution": "High Driver", "val": int(record.get("Zero_Consumption_Days", 22))},
                {"feature": "Behavioural Anomaly Score", "contribution": "High Driver", "val": float(record.get("Behavioural_Anomaly_Score", 0.78))}
            ]
        }

    return {
        "consumer": record,
        "analysis": pred_result
    }


@app.get("/analytics")
def analytics():
    df = _ensure_cached_df()
    if df is not None and not df.empty:
        total_consumers = len(df)
        theft_cases = int(df["Theft_Flag"].sum()) if "Theft_Flag" in df.columns else 3615
    else:
        total_consumers = 42372
        theft_cases = 3615

    normal_cases = total_consumers - theft_cases
    baseline_theft_rate = round(theft_cases / total_consumers * 100, 2) if total_consumers > 0 else 8.53

    return {
        "summary": {
            "total_consumers": total_consumers,
            "theft_cases": theft_cases,
            "normal_cases": normal_cases,
            "baseline_theft_rate": baseline_theft_rate,
        }
    }


@app.get("/metrics")
def metrics():
    if MODEL_COMPARISON_PATH.exists():
        df_metrics = pd.read_csv(MODEL_COMPARISON_PATH)
        return {"metrics": df_metrics.to_dict(orient="records")}
    
    return {
        "metrics": [
            {
                "Model": "Random Forest",
                "Accuracy": 0.8838,
                "Precision": 0.3541,
                "Recall": 0.5057,
                "F1 Score": 0.4164,
                "ROC AUC": 0.7670
            },
            {
                "Model": "Decision Tree",
                "Accuracy": 0.8252,
                "Precision": 0.2312,
                "Recall": 0.5287,
                "F1 Score": 0.3217,
                "ROC AUC": 0.6908
            },
            {
                "Model": "Logistic Regression",
                "Accuracy": 0.9147,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1 Score": 0.0,
                "ROC AUC": 0.6721
            }
        ]
    }
