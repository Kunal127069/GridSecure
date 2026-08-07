import io
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_utils import load_dataset, get_consumer_by_id
from src.inference import TheftInferenceEngine
from src.config import MODEL_COMPARISON_PATH

inference_engine = None
cached_df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, cached_df
    try:
        inference_engine = TheftInferenceEngine()
    except Exception as e:
        print("Engine init error:", e)

    try:
        cached_df = load_dataset()
    except Exception as e:
        print("Dataset load error:", e)
        
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
    City: str = "Metro City"
    Urban_Rural: str = "Urban"
    Consumer_Type: str = "Residential"
    Avg_Consumption: float = 25.5
    Median_Consumption: float = 22.0
    Max_Consumption: float = 120.0
    Min_Consumption: float = 0.0
    Std_Consumption: float = 18.2
    Zero_Consumption_Days: int = 15
    Zero_Consumption_Percentage: float = 12.5
    Sudden_Drop_Days: int = 4
    Largest_Drop_Percentage: float = 55.0
    First_30_Day_Avg: float = 35.0
    Last_30_Day_Avg: float = 12.0
    Long_Term_Change_Percentage: float = -65.7
    Behavioural_Anomaly_Score: float = 0.72
    Behaviour_Cluster: int = 1


@app.get("/")
def home():
    return {"message": "GridSecure API is online", "docs": "/docs"}


@app.get("/health")
def health():
    ready = inference_engine is not None and inference_engine.model is not None
    return {"status": "healthy" if ready else "degraded", "model_ready": ready}


@app.post("/predict")
def predict(request: ConsumerRequest):
    if inference_engine is None or inference_engine.model is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")
    return inference_engine.predict_single(request.dict())


@app.get("/consumer/{consumer_id}")
def get_consumer(consumer_id: str):
    if cached_df is None:
        raise HTTPException(status_code=503, detail="Dataset is not loaded.")
    record = get_consumer_by_id(consumer_id, cached_df)
    if record is None:
        raise HTTPException(status_code=404, detail="Consumer not found.")
    
    pred = inference_engine.predict_single(record) if inference_engine and inference_engine.model else None
    return {"consumer_profile": record, "prediction": pred}


@app.get("/analytics")
def analytics():
    if cached_df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    df = cached_df
    total = len(df)
    theft = int(df["Theft_Flag"].sum()) if "Theft_Flag" in df.columns else 0
    return {
        "total_consumers": total,
        "theft_cases": theft,
        "normal_cases": total - theft,
        "theft_rate_percentage": round((theft / total) * 100, 2) if total > 0 else 0.0
    }


@app.get("/metrics")
def metrics():
    if MODEL_COMPARISON_PATH.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)
        return comp_df.to_dict(orient="index")
    return {"info": "Metrics table not found."}
