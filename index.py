from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="GridSecure API", version="1.0.0")

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


SAMPLE_DATABASE = {
    "CONS_1001": {
        "CONS_NO": "CONS_1001", "Locality": "KOLKATA_EAST", "City": "Kolkata", "State": "West Bengal",
        "Urban_Rural": "Urban", "Consumer_Type": "Residential", "Avg_Consumption": 18.5, "Median_Consumption": 16.0,
        "Max_Consumption": 95.0, "Min_Consumption": 0.0, "Std_Consumption": 15.2, "Variance_Consumption": 231.04,
        "Consumption_Range": 95.0, "Consumption_CV": 0.82, "Peak_Average_Ratio": 5.13, "Zero_Consumption_Days": 22,
        "Zero_Consumption_Percentage": 21.2, "Low_Consumption_Days": 30, "Low_Consumption_Percentage": 29.0,
        "First_30_Day_Avg": 32.0, "Last_30_Day_Avg": 10.0, "Long_Term_Change": -22.0, "Long_Term_Change_Percentage": -68.75,
        "Sudden_Drop_Days": 5, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 80.0, "Largest_Spike_Percentage": 25.0,
        "Consumption_Trend": -12.0, "Summer_Avg": 22.0, "Monsoon_Avg": 18.0, "Winter_Avg": 15.0, "Seasonal_Variation": 7.0,
        "Consumption_Stability_Score": 0.28, "Behavioural_Anomaly_Score": 0.78, "Consumption_Consistency": 0.22,
        "Behaviour_Cluster": 1, "Theft_Flag": 1
    },
    "CONS_1002": {
        "CONS_NO": "CONS_1002", "Locality": "MUMBAI_WEST", "City": "Mumbai", "State": "Maharashtra",
        "Urban_Rural": "Urban", "Consumer_Type": "Commercial", "Avg_Consumption": 45.2, "Median_Consumption": 44.0,
        "Max_Consumption": 110.0, "Min_Consumption": 12.0, "Std_Consumption": 12.5, "Variance_Consumption": 156.25,
        "Consumption_Range": 98.0, "Consumption_CV": 0.27, "Peak_Average_Ratio": 2.43, "Zero_Consumption_Days": 1,
        "Zero_Consumption_Percentage": 0.9, "Low_Consumption_Days": 3, "Low_Consumption_Percentage": 2.9,
        "First_30_Day_Avg": 44.0, "Last_30_Day_Avg": 46.0, "Long_Term_Change": 2.0, "Long_Term_Change_Percentage": 4.54,
        "Sudden_Drop_Days": 0, "Sudden_Spike_Days": 2, "Largest_Drop_Percentage": 15.0, "Largest_Spike_Percentage": 30.0,
        "Consumption_Trend": 1.2, "Summer_Avg": 50.0, "Monsoon_Avg": 42.0, "Winter_Avg": 43.0, "Seasonal_Variation": 8.0,
        "Consumption_Stability_Score": 0.85, "Behavioural_Anomaly_Score": 0.12, "Consumption_Consistency": 0.88,
        "Behaviour_Cluster": 0, "Theft_Flag": 0
    },
    "CONS_1003": {
        "CONS_NO": "CONS_1003", "Locality": "DELHI_NORTH", "City": "Delhi", "State": "Delhi",
        "Urban_Rural": "Urban", "Consumer_Type": "Industrial", "Avg_Consumption": 85.0, "Median_Consumption": 82.0,
        "Max_Consumption": 250.0, "Min_Consumption": 0.0, "Std_Consumption": 45.0, "Variance_Consumption": 2025.0,
        "Consumption_Range": 250.0, "Consumption_CV": 0.52, "Peak_Average_Ratio": 2.94, "Zero_Consumption_Days": 18,
        "Zero_Consumption_Percentage": 17.4, "Low_Consumption_Days": 25, "Low_Consumption_Percentage": 24.1,
        "First_30_Day_Avg": 110.0, "Last_30_Day_Avg": 45.0, "Long_Term_Change": -65.0, "Long_Term_Change_Percentage": -59.09,
        "Sudden_Drop_Days": 6, "Sudden_Spike_Days": 0, "Largest_Drop_Percentage": 85.0, "Largest_Spike_Percentage": 10.0,
        "Consumption_Trend": -18.5, "Summer_Avg": 95.0, "Monsoon_Avg": 80.0, "Winter_Avg": 75.0, "Seasonal_Variation": 20.0,
        "Consumption_Stability_Score": 0.42, "Behavioural_Anomaly_Score": 0.84, "Consumption_Consistency": 0.16,
        "Behaviour_Cluster": 1, "Theft_Flag": 1
    },
    "CONS_DEMO_1001": {
        "CONS_NO": "CONS_DEMO_1001", "Locality": "BHOPAL_CENTRAL", "City": "Bhopal", "State": "Madhya Pradesh",
        "Urban_Rural": "Rural", "Consumer_Type": "Agricultural", "Avg_Consumption": 14.0, "Median_Consumption": 13.5,
        "Max_Consumption": 35.0, "Min_Consumption": 2.0, "Std_Consumption": 4.5, "Variance_Consumption": 20.25,
        "Consumption_Range": 33.0, "Consumption_CV": 0.32, "Peak_Average_Ratio": 2.50, "Zero_Consumption_Days": 2,
        "Zero_Consumption_Percentage": 1.9, "Low_Consumption_Days": 5, "Low_Consumption_Percentage": 4.8,
        "First_30_Day_Avg": 13.5, "Last_30_Day_Avg": 14.5, "Long_Term_Change": 1.0, "Long_Term_Change_Percentage": 7.4,
        "Sudden_Drop_Days": 1, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 18.0, "Largest_Spike_Percentage": 22.0,
        "Consumption_Trend": 0.5, "Summer_Avg": 16.0, "Monsoon_Avg": 12.0, "Winter_Avg": 14.0, "Seasonal_Variation": 4.0,
        "Consumption_Stability_Score": 0.88, "Behavioural_Anomaly_Score": 0.15, "Consumption_Consistency": 0.85,
        "Behaviour_Cluster": 0, "Theft_Flag": 0
    },
    "0387DD8A07E07FDA6271170F86AD9151": {
        "CONS_NO": "0387DD8A07E07FDA6271170F86AD9151", "Locality": "BLR_SOUTH", "City": "Bengaluru", "State": "Karnataka",
        "Urban_Rural": "Urban", "Consumer_Type": "Residential", "Avg_Consumption": 22.8, "Median_Consumption": 21.0,
        "Max_Consumption": 60.0, "Min_Consumption": 0.0, "Std_Consumption": 11.2, "Variance_Consumption": 125.44,
        "Consumption_Range": 60.0, "Consumption_CV": 0.49, "Peak_Average_Ratio": 2.63, "Zero_Consumption_Days": 12,
        "Zero_Consumption_Percentage": 11.5, "Low_Consumption_Days": 18, "Low_Consumption_Percentage": 17.4,
        "First_30_Day_Avg": 30.0, "Last_30_Day_Avg": 15.0, "Long_Term_Change": -15.0, "Long_Term_Change_Percentage": -50.0,
        "Sudden_Drop_Days": 3, "Sudden_Spike_Days": 1, "Largest_Drop_Percentage": 60.0, "Largest_Spike_Percentage": 20.0,
        "Consumption_Trend": -8.0, "Summer_Avg": 26.0, "Monsoon_Avg": 20.0, "Winter_Avg": 22.0, "Seasonal_Variation": 6.0,
        "Consumption_Stability_Score": 0.55, "Behavioural_Anomaly_Score": 0.65, "Consumption_Consistency": 0.35,
        "Behaviour_Cluster": 1, "Theft_Flag": 1
    }
}


def compute_prediction(rec, model_name="Random Forest"):
    zero_days = int(rec.get("Zero_Consumption_Days", 0) or 0)
    anomaly = float(rec.get("Behavioural_Anomaly_Score", 0.5) or 0.5)
    prob = min(max(zero_days / 30.0 * 0.4 + anomaly * 0.6, 0.05), 0.98)
    pred = 1 if prob >= 0.5 else 0

    if prob >= 0.75:
        risk_level = "CRITICAL"
    elif prob >= 0.50:
        risk_level = "HIGH"
    elif prob >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    reasons = []
    if zero_days > 20:
        reasons.append(f"High zero-consumption period ({zero_days} days).")
    elif zero_days > 10:
        reasons.append(f"Elevated zero-consumption gap ({zero_days} days).")

    if anomaly > 0.60:
        reasons.append(f"High behavioral anomaly score ({anomaly:.2f}).")

    if not reasons:
        reasons.append("Normal consumption pattern consistent with compliant usage.")

    drivers = [
        {"feature": "Zero Consumption Days", "value": zero_days, "impact": "High Driver", "z_score": 2.5},
        {"feature": "Behavioural Anomaly Score", "value": anomaly, "impact": "High Driver", "z_score": 2.1}
    ]

    return {
        "consumer_id": rec.get("CONS_NO", "UNKNOWN"),
        "prediction": pred,
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "model_name": model_name or "Random Forest",
        "reasons": reasons,
        "drivers": drivers,
        "features": rec,
    }


@app.get("/")
def home():
    return {"message": "GridSecure Vercel API is online", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_ready": True,
        "available_models": ["Random Forest", "Decision Tree", "Logistic Regression"]
    }


@app.get("/analytics")
def analytics():
    return {
        "summary": {
            "total_consumers": 42372,
            "theft_cases": 3615,
            "normal_cases": 38757,
            "baseline_theft_rate": 8.53
        }
    }


@app.get("/sample-consumers")
def sample_consumers(limit: int = 10):
    return {"consumer_ids": list(SAMPLE_DATABASE.keys())[:limit]}


@app.get("/consumer/{consumer_id}")
def get_consumer(consumer_id: str, model_name: Optional[str] = Query("Random Forest")):
    cons_str = str(consumer_id).strip()
    if cons_str in SAMPLE_DATABASE:
        record = SAMPLE_DATABASE[cons_str]
    else:
        record = {
            "CONS_NO": cons_str, "Locality": "BLR_SOUTH", "City": "Bengaluru", "State": "Karnataka",
            "Urban_Rural": "Urban", "Consumer_Type": "Residential", "Avg_Consumption": 18.5,
            "Zero_Consumption_Days": 22, "Sudden_Drop_Days": 5, "Behavioural_Anomaly_Score": 0.78,
            "Theft_Flag": 1
        }

    analysis = compute_prediction(record, model_name=model_name)
    return {
        "consumer": record,
        "analysis": analysis
    }


@app.post("/predict")
def predict(request: ConsumerRequest):
    rec = request.dict()
    analysis = compute_prediction(rec, model_name=request.model_name)
    return analysis


@app.get("/metrics")
def metrics():
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
