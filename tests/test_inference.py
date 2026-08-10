import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import TheftInferenceEngine


class DummyModel:
    def predict_proba(self, X):
        probs = np.array([[0.2, 0.8]] * len(X), dtype=float)
        return probs


class DummyTransformer:
    def __init__(self):
        self.feature_names = [
            "Urban_Rural",
            "Consumer_Type",
            "Avg_Consumption",
            "Median_Consumption",
            "Max_Consumption",
            "Min_Consumption",
            "Std_Consumption",
            "Variance_Consumption",
            "Consumption_Range",
            "Consumption_CV",
            "Peak_Average_Ratio",
            "Zero_Consumption_Days",
            "Zero_Consumption_Percentage",
            "Low_Consumption_Days",
            "Low_Consumption_Percentage",
            "First_30_Day_Avg",
            "Last_30_Day_Avg",
            "Long_Term_Change",
            "Long_Term_Change_Percentage",
            "Sudden_Drop_Days",
            "Sudden_Spike_Days",
            "Largest_Drop_Percentage",
            "Largest_Spike_Percentage",
            "Consumption_Trend",
            "Summer_Avg",
            "Monsoon_Avg",
            "Winter_Avg",
            "Seasonal_Variation",
            "Consumption_Stability_Score",
            "Behavioural_Anomaly_Score",
            "Consumption_Consistency",
            "Behaviour_Cluster",
        ]

    def transform(self, X):
        return np.zeros((len(X), len(self.feature_names)), dtype=float)


def test_predict_single_handles_partial_and_invalid_values():
    engine = TheftInferenceEngine(models={"Random Forest": DummyModel()}, transformer=DummyTransformer())

    record = {
        "CONS_NO": "CONS_TEST_001",
        "Locality": "KOLKATA_EAST",
        "City": "Kolkata",
        "State": "West Bengal",
        "Avg_Consumption": "not-a-number",
        "Zero_Consumption_Days": "",
        "Sudden_Drop_Days": "N/A",
        "Behavioural_Anomaly_Score": "NaN",
        "Behaviour_Cluster": None,
    }

    prediction = engine.predict_single(record, model_name="Random Forest")

    assert prediction["consumer_id"] == "CONS_TEST_001"
    assert prediction["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert prediction["is_theft_predicted"] in {0, 1}
    assert prediction["theft_risk_percentage"] >= 0.0
    assert prediction["risk_factors"]
