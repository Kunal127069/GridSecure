import pandas as pd
import numpy as np
from src.data_utils import clean_data
from src.model_utils import load_pipeline_and_model


class TheftInferenceEngine:
    def __init__(self, model=None, transformer=None):
        if model is not None and transformer is not None:
            self.model = model
            self.transformer = transformer
        else:
            try:
                self.model, self.transformer = load_pipeline_and_model()
            except Exception as e:
                print("Engine uninitialized:", e)
                self.model = None
                self.transformer = None

    def get_risk_level(self, prob):
        if prob >= 0.75:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.25:
            return "MEDIUM"
        return "LOW"

    def get_reasons(self, data, prob):
        reasons = []
        zero_days = data.get("Zero_Consumption_Days", 0)
        if zero_days > 20:
            reasons.append(f"High zero-consumption period ({zero_days} days).")

        sudden_drops = data.get("Sudden_Drop_Days", 0)
        if sudden_drops > 3:
            reasons.append(f"Frequent sudden consumption drop events ({sudden_drops} events).")

        anomaly_score = data.get("Behavioural_Anomaly_Score", 0)
        if anomaly_score > 0.60:
            reasons.append(f"High behavioral anomaly score ({anomaly_score:.2f}).")

        cluster = data.get("Behaviour_Cluster", 0)
        if cluster == 1:
            reasons.append("Assigned to high-risk behavioral cluster (Cluster 1).")

        if not reasons:
            if prob >= 0.50:
                reasons.append("Consumption pattern matches historical theft signatures.")
            else:
                reasons.append("Normal usage pattern consistent with legitimate profile.")

        return reasons

    def predict_single(self, record):
        if self.model is None or self.transformer is None:
            raise RuntimeError("Engine is not ready.")

        df_single = pd.DataFrame([record])
        df_clean = clean_data(df_single)
        meta_cols = ["CONS_NO", "Locality", "City", "State", "Theft_Flag"]
        feature_df = df_clean.drop(columns=[c for c in meta_cols if c in df_clean.columns])

        X_scaled = self.transformer.transform(feature_df)
        prob = float(self.model.predict_proba(X_scaled)[0, 1]) if hasattr(self.model, "predict_proba") else float(self.model.predict(X_scaled)[0])

        return {
            "consumer_id": str(record.get("CONS_NO", "UNKNOWN")),
            "locality": str(record.get("Locality", "Unknown")),
            "city": str(record.get("City", "Unknown")),
            "is_theft_predicted": int(prob >= 0.50),
            "theft_probability": round(prob, 4),
            "theft_risk_percentage": round(prob * 100, 2),
            "risk_level": self.get_risk_level(prob),
            "risk_factors": self.get_reasons(record, prob)
        }

    def predict_batch(self, df):
        df_clean = clean_data(df)
        meta_cols = ["CONS_NO", "Locality", "City", "State", "Theft_Flag"]
        feature_df = df_clean.drop(columns=[c for c in meta_cols if c in df_clean.columns])

        X_scaled = self.transformer.transform(feature_df)
        probs = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, "predict_proba") else self.model.predict(X_scaled).astype(float)

        results_df = df.copy()
        results_df["Theft_Probability"] = np.round(probs, 4)
        results_df["Theft_Risk_Percentage"] = np.round(probs * 100, 2)
        results_df["Predicted_Theft_Flag"] = (probs >= 0.50).astype(int)
        results_df["Risk_Level"] = [self.get_risk_level(p) for p in probs]

        return results_df
