import os
import sys
from pathlib import Path

# Add project root directory to python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.data_utils import load_dataset, separate_features_and_target
from src.model_utils import train_and_evaluate_all, save_pipeline_and_model, generate_evaluation_visualizations
from src.inference import TheftInferenceEngine
from src.config import MODEL_COMPARISON_PATH, EDA_REPORT_PATH


def main():
    print("==================================================")
    print("   GridSecure — Electricity Theft Detection System ")
    print("==================================================\n")

    print("[1/4] Loading Smart Meter Dataset...")
    df = load_dataset()
    print(f"      Loaded {len(df)} consumer records successfully.")

    print("\n[2/4] Generating Summary Statistics & Preprocessing...")
    eda_summary = df.describe(include='all').T
    eda_summary.to_csv(EDA_REPORT_PATH)
    print(f"      Saved EDA Summary Report to '{EDA_REPORT_PATH}'")

    X, y = separate_features_and_target(df)
    print(f"      Features Matrix Shape: {X.shape}, Target Series Shape: {y.shape}")

    print("\n[3/4] Training & Evaluating Models (Logistic Regression, Decision Tree, Random Forest)...")
    comp_df, fitted_models, transformer, best_model, test_eval_data = train_and_evaluate_all(X, y)

    print("\n=================== MODEL PERFORMANCE COMPARISON TABLE ===================")
    print(comp_df.to_string())
    print("==========================================================================")
    
    comp_df.to_csv(MODEL_COMPARISON_PATH)
    save_pipeline_and_model(fitted_models, transformer, "Random Forest")
    generate_evaluation_visualizations(comp_df, test_eval_data, transformer.feature_names, best_model)
    print("      Saved best model to 'models/gridsecure_best_model.pkl' and plots to 'docs/'")

    print("\n[4/4] Running Standalone Inference Engine Test...")
    engine = TheftInferenceEngine(best_model, transformer)
    
    sample_consumer = {
        "CONS_NO": "CONS_DEMO_1001",
        "Locality": "Substation-Zone-4",
        "City": "GridCity",
        "Urban_Rural": "Urban",
        "Consumer_Type": "Residential",
        "Avg_Consumption": 18.5,
        "Zero_Consumption_Days": 22,
        "Sudden_Drop_Days": 5,
        "Behavioural_Anomaly_Score": 0.78,
        "Behaviour_Cluster": 1
    }

    prediction = engine.predict_single(sample_consumer)
    print("\n---------------- Sample Consumer Prediction Output ----------------")
    print(f"Consumer ID           : {prediction['consumer_id']}")
    print(f"Locality / City       : {prediction['locality']}, {prediction['city']}")
    print(f"Predicted Theft Flag  : {prediction['is_theft_predicted']} (1 = Theft, 0 = Normal)")
    print(f"Theft Probability     : {prediction['theft_risk_percentage']}%")
    print(f"Assigned Risk Level   : {prediction['risk_level']}")
    print("Actionable Risk Factors:")
    for factor in prediction['risk_factors']:
        print(f"  • {factor}")
    print("-------------------------------------------------------------------\n")

    print("Execution complete! Project ran successfully standalone.")


if __name__ == "__main__":
    main()
