import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.config import (
    BEST_MODEL_PATH, PIPELINE_PATH, MODEL_COMPARISON_PATH,
    DOCS_DIR, RANDOM_STATE, TEST_SIZE
)
from src.features import FeaturePipelineTransformer


def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_split=10, random_state=RANDOM_STATE, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    }


def train_and_evaluate_all(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    transformer = FeaturePipelineTransformer()
    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled = transformer.transform(X_test)

    models = get_models()
    results = {}
    fitted_models = {}
    test_eval_data = {"y_test": y_test, "predictions": {}, "probabilities": {}}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        }
        fitted_models[name] = model
        test_eval_data["predictions"][name] = y_pred
        test_eval_data["probabilities"][name] = y_prob

    comparison_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    best_model_name = comparison_df["F1-Score"].idxmax()
    best_model = fitted_models[best_model_name]

    return comparison_df, fitted_models, transformer, best_model, test_eval_data


def save_pipeline_and_model(models, transformer, best_model_name):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(models, dict):
        fitted_dict = models
        best_model = models.get(best_model_name, list(models.values())[0])
    else:
        fitted_dict = {best_model_name: models}
        best_model = models

    bundle = {
        "models": fitted_dict,
        "best_model_name": best_model_name,
        "transformer": transformer
    }
    joblib.dump(bundle, PIPELINE_PATH)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    return PIPELINE_PATH, BEST_MODEL_PATH


def load_pipeline_and_model():
    if PIPELINE_PATH.exists():
        bundle = joblib.load(PIPELINE_PATH)
        if "models" in bundle:
            return bundle["models"], bundle["transformer"], bundle.get("best_model_name", "Random Forest")
        elif "model" in bundle:
            name = bundle.get("model_name", "Random Forest")
            return {name: bundle["model"]}, bundle["transformer"], name
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return {"Random Forest": model}, FeaturePipelineTransformer(), "Random Forest"
    raise FileNotFoundError("Model file not found.")


def generate_evaluation_visualizations(results_df, test_eval_data, feature_names, best_model):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    y_test = test_eval_data["y_test"]

    fig, axes = plt.subplots(1, len(test_eval_data["predictions"]), figsize=(15, 4))
    for idx, (name, y_pred) in enumerate(test_eval_data["predictions"].items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "confusion_matrices.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for name, y_prob in test_eval_data["probabilities"].items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(DOCS_DIR / "roc_curves.png", dpi=300)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette="viridis")
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "feature_importance.png", dpi=300)
        plt.close()
