import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeaturePipelineTransformer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.column_medians_ = {}
        self.is_fitted = False

    def fit(self, X):
        X_encoded = X.copy()
        cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns

        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_encoded[col].astype(str).tolist() + ["Unknown"])
            self.label_encoders[col] = le

            mapping = {val: idx for idx, val in enumerate(le.classes_)}
            unknown_val = mapping.get("Unknown", 0)
            X_encoded[col] = (
                X_encoded[col]
                .astype(str)
                .map(lambda s: mapping.get(s, unknown_val))
                .fillna(unknown_val)
                .astype(int)
            )

        X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
        self.column_medians_ = X_encoded.median().to_dict()
        X_encoded = X_encoded.fillna(self.column_medians_)
        self.feature_names = X_encoded.columns.tolist()
        self.scaler.fit(X_encoded)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("FeaturePipelineTransformer must be fitted before transform.")

        X_encoded = X.copy()
        for col in self.feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = self.column_medians_.get(col, 0.0)

        X_encoded = X_encoded[self.feature_names]

        for col, le in self.label_encoders.items():
            if col in X_encoded.columns:
                mapping = {val: idx for idx, val in enumerate(le.classes_)}
                unknown_val = mapping.get("Unknown", 0)
                X_encoded[col] = (
                    X_encoded[col]
                    .astype(str)
                    .map(lambda s: mapping.get(s, unknown_val))
                    .fillna(unknown_val)
                    .astype(int)
                )

        X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
        X_encoded = X_encoded.fillna(pd.Series(self.column_medians_))
        return self.scaler.transform(X_encoded)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
