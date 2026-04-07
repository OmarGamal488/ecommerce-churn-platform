"""Shared data loading for dashboard tabs."""

import os
import pandas as pd
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def load_raw():
    return pd.read_excel(os.path.join(DATA_DIR, "raw", "E Commerce Dataset.xlsx"), sheet_name="E Comm")


def load_cleaned():
    return pd.read_csv(os.path.join(DATA_DIR, "processed", "cleaned.csv"))


def load_engineered():
    return pd.read_csv(os.path.join(DATA_DIR, "processed", "engineered.csv"))


def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))


def load_feature_names():
    return joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))


def load_shap_explainer():
    return joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
