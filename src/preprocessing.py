"""
preprocessing.py

- defines features + builds sklearn ColumnTransformer
- works for both models:
    - Logistic Regression: scale_numeric=True
    - XGBoost:            scale_numeric=False

expected raw columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal, target
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


# feature def

# numeric / continuous features (including ca)
NUMERIC_FEATURES: List[str] = [
    "age",
    "trestbps",  # resting blood pressure
    "chol",      # serum cholesterol
    "thalach",   # maximum heart rate achieved
    "oldpeak",   # ST depression induced by exercise
    "ca",        # number of major vessels (0–3)
]

BINARY_FEATURES: List[str] = [
    "sex",   # 0 = female, 1 = male (dataset-dependent)
    "fbs",   # fasting blood sugar > 120 mg/dl
    "exang"  # exercise-induced angina
]

# one-hot encoded
CATEGORICAL_FEATURES: List[str] = [
    "cp",       # chest pain type
    "restecg",  # resting ECG results
    "slope",    # slope of peak exercise ST segment
    "thal",     # thalassemia
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# preprocessor builder

def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    columnTransformer:
    - numeric: median impute (+ optional scaling)
    - binary: most_frequent impute
    - categorical: most_frequent impute + one-hot encode (no drop)
    """

    # numeric pipeline
    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    # binary pipeline
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # categorical pipeline
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop=None,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# data loading helper

def load_xy(
    csv_path: str,
    *,
    target: str = TARGET_COLUMN,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    loading X, y from a raw CSV.
    preprocessing applied later inside sklearn pipelines
    """
    df = pd.read_csv(csv_path)

    cols = feature_cols if feature_cols is not None else ALL_FEATURES
    missing = [c for c in cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    X = df[cols].copy()
    y = df[target].copy()

    return X, y
