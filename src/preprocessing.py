"""
preprocessing.py

- defines features + builds sklearn ColumnTransformer
- works for both models:
    - Logistic Regression: scale_numeric=True
    - XGBoost:            scale_numeric=False

expected raw columns:
Sex, GeneralHealth, PhysicalActivities, SleepHours, HadHeartAttack,
SmokerStatus, RaceEthnicityCategory, AgeCategory, HeightInMeters,
WeightInKilograms, AlcoholDrinkers
"""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


# feature definitions

# numeric / continuous features
NUMERIC_FEATURES: List[str] = [
    "SleepHours",         # hours of sleep per night
    "HeightInMeters",     # height in meters
    "WeightInKilograms",  # weight in kilograms
]

# binary features (will be converted to 0/1)
BINARY_FEATURES: List[str] = [
    "Sex",                  # Male/Female
    "PhysicalActivities",   # Yes/No
    "AlcoholDrinkers"       # Yes/No
]

# categorical features (one-hot encoded)
CATEGORICAL_FEATURES: List[str] = [
    "GeneralHealth",          # Excellent, Very good, Good, Fair, Poor
    "SmokerStatus",           # Never smoked, Former smoker, Current smoker, etc.
    "RaceEthnicityCategory",  # race/ethnicity categories
    "AgeCategory",            # age ranges
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# preprocessor builder

def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    columnTransformer:
    - numeric: median impute (+ optional scaling)
    - binary: most_frequent impute + convert to 0/1
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

    # binary pipeline (impute + will handle string to binary conversion)
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary",  # for binary features, drop one category
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # categorical pipeline
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop=None,  # keeping all categories for multi-class features
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
    
    # drop duplicates
    original_len = len(df)
    df = df.drop_duplicates()
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} duplicate rows ({(original_len - len(df))/original_len*100:.1f}%)")

    cols = feature_cols if feature_cols is not None else ALL_FEATURES
    missing = [c for c in cols + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # drop rows where target is missing
    before_drop = len(df)
    df = df.dropna(subset=[target])
    if len(df) < before_drop:
        print(f"Dropped {before_drop - len(df)} rows with missing target values")

    X = df[cols].copy()
    y = df[target].copy()
    
    # converting to 1/0
    if y.dtype == 'object':
        # handling yes/no
        y_mapped = y.str.strip().str.lower().map({'yes': 1, 'no': 0})
        if y_mapped.isnull().any():
            invalid_values = y[y_mapped.isnull()].unique()
            raise ValueError(f"Target column '{target}' contains unexpected values: {invalid_values}")
        y = y_mapped.astype(int)
        print(f"Converted target '{target}' from Yes/No to 1/0")

    return X, y
