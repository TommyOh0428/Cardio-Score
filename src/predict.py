import joblib
import pandas as pd
from src.preprocessing import make_preprocessor
from src.config import FEATURE_COLUMNS

model_paths = {
    'lr_model': 'models/logistic_regression.joblib',
    'xgb_model': 'models/xgboost_model.joblib'
}

# Load preprocessors (created once, reused)
_lr_preprocessor = None
_xgb_preprocessor = None

def load_model(model_name: str):
    """
    Load a trained model from disk.

    Parameters:
    - model_name: str, name of the model to load ('logistic_regression' or 'xgboost')

    Returns:
    - Loaded model object
    """
    if model_name not in model_paths:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models: {list(model_paths.keys())}")

    model_path = model_paths[model_name]
    model = joblib.load(model_path)
    return model

def _get_preprocessors():
    """Load and fit preprocessors on training data once."""
    global _lr_preprocessor, _xgb_preprocessor
    
    if _lr_preprocessor is None or _xgb_preprocessor is None:
        from src.config import RAW_DATA_PATH, TARGET_COLUMN
        from src.preprocessing import load_xy
        
        # Load training data to fit preprocessors
        data_X, data_y = load_xy(RAW_DATA_PATH, target=TARGET_COLUMN, feature_cols=FEATURE_COLUMNS)
        
        _lr_preprocessor = make_preprocessor(True)  # with scaling
        _xgb_preprocessor = make_preprocessor(False)  # without scaling
        
        _lr_preprocessor.fit(data_X)
        _xgb_preprocessor.fit(data_X)
    
    return _lr_preprocessor, _xgb_preprocessor

def predict_with_two_model(input_data):
    """
    Make predictions using both LR and XGBoost models.

    Parameters:
    - input_data: list or array-like with 13 features in order:
      [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    Returns:
    - Dictionary with lr_predictions and xgb_predictions as percentages
    """
    
    # Convert input to DataFrame with proper column names
    if isinstance(input_data, list):
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("input_data must be a list or DataFrame")
    
    # Load models and preprocessors
    lr_model = load_model('lr_model')
    xgb_model = load_model('xgb_model')
    lr_preprocessor, xgb_preprocessor = _get_preprocessors()
    
    # Preprocess input
    lr_processed = lr_preprocessor.transform(input_df)
    xgb_processed = xgb_preprocessor.transform(input_df)
    
    # Get predictions
    lr_predictions = lr_model.predict_proba(lr_processed)[0][1]
    xgb_predictions = xgb_model.predict_proba(xgb_processed)[0][1]

    return {
        "lr_predictions": round(float(lr_predictions * 100), 1),
        "xgb_predictions": round(float(xgb_predictions * 100), 1)
    }
