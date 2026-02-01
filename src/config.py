import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw/heart_2022_with_nans.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed/processed_heart.csv')
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target defines "Did the patient have a heart attack?"
TARGET_COLUMN = 'target' # Assumed target column name

FEATURE_COLUMNS = [
    'age',
    'sex',
    'cp', # Chest pain type
    'trestbps', # Resting blood pressure
    'chol', # Serum cholesterol in mg/dl
    'fbs', # Fasting blood sugar > 120 mg/dl
    'restecg', # Resting electrocardiographic results
    'thalach', # Maximum heart rate achieved
    'exang', # Exercise induced angina
    'oldpeak', # ST depression induced by exercise relative to rest
    'slope', # Slope of the peak exercise ST segment
    'ca', # Number of major vessels (0-3) colored by fluoroscopy
    'thal' # Thalassemia
]