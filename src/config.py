import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw/dataset.csv')  # Updated to new dataset
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed/processed_heart.csv')
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target defines "Did the patient have a heart attack?"
TARGET_COLUMN = 'HadHeartAttack'

# Feature columns for the new dataset
FEATURE_COLUMNS = [
    'Sex',
    'GeneralHealth',
    'PhysicalActivities',
    'SleepHours',
    'SmokerStatus',
    'RaceEthnicityCategory',
    'AgeCategory',
    'HeightInMeters',
    'WeightInKilograms',
    'AlcoholDrinkers'
]