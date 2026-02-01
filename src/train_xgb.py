import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from src.preprocessing import load_xy, make_preprocessor
from src.config import RAW_DATA_PATH, XGB_MODEL_PATH, RANDOM_STATE, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.utils import logger

def train_xgboost():
    logger.info("Starting XGBoost Training Pipeline...")

    data_X, data_y = load_xy(RAW_DATA_PATH, target=TARGET_COLUMN, feature_cols=FEATURE_COLUMNS)
    preprocessor = make_preprocessor(True)
    data_X = preprocessor.fit_transform(data_X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data_y
    )

    xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (Heart Attack)

    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")  

    joblib.dump(xgb_model, XGB_MODEL_PATH)
    relative_path = os.path.relpath(XGB_MODEL_PATH)
    logger.info(f"Model saved to {relative_path}")
    logger.info("XGBoost Training Pipeline finished.")
