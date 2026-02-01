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

    # load raw data (NO preprocessing yet)
    data_X, data_y = load_xy(RAW_DATA_PATH, target=TARGET_COLUMN, feature_cols=FEATURE_COLUMNS)
    
    # debug
    logger.info(f"Loaded data shape: X={data_X.shape}, y={data_y.shape}")
    logger.info(f"Target distribution: {data_y.value_counts().to_dict()}")

    # splitting data first to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data_y
    )
    
    logger.info(f"After split: X_train={X_train.shape}, X_test={X_test.shape}")

    # preprocess: fit on train only, transform both
    preprocessor = make_preprocessor(False)  # doesn't need scaling
    X_train = preprocessor.fit_transform(X_train)  # Learn from train only
    X_test = preprocessor.transform(X_test)        # Apply to test
    
    logger.info(f"After preprocessing: X_train={X_train.shape}, X_test={X_test.shape}")

    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE, 
        eval_metric='logloss',
        max_depth=3,           # limit tree depth to prevent overfitting
        n_estimators=100,      # number of trees
        learning_rate=0.1,     # step size
        subsample=0.8,         # use 80% of data per tree
        colsample_bytree=0.8   # use 80% of features per tree
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # debug
    logger.info(f"Unique predictions: {set(y_pred)}")
    logger.info(f"Prediction distribution: 0={sum(y_pred==0)}, 1={sum(y_pred==1)}")

    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    joblib.dump(xgb_model, XGB_MODEL_PATH)
    relative_path = os.path.relpath(XGB_MODEL_PATH)
    logger.info(f"Model saved to {relative_path}")
    logger.info("XGBoost Training Pipeline finished.")
