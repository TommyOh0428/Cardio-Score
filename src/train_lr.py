import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Import from your project structure
from src.preprocessing import load_xy, make_preprocessor
from src.config import (
    RAW_DATA_PATH, LR_MODEL_PATH, RANDOM_STATE, 
    FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE
)
from src.utils import logger

def train_logistic_regression():
    logger.info("Starting Logistic Regression Training Pipeline...")

    # 1. Load raw data (NO preprocessing yet)
    data_X, data_y = load_xy(RAW_DATA_PATH, target=TARGET_COLUMN, feature_cols=FEATURE_COLUMNS)
    
    # DEBUG: Check data
    logger.info(f"Loaded data shape: X={data_X.shape}, y={data_y.shape}")
    logger.info(f"Target distribution: {data_y.value_counts().to_dict()}")

    # 2. Split data FIRST to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=data_y
    )
    
    logger.info(f"After split: X_train={X_train.shape}, X_test={X_test.shape}")

    # 3. Preprocess: fit on TRAIN only, transform both
    preprocessor = make_preprocessor(True)  # Logistic Regression needs scaling
    X_train = preprocessor.fit_transform(X_train)  # Learn from train only
    X_test = preprocessor.transform(X_test)        # Apply to test
    
    logger.info(f"After preprocessing: X_train={X_train.shape}, X_test={X_test.shape}")

    # 4. Initialize and Train Model
    lr_model = LogisticRegression(
        random_state=RANDOM_STATE, 
        class_weight='balanced',
        max_iter=1000
    )
    
    logger.info("Fitting the Logistic Regression model...")
    lr_model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    # DEBUG: Check predictions
    logger.info(f"Unique predictions: {set(y_pred)}")
    logger.info(f"Prediction distribution: 0={sum(y_pred==0)}, 1={sum(y_pred==1)}")

    logger.info("\n--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # 6. Save Model
    joblib.dump(lr_model, LR_MODEL_PATH)
    relative_path = os.path.relpath(LR_MODEL_PATH)
    logger.info(f"Model saved to {relative_path}")
    logger.info("Logistic Training Pipeline finished.")

if __name__ == "__main__":
    train_logistic_regression()