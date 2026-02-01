import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import from your existing files
from config import (
    DATA_PATH, TARGET_COLUMN, LR_PARAMS, 
    RANDOM_STATE, TEST_SIZE, LR_MODEL_PATH
)
from preprocessing import make_preprocessor, load_xy, ALL_FEATURES

def train():
    # 1. Load Data using your helper
    X, y = load_xy(DATA_PATH)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # 3. Build Pipeline
    # For Logistic Regression, scale_numeric MUST be True
    preprocessor = make_preprocessor(scale_numeric=True)
    
    lr_pipeline = Pipeline(steps=[  
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            **LR_PARAMS, 
            random_state=RANDOM_STATE,
            class_weight='balanced' # Handles class balance automatically
        ))
    ])

    # 4. Train Model
    print(f"Training Logistic Regression on {len(X_train)} samples...")
    lr_pipeline.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = lr_pipeline.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Inspect Feature "Emphasis" (Weights)
    feature_names = lr_pipeline.named_steps['preprocessor'].get_feature_names_out()
    coefficients = lr_pipeline.named_steps['classifier'].coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': coefficients
    }).sort_values(by='Weight', ascending=False)
    
    print("\n--- Top Feature Weights (Emphasis) ---")
    print(importance_df.head(10))

    # 7. Save Pipeline
    joblib.dump(lr_pipeline, LR_MODEL_PATH)
    print(f"\nModel successfully saved to {LR_MODEL_PATH}")

if __name__ == "__main__":
    train()