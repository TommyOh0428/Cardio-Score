import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_both_models(X_test, y_test):
    """
    Evaluate and compare both LR and XGBoost models.
    
    Parameters:
    - X_test: Test features
    - y_test: True labels
    
    Returns:
    - Dictionary containing metrics for both models
    """
    
    # Load both models
    lr_model = joblib.load('models/logistic_regression.joblib')
    xgb_model = joblib.load('models/xgboost_model.joblib')
    
    # Get predictions for both models
    lr_pred = lr_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    # Get probabilities for ROC-AUC
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for LR
    lr_metrics = {
        'model': 'Logistic Regression',
        'accuracy': round(accuracy_score(y_test, lr_pred) * 100, 2),
        'precision': round(precision_score(y_test, lr_pred) * 100, 2),
        'recall': round(recall_score(y_test, lr_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, lr_pred) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, lr_proba) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, lr_pred).tolist()
    }
    
    # Calculate metrics for XGBoost
    xgb_metrics = {
        'model': 'XGBoost',
        'accuracy': round(accuracy_score(y_test, xgb_pred) * 100, 2),
        'precision': round(precision_score(y_test, xgb_pred) * 100, 2),
        'recall': round(recall_score(y_test, xgb_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, xgb_pred) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, xgb_proba) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist()
    }
    
    # Create comparison
    comparison = {
        'logistic_regression': lr_metrics,
        'xgboost': xgb_metrics,
        'winner': {
            'accuracy': 'LR' if lr_metrics['accuracy'] > xgb_metrics['accuracy'] else 'XGB',
            'precision': 'LR' if lr_metrics['precision'] > xgb_metrics['precision'] else 'XGB',
            'recall': 'LR' if lr_metrics['recall'] > xgb_metrics['recall'] else 'XGB',
            'f1_score': 'LR' if lr_metrics['f1_score'] > xgb_metrics['f1_score'] else 'XGB',
            'roc_auc': 'LR' if lr_metrics['roc_auc'] > xgb_metrics['roc_auc'] else 'XGB'
        }
    }
    
    return comparison

def print_evaluation_results(comparison):
    """
    Pretty print the evaluation results.
    
    Parameters:
    - comparison: Dictionary from evaluate_both_models()
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION COMPARISON")
    print("="*60)
    
    # Logistic Regression Results
    lr = comparison['logistic_regression']
    print(f"\n📊 {lr['model']}")
    print("-" * 60)
    print(f"Accuracy:  {lr['accuracy']}%")
    print(f"Precision: {lr['precision']}%")
    print(f"Recall:    {lr['recall']}%")
    print(f"F1 Score:  {lr['f1_score']}%")
    print(f"ROC-AUC:   {lr['roc_auc']}%")
    print(f"\nConfusion Matrix:")
    cm = lr['confusion_matrix']
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    # XGBoost Results
    xgb = comparison['xgboost']
    print(f"\n📊 {xgb['model']}")
    print("-" * 60)
    print(f"Accuracy:  {xgb['accuracy']}%")
    print(f"Precision: {xgb['precision']}%")
    print(f"Recall:    {xgb['recall']}%")
    print(f"F1 Score:  {xgb['f1_score']}%")
    print(f"ROC-AUC:   {xgb['roc_auc']}%")
    print(f"\nConfusion Matrix:")
    cm = xgb['confusion_matrix']
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    
    # Winner Summary
    print("\n" + "="*60)
    print("🏆 WINNER BY METRIC")
    print("="*60)
    winners = comparison['winner']
    for metric, winner in winners.items():
        print(f"{metric.upper():12} → {winner}")
    
    # Overall recommendation
    print("\n" + "="*60)
    lr_wins = sum(1 for w in winners.values() if w == 'LR')
    xgb_wins = sum(1 for w in winners.values() if w == 'XGB')
    
    if xgb_wins > lr_wins:
        print("✅ OVERALL: XGBoost performs better on most metrics")
    elif lr_wins > xgb_wins:
        print("✅ OVERALL: Logistic Regression performs better on most metrics")
    else:
        print("✅ OVERALL: Both models perform equally well")
    print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    from src.preprocessing import make_preprocessor, load_xy
    from src.config import RAW_DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS
    
    parser = argparse.ArgumentParser(description='Evaluate and compare LR and XGBoost models')
    parser.add_argument('--test-data', required=True, help='Path to test CSV file')
    parser.add_argument('--target', default='target', help='Target column name')
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    df = pd.read_csv(args.test_data)
    
    # Separate features and target
    y_test = df[args.target]
    
    # Convert target to numeric if it's Yes/No
    if y_test.dtype == 'object' and set(y_test.unique()) <= {'Yes', 'No', 'yes', 'no'}:
        y_test = (y_test.str.lower() == 'yes').astype(int)
        print(f"Converted target '{args.target}' from Yes/No to 1/0")
    
    X_test_raw = df.drop(columns=[args.target])
    
    # Load training data to fit preprocessors
    print("Fitting preprocessors on training data...")
    train_X, train_y = load_xy(RAW_DATA_PATH, target=TARGET_COLUMN, feature_cols=FEATURE_COLUMNS)
    
    # Create and fit preprocessors
    lr_preprocessor = make_preprocessor(True)  # with scaling
    xgb_preprocessor = make_preprocessor(False)  # without scaling
    lr_preprocessor.fit(train_X)
    xgb_preprocessor.fit(train_X)
    
    # Preprocess test data
    X_test_lr = lr_preprocessor.transform(X_test_raw)
    X_test_xgb = xgb_preprocessor.transform(X_test_raw)
    
    # Load models
    lr_model = joblib.load('models/logistic_regression.joblib')
    xgb_model = joblib.load('models/xgboost_model.joblib')
    
    # Evaluate both models
    print("\nEvaluating models...")
    
    # Get predictions for both models
    lr_pred = lr_model.predict(X_test_lr)
    xgb_pred = xgb_model.predict(X_test_xgb)
    
    # Get probabilities for ROC-AUC
    lr_proba = lr_model.predict_proba(X_test_lr)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # Build results manually since we have different preprocessed data
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    lr_metrics = {
        'model': 'Logistic Regression',
        'accuracy': round(accuracy_score(y_test, lr_pred) * 100, 2),
        'precision': round(precision_score(y_test, lr_pred) * 100, 2),
        'recall': round(recall_score(y_test, lr_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, lr_pred) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, lr_proba) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, lr_pred).tolist()
    }
    
    xgb_metrics = {
        'model': 'XGBoost',
        'accuracy': round(accuracy_score(y_test, xgb_pred) * 100, 2),
        'precision': round(precision_score(y_test, xgb_pred) * 100, 2),
        'recall': round(recall_score(y_test, xgb_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, xgb_pred) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, xgb_proba) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist()
    }
    
    results = {
        'logistic_regression': lr_metrics,
        'xgboost': xgb_metrics,
        'winner': {
            'accuracy': 'LR' if lr_metrics['accuracy'] > xgb_metrics['accuracy'] else 'XGB',
            'precision': 'LR' if lr_metrics['precision'] > xgb_metrics['precision'] else 'XGB',
            'recall': 'LR' if lr_metrics['recall'] > xgb_metrics['recall'] else 'XGB',
            'f1_score': 'LR' if lr_metrics['f1_score'] > xgb_metrics['f1_score'] else 'XGB',
            'roc_auc': 'LR' if lr_metrics['roc_auc'] > xgb_metrics['roc_auc'] else 'XGB'
        }
    }
    
    # Print results
    print_evaluation_results(results)
    
