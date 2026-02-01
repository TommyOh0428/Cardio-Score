"""
Quick test script for the prediction function.
Tests with sample patient data.
"""

from src.predict import predict_with_two_model
import numpy as np

# Sample patient data (matching the 13 features from config.py)
# ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
#  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

print("="*60)
print("HEART ATTACK RISK PREDICTION TEST")
print("="*60)

# Test Case 1: High-risk patient
print("\n📋 Test Case 1: High-Risk Patient")
print("-" * 60)
high_risk_patient = [
    65,   # age: 65 years old
    1,    # sex: male
    3,    # cp: chest pain type 3 (asymptomatic)
    160,  # trestbps: high blood pressure
    286,  # chol: high cholesterol
    0,    # fbs: fasting blood sugar <= 120
    2,    # restecg: abnormal ECG
    108,  # thalach: low max heart rate
    1,    # exang: exercise induced angina
    1.5,  # oldpeak: ST depression
    2,    # slope: downsloping
    3,    # ca: 3 major vessels
    3     # thal: reversible defect
]

try:
    result = predict_with_two_model(high_risk_patient)
    print(f"Logistic Regression: {result['lr_predictions']}% risk")
    print(f"XGBoost:            {result['xgb_predictions']}% risk")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test Case 2: Low-risk patient
print("\n📋 Test Case 2: Low-Risk Patient")
print("-" * 60)
low_risk_patient = [
    35,   # age: 35 years old
    0,    # sex: female
    0,    # cp: typical angina
    110,  # trestbps: normal blood pressure
    180,  # chol: normal cholesterol
    0,    # fbs: fasting blood sugar <= 120
    0,    # restecg: normal ECG
    180,  # thalach: high max heart rate
    0,    # exang: no exercise induced angina
    0.0,  # oldpeak: no ST depression
    1,    # slope: upsloping
    0,    # ca: 0 major vessels
    2     # thal: normal
]

try:
    result = predict_with_two_model(low_risk_patient)
    print(f"Logistic Regression: {result['lr_predictions']}% risk")
    print(f"XGBoost:            {result['xgb_predictions']}% risk")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test Case 3: Medium-risk patient
print("\n📋 Test Case 3: Medium-Risk Patient")
print("-" * 60)
medium_risk_patient = [
    50,   # age: 50 years old
    1,    # sex: male
    1,    # cp: atypical angina
    130,  # trestbps: slightly elevated BP
    220,  # chol: slightly elevated cholesterol
    0,    # fbs: normal
    0,    # restecg: normal
    150,  # thalach: moderate heart rate
    0,    # exang: no exercise angina
    1.0,  # oldpeak: mild ST depression
    1,    # slope: flat
    1,    # ca: 1 vessel
    2     # thal: normal
]

try:
    result = predict_with_two_model(medium_risk_patient)
    print(f"Logistic Regression: {result['lr_predictions']}% risk")
    print(f"XGBoost:            {result['xgb_predictions']}% risk")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

print("\n" + "="*60)
print("✅ Prediction test complete!")
print("="*60)
