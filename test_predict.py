"""
Quick test script for the prediction function.
Tests with sample patient data (10 features from new dataset).
"""

from src.predict import predict_with_two_model

print("="*60)
print("HEART ATTACK RISK PREDICTION TEST (New Dataset - 10 Features)")
print("="*60)

# Features: Sex, GeneralHealth, PhysicalActivities, SleepHours, SmokerStatus,
#           RaceEthnicityCategory, AgeCategory, HeightInMeters, WeightInKilograms, AlcoholDrinkers

# Test Case 1: High-Risk Patient
print("\n📋 Test Case 1: High-Risk Patient")
print("-" * 60)
high_risk_patient = [
    "Male",              # Sex
    "Poor",              # GeneralHealth
    "No",                # PhysicalActivities
    5,                   # SleepHours
    "Current smoker",    # SmokerStatus
    "White",             # RaceEthnicityCategory
    "65 or older",       # AgeCategory
    1.70,                # HeightInMeters
    100,                 # WeightInKilograms
    "Yes"                # AlcoholDrinkers
]

try:
    result = predict_with_two_model(high_risk_patient)
    print(f"Logistic Regression: {result['lr_predictions']}% risk")
    print(f"XGBoost:            {result['xgb_predictions']}% risk")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test Case 2: Low-Risk Patient
print("\n📋 Test Case 2: Low-Risk Patient")
print("-" * 60)
low_risk_patient = [
    "Female",            # Sex
    "Excellent",         # GeneralHealth
    "Yes",               # PhysicalActivities
    8,                   # SleepHours
    "Never smoked",      # SmokerStatus
    "White",             # RaceEthnicityCategory
    "18-24",             # AgeCategory
    1.65,                # HeightInMeters
    65,                  # WeightInKilograms
    "No"                 # AlcoholDrinkers
]

try:
    result = predict_with_two_model(low_risk_patient)
    print(f"Logistic Regression: {result['lr_predictions']}% risk")
    print(f"XGBoost:            {result['xgb_predictions']}% risk")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test Case 3: Medium-Risk Patient
print("\n📋 Test Case 3: Medium-Risk Patient")
print("-" * 60)
medium_risk_patient = [
    "Male",              # Sex
    "Good",              # GeneralHealth
    "Yes",               # PhysicalActivities
    7,                   # SleepHours
    "Former smoker",     # SmokerStatus
    "Black",             # RaceEthnicityCategory
    "45-49",             # AgeCategory
    1.75,                # HeightInMeters
    85,                  # WeightInKilograms
    "Yes"                # AlcoholDrinkers
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
