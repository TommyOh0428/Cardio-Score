import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# page config
st.set_page_config(
    page_title="Cardio-Score - AI Heart Attack Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load trained models: Logistic Regression and XGBoost"""
    try:
        lr_model = joblib.load('models/logistic_regression.joblib')
        xgb_model = joblib.load('models/xgboost_model.joblib')
        lr_preprocessor = joblib.load('models/lr_preprocessor.joblib')
        xgb_preprocessor = joblib.load('models/xgb_preprocessor.joblib')
        return lr_model, xgb_model, lr_preprocessor, xgb_preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

lr_model, xgb_model, lr_preprocessor, xgb_preprocessor = load_models()

st.title("🫀 Cardio-Score - AI Heart Attack Risk Predictor")
st.markdown("### Get AI-powered heart attack risk assessment based on health metrics")

# input feature sidebar
st.sidebar.header("🩺 Patient Health Information")

# demographic info
st.sidebar.subheader("👤 Demographics")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age_category = st.sidebar.selectbox(
    "Age Category",
    ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", 
     "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
     "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
     "Age 80 or older"],
    index=7  # Default to 55-59
)

race_ethnicity = st.sidebar.selectbox(
    "Race/Ethnicity",
    ["White only, Non-Hispanic", "Black only, Non-Hispanic", 
     "Hispanic", "Asian only, Non-Hispanic", "Other race only, Non-Hispanic",
     "Multiracial, Non-Hispanic"],
    index=0
)

# health metric
st.sidebar.subheader("📊 Physical Metrics")
col1, col2 = st.sidebar.columns(2)
with col1:
    height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
with col2:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

sleep_hours = st.sidebar.slider("Sleep Hours per Night", min_value=1.0, max_value=12.0, value=7.0, step=0.5)

# health status
st.sidebar.subheader("🏥 General Health")
general_health = st.sidebar.selectbox(
    "Overall Health Status",
    ["Excellent", "Very good", "Good", "Fair", "Poor"],
    index=2
)

# lifestyle habits
st.sidebar.subheader("🚶 Lifestyle")
physical_activities = st.sidebar.checkbox("Regular Physical Activities", value=True)
alcohol_drinkers = st.sidebar.checkbox("Alcohol Drinker")

# smoking status
st.sidebar.subheader("🚭 Smoking Status")
smoker_status = st.sidebar.selectbox(
    "Smoking History",
    ["Never smoked", "Former smoker", "Current smoker - now smokes every day",
     "Current smoker - now smokes some days"],
    index=0
)

# creating feature vector for prediction
def create_feature_vector():
    """Create feature DataFrame from user inputs"""
    features = {
        'Sex': sex,
        'GeneralHealth': general_health,
        'PhysicalActivities': 'Yes' if physical_activities else 'No',
        'SleepHours': sleep_hours,
        'SmokerStatus': smoker_status,
        'RaceEthnicityCategory': race_ethnicity,
        'AgeCategory': age_category,
        'HeightInMeters': height,
        'WeightInKilograms': weight,
        'AlcoholDrinkers': 'Yes' if alcohol_drinkers else 'No'
    }
    
    return pd.DataFrame([features])

# bmi helper
def calculate_bmi(height_m, weight_kg):
    """Calculate BMI from height and weight"""
    return weight_kg / (height_m ** 2)

# main content area
tab1, tab2 = st.tabs(["🔮 Risk Assessment", "📊 Model Comparison"])

with tab1:
    st.header("Heart Attack Risk Assessment")
    st.markdown("Get AI-powered risk assessment from two complementary models.")
    
    # display BMI
    bmi = calculate_bmi(height, weight)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📏 Height", f"{height} m")
    with col2:
        st.metric("⚖️ Weight", f"{weight} kg")
    with col3:
        bmi_status = "Normal" if 18.5 <= bmi < 25 else ("Underweight" if bmi < 18.5 else "Overweight")
        st.metric("🧮 BMI", f"{bmi:.1f}", delta=bmi_status)
    
    st.markdown("---")
    
    if st.button("🔮 Assess Heart Attack Risk", type="primary", use_container_width=True):
        # get feature vector
        input_features = create_feature_vector()
        
        try:
            # preprocess features using the fitted preprocessors
            lr_features = lr_preprocessor.transform(input_features)
            xgb_features = xgb_preprocessor.transform(input_features)
            
            # get predictions from both models (probability of heart attack)
            lr_risk = float(lr_model.predict_proba(lr_features)[0][1])
            xgb_risk = float(xgb_model.predict_proba(xgb_features)[0][1])
            
        except Exception as e:
            st.error(f"⚠️ Error making predictions: {e}")
            st.info("💡 Make sure you've run the training scripts first to generate the preprocessors!")
            st.stop()
        
        # display results
        st.subheader("🎯 Risk Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Logistic Regression Model")
            st.markdown("*Optimized for high recall - catches most at-risk patients*")
            
            # risk percentage
            lr_percentage = lr_risk * 100
            
            if lr_percentage < 30:
                st.success(f"**Low Risk: {lr_percentage:.1f}%**")
                risk_level = "Low"
                color = "green"
            elif lr_percentage < 60:
                st.warning(f"**Moderate Risk: {lr_percentage:.1f}%**")
                risk_level = "Moderate"
                color = "orange"
            else:
                st.error(f"**High Risk: {lr_percentage:.1f}%**")
                risk_level = "High"
                color = "red"
            
            # progress bar
            st.progress(lr_risk)
            
            st.markdown(f"""
            **Risk Level**: {risk_level}  
            **Model Strength**: High sensitivity (87% recall)  
            **Best For**: Early screening, catching at-risk patients
            """)
        
        with col2:
            st.markdown("### 🌳 XGBoost Model")
            st.markdown("*Balanced accuracy - fewer false alarms*")
            
            # risk percentage
            xgb_percentage = xgb_risk * 100
            
            if xgb_percentage < 30:
                st.success(f"**Low Risk: {xgb_percentage:.1f}%**")
                risk_level = "Low"
                color = "green"
            elif xgb_percentage < 60:
                st.warning(f"**Moderate Risk: {xgb_percentage:.1f}%**")
                risk_level = "Moderate"
                color = "orange"
            else:
                st.error(f"**High Risk: {xgb_percentage:.1f}%**")
                risk_level = "High"
                color = "red"
            
            # progress bar
            st.progress(xgb_risk)
            
            st.markdown(f"""
            **Risk Level**: {risk_level}  
            **Model Strength**: Balanced performance (70% accuracy)  
            **Best For**: General risk assessment
            """)
        
        # avg risk
        avg_risk = (lr_risk + xgb_risk) / 2
        avg_percentage = avg_risk * 100
        
        st.markdown("---")
        st.subheader("📊 Combined Assessment")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            if avg_percentage < 30:
                st.success(f"### ✅ Overall Low Risk: {avg_percentage:.1f}%")
            elif avg_percentage < 60:
                st.warning(f"### ⚠️ Overall Moderate Risk: {avg_percentage:.1f}%")
            else:
                st.error(f"### 🚨 Overall High Risk: {avg_percentage:.1f}%")
        
        with col2:
            st.metric("Average", f"{avg_percentage:.1f}%")
        
        # recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if avg_percentage >= 60:
            st.error("""
            **🚨 High Risk - Immediate Action Recommended**
            - Consult with a healthcare provider immediately
            - Consider lifestyle modifications (diet, exercise, smoking cessation)
            - Regular health monitoring and check-ups
            - Discuss preventive medications with your doctor
            """)
        elif avg_percentage >= 30:
            st.warning("""
            **⚠️ Moderate Risk - Preventive Measures Recommended**
            - Schedule a check-up with your healthcare provider
            - Focus on healthy lifestyle habits
            - Monitor blood pressure and cholesterol
            - Increase physical activity if sedentary
            """)
        else:
            st.success("""
            **✅ Low Risk - Maintain Healthy Habits**
            - Continue current healthy lifestyle
            - Regular annual check-ups
            - Stay physically active
            - Maintain balanced diet
            """)

with tab2:
    st.header("📊 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Logistic Regression")
        st.markdown("""
        **Performance Metrics:**
        - Accuracy: **69.73%**
        - ROC-AUC: **80.62%**
        - **Recall: 77%** (Catches 77% of heart attacks)
        - Precision: 15%
        - F1-Score: 0.25
        
        **Strengths:**
        - Catches 77% of actual heart attack cases (3,798 out of 4,933)
        - Better for medical screening where missing cases is dangerous
        - Interpretable model - can explain predictions
        
        **Trade-offs:**
        - Higher false positive rate (85% of positive predictions are false alarms)
        - Predicts 25,668 heart attacks, but only 15% are correct
        """)
    
    with col2:
        st.subheader("🌳 XGBoost")
        st.markdown("""
        **Performance Metrics:**
        - Accuracy: **69.35%**
        - ROC-AUC: **80.84%**
        - **Recall: 78%** (Catches 78% of heart attacks)
        - Precision: 15%
        - F1-Score: 0.25
        
        **Strengths:**
        - Slightly higher ROC-AUC (80.84% vs 80.62%)
        - Catches 78% of actual heart attack cases (3,848 out of 4,933)
        - Handles complex non-linear patterns
        
        **Trade-offs:**
        - Similar false positive rate to LogReg
        - Less interpretable (black box model)
        """)
    
    st.markdown("---")
    st.info("""
    **🎯 Why Two Models?**
    
    Both models perform very similarly:
    - **Accuracy**: ~69-70% (correctly predicts outcome for ~70% of patients)
    - **ROC-AUC**: ~81% (good discrimination ability between classes)
    - **Recall**: ~77-78% (catches most at-risk patients)
    - **Precision**: 15% (trade-off due to severe 14:1 class imbalance)
    
    The **combined average** gives you a consensus prediction from two different modeling approaches.
    
    **Note on Low Precision:**
    The 15% precision is expected given the severe class imbalance (93.5% healthy vs 6.5% heart attacks).
    To catch 77-78% of heart attacks, the models must be aggressive, resulting in false alarms.
    For medical screening, high recall (catching cases) is more important than precision.
    """)

# footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p><strong>Cardio-Score</strong> - AI Heart Attack Risk Predictor<br>
        Made by Alex Chung & Tommy Oh & Yoon You & Bedro Park
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
