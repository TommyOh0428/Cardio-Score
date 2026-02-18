<h4 align="center">
    <img src="img/logo.png" width="250" alt="Cardio Score Logo"><br>
    Cardio Score<br>
    SystemHacks XHacks
    <div align="center">
    <br>
        <!-- <a href=".">
            <img src="https://github.com/sfuosdev/Website/actions/workflows/node.yml/badge.svg"/>
        </a> -->
    </div>
</h4>

<p align="center">
    <a href="#motivation">Motivation</a> •
    <a href="#team-member">Team Members</a> •
    <a href="#how-to-run-virtual-environment">Set Up</a> •
    <a href="#file-structure">Project Structure</a> 
</p>

## Demo Video

[![Demo Video](https://img.youtube.com)](https://www.youtube.com:9VlthbPYrPU)

## Motivation

Cardiovascular disease remains the leading cause of mortality globally, yet many of its risk factors such as smoking, physical inactivity, and sleep patterns—are modifiable with early intervention. However, traditional medical screenings can be expensive, time-consuming, or inaccessible, leading many individuals to ignore early warning signs until a critical event occurs.

**Cardio-Score** serves as a bridge between complex medical data and personal health awareness. We developed this platform to democratize early screening, allowing anyone to assess their heart attack risk in seconds using non-invasive, self-reportable metrics.

Our system employs a **Hybrid AI Approach**:

1.  **Robust Prediction:** We use two complementary models—**Logistic Regression** (optimized for high recall to ensure at-risk patients aren't missed) and **XGBoost** (for high accuracy on complex, non-linear patient data).
2.  **Personalized Care:** Unlike standard calculators that output a scary number, we integrate **Generative AI (Gemini)** to act as a virtual health consultant, translating raw risk probabilities into compassionate, actionable, and personalized lifestyle advice.

By making risk assessment instant and understandable, we aim to encourage proactive healthcare and motivate users to make life-saving lifestyle changes.

## Team Member

- Tommy Oh koa18@sfu.ca
- Alex Chung sca372@sfu.ca
- Bedro Park bpa51@sfu.ca
- Yoonsang You yya270@sfu.ca

## How to run virtual environment

```bash
# Create the viritual environment
python3 -m venv venv

# Run the script to activate the virtual environment
source ./activate_venv.sh

# Train the each model and saved to models folder
python -m src.train_lr # Training Logistic Regression model
python -m src.train_xgb # Training XGBoost model

# Run the streamlit application
streamlit run app.py
```

## File Structure

```txt
cardio-score/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ app.py (streamlit application)
├─ test_predict.py
├─ activate_venv.sh (shell script to activate the virtual environment easily)
│
├─ data/
│  └─ heart_data.csv
│
├─ models/
│  ├─ lr_model.joblib
│  └─ xgb_model.joblib
│
├─ src/
│  ├─ config.py            # column names, label, feature sets
│  ├─ preprocess.py        # build preprocessor (impute/encode/scale)
│  ├─ train_lr.py          # trains + saves LR
│  ├─ train_xgb.py         # trains + saves XGB
│  ├─ evaluate.py          # AUROC + (optional) calibration + save metrics
│  └─ predict.py           # load model -> prob -> % risk
```
