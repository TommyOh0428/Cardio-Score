# Cardio Score

## How to run virtual environment

```bash
# Create the viritual environment
python3 -m venv venv

# Run the script to activate the virtual environment
source ./activate_venv.sh
```
## Pipeline

```txt
heart-attack-risk/
├─ README.md
├─ requirements.txt
├─ .gitignore
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
│
└─ app/
   └─ streamlit_app.py     # demo UI (basic vs advanced toggle)
```
