# Telco-Customer-Churn-Prediction
A machine learning project that predicts whether a telecommunications customer is likely to churn based on demographic information, subscribed services, billing information, and contract details.

The project covers the complete machine learning lifecycle including:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model serialization
- REST API deployment using FastAPI

## Project Structure
telco-churn-prediction/
│
├── api/
│   └── main.py                # FastAPI application
│
├── data/
│   └── telco_churn.csv        # Dataset
│
├── models/
│   ├── ml_pipeline.joblib     # Trained machine learning pipeline
│   └── target_labels.joblib   # LabelEncoder for target decoding
│
├── notebooks/
│   └── project.ipynb          # Data analysis and model development
│
├── README.md
└── .gitignore
