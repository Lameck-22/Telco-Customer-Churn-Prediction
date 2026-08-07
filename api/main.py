import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

labels = joblib.load(r"D:\Telco-Customer-Churn-Prediction\models\target_labels.joblib")
model = joblib.load(r"D:\Telco-Customer-Churn-Prediction\models\ml_pipeline.joblib")

app = FastAPI(title="customer_churn_api")


class Input(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float


@app.get("/health")
def healthcheck():
    return {"status": "API is running successfully"}


@app.post("/predict")
def predict(payload: Input):
    input_df = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(input_df)[0]
    prediction_label = labels.inverse_transform([prediction])[0]
    return {"prediction": prediction_label}