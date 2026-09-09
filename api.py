import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from feature_engineering import LoanFeatureEngineer

app = FastAPI(title="CrediFlux KNN Loan Intelligence API", version="2.0.0")

# Enable CORS for Next.js development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
PIPELINE = None
RAW_TRAIN_DATA = None
METRICS_DATA = None
DATASET_DF = None


def load_artifacts():
    global PIPELINE, RAW_TRAIN_DATA, METRICS_DATA, DATASET_DF
    model_path = os.path.join(os.path.dirname(__file__), "model", "knn_pipeline.joblib")
    train_path = os.path.join(os.path.dirname(__file__), "model", "train_raw_with_target.csv")
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
    dataset_path = os.path.join(os.path.dirname(__file__), "loan_data.csv")

    if os.path.exists(model_path):
        PIPELINE = joblib.load(model_path)
    if os.path.exists(train_path):
        RAW_TRAIN_DATA = pd.read_csv(train_path)
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            METRICS_DATA = json.load(f)
    if os.path.exists(dataset_path):
        DATASET_DF = pd.read_csv(dataset_path)


load_artifacts()


class ApplicantInput(BaseModel):
    Gender: str = Field(default="Male")
    Married: str = Field(default="Yes")
    Dependents: str = Field(default="0")
    Education: str = Field(default="Graduate")
    Self_Employed: str = Field(default="No")
    ApplicantIncome: float = Field(default=6500.0)
    CoapplicantIncome: float = Field(default=2500.0)
    LoanAmount: float = Field(default=140.0)
    Loan_Amount_Term: float = Field(default=360.0)
    Credit_History: float = Field(default=1.0)
    Property_Area: str = Field(default="Semiurban")
    k_neighbors: Optional[int] = Field(default=15)


class BatchInput(BaseModel):
    applicants: List[ApplicantInput]


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": PIPELINE is not None,
        "training_samples": len(RAW_TRAIN_DATA) if RAW_TRAIN_DATA is not None else 0,
        "default_k": 15,
    }


@app.post("/api/predict")
def predict_single(data: ApplicantInput):
    if PIPELINE is None or RAW_TRAIN_DATA is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    k = max(3, min(data.k_neighbors or 15, 25))

    input_df = pd.DataFrame([{
        "Gender": data.Gender,
        "Married": data.Married,
        "Dependents": data.Dependents,
        "Education": data.Education,
        "Self_Employed": data.Self_Employed,
        "ApplicantIncome": float(data.ApplicantIncome),
        "CoapplicantIncome": float(data.CoapplicantIncome),
        "LoanAmount": float(data.LoanAmount),
        "Loan_Amount_Term": float(data.Loan_Amount_Term),
        "Credit_History": float(data.Credit_History),
        "Property_Area": data.Property_Area,
    }])

    # Dynamically set K on the classifier
    transformer = PIPELINE[:-1]
    knn_clf = PIPELINE[-1]
    knn_clf.set_params(n_neighbors=k)

    transformed_sample = transformer.transform(input_df)
    prediction = int(PIPELINE.predict(input_df)[0])
    proba = PIPELINE.predict_proba(input_df)[0]
    prob_approved = round(float(proba[1]) * 100, 1)
    prob_rejected = round(float(proba[0]) * 100, 1)

    # Nearest neighbors lookup
    distances, indices = knn_clf.kneighbors(transformed_sample, n_neighbors=k)
    neighbor_indices = indices[0]
    neighbor_distances = distances[0]

    neighbors_list = []
    approved_count = 0

    for idx, dist in zip(neighbor_indices, neighbor_distances):
        row = RAW_TRAIN_DATA.iloc[idx]
        status = int(row["Loan_Status"])
        if status == 1:
            approved_count += 1
        
        sim_pct = max(0.0, round(float(1.0 / (1.0 + dist)) * 100, 1))
        
        neighbors_list.append({
            "index": int(idx),
            "distance": round(float(dist), 3),
            "similarity_pct": sim_pct,
            "loan_status": status,
            "applicant_income": float(row.get("ApplicantIncome", 0)),
            "coapplicant_income": float(row.get("CoapplicantIncome", 0)),
            "loan_amount": float(row.get("LoanAmount", 0)) if pd.notna(row.get("LoanAmount")) else None,
            "term": float(row.get("Loan_Amount_Term", 360)) if pd.notna(row.get("Loan_Amount_Term")) else 360.0,
            "credit_history": float(row.get("Credit_History", 1.0)) if pd.notna(row.get("Credit_History")) else None,
            "property_area": str(row.get("Property_Area", "")),
            "education": str(row.get("Education", "")),
            "married": str(row.get("Married", "")),
        })

    total_income = data.ApplicantIncome + data.CoapplicantIncome
    term = max(data.Loan_Amount_Term, 12.0)
    monthly_emi = round((data.LoanAmount * 1000.0) / term, 2)
    dti = round((monthly_emi / (total_income + 1e-5)) * 100, 1)

    dti_cat = "Prime" if dti < 15 else ("Standard" if dti < 35 else "Elevated Leverage")

    return {
        "prediction": prediction,
        "verdict": "APPROVED" if prediction == 1 else "REJECTED",
        "approval_probability": prob_approved,
        "rejection_probability": prob_rejected,
        "approved_neighbors": approved_count,
        "total_neighbors": k,
        "financials": {
            "total_household_income": round(total_income, 2),
            "monthly_emi": monthly_emi,
            "debt_to_income_pct": dti,
            "dti_category": dti_cat,
        },
        "neighbors": neighbors_list,
    }


@app.get("/api/metrics")
def get_metrics():
    if METRICS_DATA is None:
        raise HTTPException(status_code=500, detail="Metrics not available")
    return METRICS_DATA


@app.get("/api/dataset")
def get_dataset_summary(limit: int = 50, offset: int = 0):
    if DATASET_DF is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")

    total_records = len(DATASET_DF)
    records = DATASET_DF.iloc[offset:offset + limit].fillna("N/A").to_dict(orient="records")

    # High-level EDA aggregates
    approved_total = int((DATASET_DF["Loan_Status"] == "Y").sum())
    rejected_total = int((DATASET_DF["Loan_Status"] == "N").sum())

    # Credit History impact
    credit_agg = DATASET_DF.groupby("Credit_History")["Loan_Status"].value_counts(normalize=True).unstack().fillna(0).to_dict()

    # Property Area distribution
    area_agg = DATASET_DF.groupby("Property_Area")["Loan_Status"].value_counts().unstack().fillna(0).to_dict()

    return {
        "total_records": total_records,
        "approved_total": approved_total,
        "rejected_total": rejected_total,
        "overall_approval_rate": round(approved_total / total_records * 100, 1),
        "records": records,
        "aggregates": {
            "credit_history": credit_agg,
            "property_area": area_agg,
        }
    }


@app.post("/api/batch")
def predict_batch(payload: BatchInput):
    results = []
    for item in payload.applicants:
        res = predict_single(item)
        results.append({
            "applicant": item.dict(),
            "verdict": res["verdict"],
            "probability": res["approval_probability"],
            "dti_pct": res["financials"]["debt_to_income_pct"],
        })
    return {"total_evaluated": len(results), "results": results}
