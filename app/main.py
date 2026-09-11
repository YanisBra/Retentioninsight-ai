"""
app/main.py — API FastAPI de prédiction du Churn

Projet  : RetentionInsight AI
Auteur  : Équipe Data Science

Routes :
    GET  /          → Message de bienvenue
    GET  /health    → Health check
    GET  /features  → Liste des 30 features attendues
    POST /predict   → Prédiction churn (probabilité + classe)
"""

# stdlib
import json
import os

# third-party
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Chargement du modèle et des feature names
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "churn_model.pkl")
FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "feature_names.json"
)

try:
    pipeline = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        FEATURE_NAMES: list[str] = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Modèle ou feature_names introuvable : {e}\n"
        "Veuillez d'abord exécuter save_model.py."
    )

# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="RetentionInsight AI — Churn Prediction API",
    description=(
        "API de prédiction du churn client pour une entreprise de "
        "télécommunications. Basée sur un RandomForestClassifier entraîné "
        "sur le dataset Telco Customer Churn (Kaggle).\n\n"
        "**Modèle** : RandomForest (n_estimators=300, max_depth=15)\n"
        "**AUC-ROC (test)** : 0.842\n"
        "**Projet** : Directeur de projet IA — Année 1 — L'École Multimédia"
    ),
    version="1.0.0",
)


# ─────────────────────────────────────────────
# Schéma d'entrée
# ─────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Schéma des features d'un client après prétraitement.
    Les 30 colonnes correspondent au dataset processed_telco.csv
    (one-hot encoded, les variables continues tenure, MonthlyCharges et TotalCharges
    sont passées brutes et seront normalisées par le pipeline).
    """

    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    tenure: float = Field(..., example=12.0)
    MonthlyCharges: float = Field(..., example=70.0)
    TotalCharges: float = Field(..., example=840.0)
    gender_Male: int = Field(..., ge=0, le=1, example=0)
    Partner_Yes: int = Field(..., ge=0, le=1, example=1)
    Dependents_Yes: int = Field(..., ge=0, le=1, example=0)
    PhoneService_Yes: int = Field(..., ge=0, le=1, example=0)
    MultipleLines_No_phone_service: int = Field(
        ..., alias="MultipleLines_No phone service", ge=0, le=1, example=1
    )
    MultipleLines_Yes: int = Field(..., ge=0, le=1, example=0)
    InternetService_Fiber_optic: int = Field(
        ..., alias="InternetService_Fiber optic", ge=0, le=1, example=0
    )
    InternetService_No: int = Field(..., ge=0, le=1, example=0)
    OnlineSecurity_No_internet_service: int = Field(
        ..., alias="OnlineSecurity_No internet service", ge=0, le=1, example=0
    )
    OnlineSecurity_Yes: int = Field(..., ge=0, le=1, example=0)
    OnlineBackup_No_internet_service: int = Field(
        ..., alias="OnlineBackup_No internet service", ge=0, le=1, example=0
    )
    OnlineBackup_Yes: int = Field(..., ge=0, le=1, example=1)
    DeviceProtection_No_internet_service: int = Field(
        ..., alias="DeviceProtection_No internet service", ge=0, le=1, example=0
    )
    DeviceProtection_Yes: int = Field(..., ge=0, le=1, example=0)
    TechSupport_No_internet_service: int = Field(
        ..., alias="TechSupport_No internet service", ge=0, le=1, example=0
    )
    TechSupport_Yes: int = Field(..., ge=0, le=1, example=0)
    StreamingTV_No_internet_service: int = Field(
        ..., alias="StreamingTV_No internet service", ge=0, le=1, example=0
    )
    StreamingTV_Yes: int = Field(..., ge=0, le=1, example=0)
    StreamingMovies_No_internet_service: int = Field(
        ..., alias="StreamingMovies_No internet service", ge=0, le=1, example=0
    )
    StreamingMovies_Yes: int = Field(..., ge=0, le=1, example=0)
    Contract_One_year: int = Field(
        ..., alias="Contract_One year", ge=0, le=1, example=0
    )
    Contract_Two_year: int = Field(
        ..., alias="Contract_Two year", ge=0, le=1, example=0
    )
    PaperlessBilling_Yes: int = Field(..., ge=0, le=1, example=1)
    PaymentMethod_Credit_card_automatic: int = Field(
        ..., alias="PaymentMethod_Credit card (automatic)", ge=0, le=1, example=0
    )
    PaymentMethod_Electronic_check: int = Field(
        ..., alias="PaymentMethod_Electronic check", ge=0, le=1, example=1
    )
    PaymentMethod_Mailed_check: int = Field(
        ..., alias="PaymentMethod_Mailed check", ge=0, le=1, example=0
    )

    model_config = {"populate_by_name": True}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Général"])
def read_root() -> dict:
    """Message de bienvenue."""
    return {
        "message": "RetentionInsight AI — API de prédiction du Churn",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Général"])
def health_check() -> dict:
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "n_features": len(FEATURE_NAMES),
    }


@app.get("/features", tags=["Modèle"])
def get_feature_names() -> dict:
    """Retourne la liste ordonnée des 30 features attendues par le modèle."""
    return {"n_features": len(FEATURE_NAMES), "features": FEATURE_NAMES}


@app.post("/predict", tags=["Modèle"])
def predict_churn(customer: CustomerFeatures) -> dict:
    """
    Prédit la probabilité de churn d'un client.

    **Input** : 30 features pré-traitées (encodage one-hot, variables continues brutes).

    **Output** :
    - `churn_probability` : probabilité de résiliation (0 à 1).
    - `churn_prediction`  : 1 si churn prédit, 0 sinon.
    - `risk_level`        : LOW / MEDIUM / HIGH selon la probabilité.
    """
    try:
        # Reconstruire le vecteur dans le bon ordre sous forme de DataFrame (attendu par le preprocessor)
        data = customer.dict(by_alias=True)
        df_input = pd.DataFrame([data])
        df_input = df_input[FEATURE_NAMES]
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature manquante dans la requête : {e}",
        )

    proba = float(pipeline.predict_proba(df_input)[0][1])
    prediction = int(pipeline.predict(df_input)[0])

    if proba < 0.35:
        risk_level = "LOW"
    elif proba < 0.65:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "model": "RandomForest v2 (AUC=0.842)",
    }
