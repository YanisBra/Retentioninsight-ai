import sys
import os
from fastapi.testclient import TestClient

# Ajouter le chemin parent pour importer app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "n_features" in data


def test_get_features():
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 30
    assert len(data["features"]) == 30


def test_predict_churn_success():
    payload = {
        "SeniorCitizen": 0,
        "tenure": 12.0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender_Male": 1,
        "Partner_Yes": 1,
        "Dependents_Yes": 0,
        "PhoneService_Yes": 1,
        "MultipleLines_No phone service": 0,
        "MultipleLines_Yes": 1,
        "InternetService_Fiber optic": 1,
        "InternetService_No": 0,
        "OnlineSecurity_No internet service": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_No internet service": 0,
        "OnlineBackup_Yes": 1,
        "DeviceProtection_No internet service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No internet service": 0,
        "TechSupport_Yes": 1,
        "StreamingTV_No internet service": 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_No internet service": 0,
        "StreamingMovies_Yes": 1,
        "Contract_One year": 0,
        "Contract_Two year": 0,
        "PaperlessBilling_Yes": 1,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 1,
        "PaymentMethod_Mailed check": 0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_churn_validation_error():
    # Corps incomplet et valeurs hors limites (SeniorCitizen=5)
    payload = {
        "SeniorCitizen": 5,
        "tenure": 12.0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender_Male": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
