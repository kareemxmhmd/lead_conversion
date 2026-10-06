import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_LEAD_HIGH_POTENTIAL = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 8500,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "oct",
    "campaign": 1,
    "pdays": 90,
    "previous": 2,
    "poutcome": "success",
    "duration": 300,
}

SAMPLE_LEAD_LOW_POTENTIAL = {
    "age": 22,
    "job": "blue-collar",
    "marital": "single",
    "education": "primary",
    "default": "yes",
    "balance": -300,
    "housing": "yes",
    "loan": "yes",
    "contact": "unknown",
    "day": 20,
    "month": "may",
    "campaign": 6,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert "health_url" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "champion_model" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "metrics" in data
    assert "champion_model" in data["metrics"]


def test_predict_single_lead_success():
    response = client.post("/predict", json=SAMPLE_LEAD_HIGH_POTENTIAL)
    assert response.status_code == 200
    data = response.json()
    assert "conversion_probability" in data
    assert isinstance(data["conversion_probability"], float)
    assert 0.0 <= data["conversion_probability"] <= 1.0
    assert "will_convert" in data
    assert isinstance(data["will_convert"], bool)
    assert data["lead_tier"] in ["High", "Medium", "Low"]
    assert "decision_threshold" in data
    assert "recommendation" in data


def test_predict_single_lead_validation_error():
    # Age < 18 should fail pydantic validation
    invalid_lead = SAMPLE_LEAD_HIGH_POTENTIAL.copy()
    invalid_lead["age"] = 12
    response = client.post("/predict", json=invalid_lead)
    assert response.status_code == 422


def test_predict_batch_leads_success():
    payload = {"leads": [SAMPLE_LEAD_HIGH_POTENTIAL, SAMPLE_LEAD_LOW_POTENTIAL]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_leads"] == 2
    assert "mean_conversion_probability" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["lead_tier"] in ["High", "Medium", "Low"]


def test_predict_batch_leads_empty():
    payload = {"leads": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400
