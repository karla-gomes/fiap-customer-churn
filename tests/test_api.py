from fastapi.testclient import TestClient
import pytest

from src.api import CustomerFeatures, app, get_predictor
from src.config import get_model_path

PAYLOAD = CustomerFeatures.model_config["json_schema_extra"]["example"]


class FakePredictor:
    def predict(self, data: dict) -> dict:
        assert "Tenure Months" in data  # payload chega com os nomes de coluna do treino
        return {"churn_propensity": 0.8, "high_risk": True}


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_propensity(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"churn_propensity": 0.8, "high_risk": True}


def test_predict_rejects_invalid_payload(client):
    invalid = {**PAYLOAD, "Contract": "Weekly"}
    assert client.post("/predict", json=invalid).status_code == 422


@pytest.mark.skipif(not get_model_path().exists(), reason="artefato do modelo ausente")
def test_predict_with_real_model():
    response = TestClient(app).post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_propensity"] <= 1.0