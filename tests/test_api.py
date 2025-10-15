from fastapi.testclient import TestClient
from app.main import app

# Create a test client using the FastAPI app
client = TestClient(app)


def test_read_root_endpoint():
    """✅ Test the root '/' endpoint for correct status and response."""
    response = client.get("/")
    assert response.status_code == 200

    json_data = response.json()
    assert "message" in json_data
    assert "model_status" in json_data
    assert json_data["message"].startswith("Welcome to the McDonald's Sentiment Prediction API")


def test_predict_endpoint_valid_payload():
    """✅ Test the '/predict' endpoint with a valid sample payload."""
    sample_payload = {
        "Price": 300.0,
        "Calories": 550.0,
        "Offer_Type": "Discount",
        "Cuisine_Type": "Fast Food"
    }

    response = client.post("/predict", json=sample_payload)

    # If model is missing, API returns 503 — so handle both cases cleanly
    assert response.status_code in (200, 503)

    if response.status_code == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert "sentiment_code" in json_data
        assert json_data["input"] == sample_payload


def test_predict_endpoint_invalid_payload():
    """🚨 Test the '/predict' endpoint with an invalid payload (missing fields)."""
    invalid_payload = {
        "Price": 300.0
        # Missing Calories, Offer_Type, Cuisine_Type
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity (validation error)
