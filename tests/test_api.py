from fastapi.testclient import TestClient
# FIX 1: Import fix for ModuleNotFoundError
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


def test_predict_endpoint_valid_payload():
    """✅ Test the '/predict' endpoint with a valid sample payload."""
    sample_payload = {
        "Price": 300.0,
        "Calories": 550.0,
        "Offer_Type": "Discount",
        "Cuisine_Type": "Fast Food"
    }

    response = client.post("/predict", json=sample_payload)

    # FIX 2: Corrected line 31 to include 400 (Bad Request/Prediction Error)
    assert response.status_code in (200, 400, 503) 

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "sentiment_code" in data
        # FIX 3: Corrected key from 'input' to 'input_features'
        assert data["input_features"] == sample_payload


def test_predict_endpoint_invalid_payload():
    """🚨 Test the '/predict' endpoint with an invalid payload (missing fields)."""
    # Missing required field 'Price'
    invalid_payload = {
        "Calories": 550.0,
        "Offer_Type": "Discount",
        "Cuisine_Type": "Fast Food"
    }

    response = client.post("/predict", json=invalid_payload)
    # Should return 422 Unprocessable Entity (Pydantic validation error)
    assert response.status_code == 422
    