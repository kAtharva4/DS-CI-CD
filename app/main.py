import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ------------------------------
# Initialize FastAPI app
# ------------------------------
app = FastAPI(
    title="McDonald's Sentiment Prediction API",
    description="Predict sentiment (positive/negative/neutral) based on menu item features.",
    version="1.0.0"
)

# ------------------------------
# Input schema using Pydantic
# ------------------------------
class InputFeatures(BaseModel):
    Price: float = Field(..., description="Price of the menu item (currency units).")
    Calories: float = Field(..., description="Calorie count of the menu item.")
    Offer_Type: str = Field(..., description="Type of offer (Discount, BOGO, Freebie, No Offer).")
    Cuisine_Type: str = Field(..., description="Cuisine type (Fast Food, Indian, etc.)")

    # Example for Swagger/OpenAPI
    model_config = {
        "json_schema_extra": {
            "example": {
                "Price": 300.0,
                "Calories": 550.0,
                "Offer_Type": "Discount",
                "Cuisine_Type": "Fast Food"
            }
        }
    }


# ------------------------------
# Load ML model on startup
# ------------------------------
MODEL_PATH = "best_sentiment_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    # Mapping model output integers back to sentiment labels
    PREDICTION_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
except FileNotFoundError:
    model = None
    print(f"❌ Model file '{MODEL_PATH}' not found. Place it in the same directory.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")


# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the McDonald's Sentiment Prediction API!",
        "model_status": "Loaded ✅" if model else "Error ❌ - Model not loaded"
    }


# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict", tags=["Prediction"])
def predict(input_data: InputFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot make predictions.")

    try:
        # Convert Pydantic model to a single-row DataFrame
        features_dict = input_data.model_dump()
        input_df = pd.DataFrame([features_dict], columns=['Price', 'Calories', 'Offer_Type', 'Cuisine_Type'])

        # Make prediction
        encoded_pred = model.predict(input_df)[0]
        sentiment = PREDICTION_MAPPING.get(int(encoded_pred), "Unknown")

        return {
            "prediction": sentiment,
            "sentiment_code": int(encoded_pred),
            "input_features": features_dict
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# ------------------------------
# Run locally for testing
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
