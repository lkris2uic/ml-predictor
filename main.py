from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.pkl")

# Define the input schema
class PredictionRequest(BaseModel):
    features: list[float]

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)

    # Ensure the correct number of features
    if features.shape[1] != 20:
        raise HTTPException(status_code=400, detail="Input must have 20 features")

    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
