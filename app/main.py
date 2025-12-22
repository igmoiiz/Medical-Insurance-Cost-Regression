from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Medical Insurance Cost Predictor")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_Linear_Regression.pkl")
SCALER_X_PATH = os.path.join(BASE_DIR, "models", "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "models", "scaler_y.pkl")

# Load model and scalers
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]):
    raise RuntimeError("Model or scalers not found. Please run main.py first.")

model = joblib.load(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Insurance Cost Prediction API"}

@app.post("/predict")
def predict(input_data: InsuranceInput):
    try:
        # Convert input to DataFrame
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Performing encoding to ensure stability
        processed_df = pd.DataFrame({
            'age': [df['age'][0]],
            'bmi': [df['bmi'][0]],
            'children': [df['children'][0]],
            'sex_male': [1 if df['sex'][0] == 'male' else 0],
            'smoker_yes': [1 if df['smoker'][0] == 'yes' else 0],
            'region_northwest': [1 if df['region'][0] == 'northwest' else 0],
            'region_southeast': [1 if df['region'][0] == 'southeast' else 0],
            'region_southwest': [1 if df['region'][0] == 'southwest' else 0]
        })
        
        # Scale features
        X_scaled = scaler_X.transform(processed_df)
        
        # Predict
        prediction_scaled = model.predict(X_scaled)
        
        # De-scale prediction
        # model.predict returns a 1D array so reshaping the array for inverse_transform
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        return {
            "prediction_scaled": float(prediction_scaled[0]),
            "insurance_cost": float(prediction[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
