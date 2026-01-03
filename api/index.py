from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Medical Insurance Cost Predictor")

# Paths (Relative to this file's location in api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_Linear_Regression.pkl")
SCALER_X_PATH = os.path.join(BASE_DIR, "models", "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "models", "scaler_y.pkl")

# Load model and scalers
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]):
    MODEL_PATH = "models/best_model_Linear_Regression.pkl"
    SCALER_X_PATH = "models/scaler_X.pkl"
    SCALER_Y_PATH = "models/scaler_y.pkl"
    
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH]):
        raise RuntimeError(f"Model or scalers not found. Checked {BASE_DIR}/models and local models/")

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

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(input_data: InsuranceInput):
    try:
        # Manually performing encoding to ensure stability without pandas
        # Expected features: age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest
        features = [
            input_data.age,
            input_data.bmi,
            input_data.children,
            1 if input_data.sex == 'male' else 0,
            1 if input_data.smoker == 'yes' else 0,
            1 if input_data.region == 'northwest' else 0,
            1 if input_data.region == 'southeast' else 0,
            1 if input_data.region == 'southwest' else 0
        ]
        
        X = np.array([features])
        
        # Scale features
        X_scaled = scaler_X.transform(X)
        
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

# For running locally via 'python api/index.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
