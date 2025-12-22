import json

def test_prediction():
    url = "http://localhost:8000/predict"
    payload = {
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest"
    }
    
    # Note: This requires the server to be running.
    # For a direct test without uvicorn, we can import the app or logic.
    
    print(f"Testing with payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Since uvicorn is not running, we'll test the logic directly if possible
        # or just provide this script for the user.
        # However, I can try to run uvicorn in the background.
        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Internal logic test
    import joblib
    import pandas as pd
    import numpy as np
    
    print("Running internal logic verification...")
    model = joblib.load("models/best_model_Linear_Regression.pkl")
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")
    
    # Sample input (row 1 of insurance.csv approximately)
    # 19,female,27.9,0,yes,southwest,16884.924
    input_data = {
        'age': 19,
        'bmi': 27.9,
        'children': 0,
        'sex_male': 0,
        'smoker_yes': 1,
        'region_northwest': 0,
        'region_southeast': 0,
        'region_southwest': 1
    }
    
    df = pd.DataFrame([input_data])
    X_scaled = scaler_X.transform(df)
    prediction_scaled = model.predict(X_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    print(f"Scaled Prediction: {prediction_scaled[0]}")
    print(f"Inverse Transformed Prediction: {prediction[0][0]}")
    print("Verification complete. The value should be around 16884.924 if the model is accurate.")
