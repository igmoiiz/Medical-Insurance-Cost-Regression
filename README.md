# Medical Insurance Cost Analysis

## Overview
This repository contains a dataset of medical insurance costs and personal attributes.  
The accompanying Python script performs exploratory data analysis (EDA), visualizations, data preprocessing, 
and applies machine learning regression models to predict insurance charges based on features such as:

- Age  
- BMI (Body Mass Index)  
- Number of children  
- Smoking status  
- Region  
- Sex  

The analysis aims to understand feature distributions, uncover relationships in the data, and evaluate model 
performances in predicting insurance charges.

---

## Requirements

- Python **3.12** or higher  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Dataset

The dataset is provided in `insurance.csv`. It includes ~1338 observations with the following columns:

- **age**: Age of the primary beneficiary  
- **sex**: Gender of the beneficiary (male/female)  
- **bmi**: Body Mass Index  
- **children**: Number of dependents covered by insurance  
- **smoker**: Smoking status (yes/no)  
- **region**: Residential area in the US (northeast, northwest, southeast, southwest)  
- **charges**: Individual medical costs billed by health insurance  

### Sample Data Preview
```csv
age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
28,male,33,3,no,southeast,4449.462
33,male,22.705,0,no,northwest,21984.47061
32,male,28.88,0,no,northwest,3866.8552
```
*(Dataset contains ~1338 rows in total)*

---

## Analysis Workflow

### 1. Data Inspection  
- Checked datatypes, missing values, and duplicates  
- Dropped duplicate rows (if any)  

### 2. Visualizations  
- Histograms: **Age, BMI, Charges**  
- Bar plots: **Smokers vs Non-smokers**  
- Violin plots: **Age vs Smoker distribution**  

### 3. Data Preprocessing  
- One-hot encoding of categorical variables (`sex`, `region`, `smoker`)  
- Converted booleans to integers  
- Standard scaling applied to features and target variable  

### 4. Model Training  
Trained the following regression models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

### 5. Model Evaluation  
Calculated metrics:
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **Mean Absolute Error (MAE)**  
- **R² Score**  

---

## Example Code Snippet (Metrics Calculation)
```python
# Mean Absolute Error
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mae_gradient = mean_absolute_error(y_test, y_pred_gradient)

# Root Mean Squared Error
rmse_linear = np.sqrt(mse_linear)
rmse_tree = np.sqrt(mse_tree)
rmse_forest = np.sqrt(mse_forest)
rmse_gradient = np.sqrt(mse_gradient)

# Print Results
print("Linear Regression:")
print(f"  MSE: {mse_linear:.4f}, RMSE: {rmse_linear:.4f}, R2: {r2_linear:.4f}, MAE: {mae_linear:.4f}")
print("Decision Tree:")
print(f"  MSE: {mse_tree:.4f}, RMSE: {rmse_tree:.4f}, R2: {r2_tree:.4f}, MAE: {mae_tree:.4f}")
print("Random Forest:")
print(f"  MSE: {mse_forest:.4f}, RMSE: {rmse_forest:.4f}, R2: {r2_forest:.4f}, MAE: {mae_forest:.4f}")
print("Gradient Boosting:")
print(f"  MSE: {mse_gradient:.4f}, RMSE: {rmse_gradient:.4f}, R2: {r2_gradient:.4f}, MAE: {mae_gradient:.4f}")
```

---

## Results & Model Comparison
The models are evaluated and ranked by **MSE** (lower is better) and **R²** (higher is better).  
In most cases:
- **Gradient Boosting** and **Random Forest** outperform Linear Regression and Decision Tree models.  
- **Linear Regression** provides a good baseline but struggles with non-linear patterns.  
- **Decision Tree** may overfit without tuning.  

---

## API Implementation

This project includes a FastAPI-based web server to serve the model predictions.

### Running the Server
The best practice for running the API is using the **Uvicorn** CLI from the project root:

```bash
uvicorn app.main:app --reload
```
- **`app.main:app`**: Points to the `app` instance in `app/main.py`.
- **`--reload`**: Enables auto-restart on code changes.

### API Endpoints
- **`GET /`**: Health check and welcome message.
- **`POST /predict`**: Accepts user attributes and returns a prediction.

#### Sample Request Body (JSON)
```json
{
    "age": 34,
    "sex": "female",
    "bmi": 23.5,
    "children": 2,
    "smoker": "no",
    "region": "northwest"
}
```

---

## Understanding the Results

When you call the `/predict` endpoint, the API returns two main values:

### 1. `prediction_scaled`
*   **What is it?**: This is the raw output from the machine learning model.
*   **Why is it scaled?**: During training, we scale the target variable (insurance charges) to help the model learn more efficiently. This value is in the "scaled space" (typically Z-score normalized where 0 is average).
*   **Units**: Unitless (normalized value).

### 2. `insurance_cost`
*   **What is it?**: This is the final prediction converted back into real-world units.
*   **Units**: **US Dollars ($)**.
*   **Meaning**: This is the estimated **annual** medical insurance cost for the individual based on their profile.

---

## Files in Repository
- `insurance.csv` → Dataset  
- `main.py` → Script for data analysis and model training workflow.
- `app/main.py` → FastAPI application for serving predictions.
- `models/` → Saved `.pkl` files for the trained model and scalers.
- `postman.json` → Postman collection for testing the API.
- `README.md` → Documentation (this file)

---

## Conclusion
This project demonstrates how personal attributes (age, BMI, smoking status, etc.) influence medical insurance costs.  
Ensemble methods like **Random Forest** and **Gradient Boosting** consistently deliver better predictive performance, while the FastAPI implementation provides an easy way to consume these predictions in external applications.

---
