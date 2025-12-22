import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_models(X_train, y_train):
    """
    Trains multiple regression models.
    Returns a dictionary of trained models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluates trained models on test data.
    Returns a DataFrame-like structure (or dict) of results.
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Predictions": y_pred # Keep for plotting
        }
        
    return results

def print_model_rankings(results, metric="R2", ascending=False):
    """Prints models ranked by a specific metric."""
    print(f"\nModel Ranking by {metric} ({'Ascending' if ascending else 'Descending'}):")
    sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=not ascending)
    
    for i, (name, metrics) in enumerate(sorted_models, 1):
        print(f"{i}. {name}: {metrics[metric]:.4f}")

def save_best_model(models, results, scaler_X, scaler_y, output_dir='models', metric="R2"):
    """
    Identifies the best model based on R2 score and saves it, along with the scalers.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find best model name
    # We assume higher R2 is better. If using MSE/RMSE, output should be reversed.
    best_model_name = max(results, key=lambda x: results[x][metric])
    best_model = models[best_model_name]
    best_score = results[best_model_name][metric]
    
    print(f"\nBest Model: {best_model_name} with {metric} = {best_score:.4f}")
    
    # Save model
    model_filename = f"best_model_{best_model_name.replace(' ', '_')}.pkl"
    model_filepath = os.path.join(output_dir, model_filename)
    joblib.dump(best_model, model_filepath)
    print(f"Saved best model to: {model_filepath}")

    # Save scalers
    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))
    print(f"Saved scalers to: {output_dir}")
    
    return best_model_name
