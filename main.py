import os
import pandas as pd
from insurance_analysis import data, visualization, modeling

def main():
    print("Starting Medical Insurance Cost Analysis...")
    
    # 1. Load Data
    filepath = 'insurance.csv'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = data.load_data(filepath)
    print(f"Data Loaded. Shape: {df.shape}")
    
    # 2. EDA (Exploratory Data Analysis)
    print("\nPerforming EDA...")
    visualization.set_style()
    
    # Univariate Analysis
    visualization.plot_distributions(df, ['age', 'bmi', 'children', 'charges'])
    
    # Bivariate Analysis
    visualization.plot_categorical_analysis(df, 'smoker', 'charges')
    visualization.plot_categorical_analysis(df, 'region', 'charges')
    visualization.plot_categorical_analysis(df, 'sex', 'charges')
    
    # Correlation
    visualization.plot_correlation_heatmap(df)
    
    # Pairplot (might be slow for large data, but okay here)
    visualization.plot_pairplot(df, hue='smoker')
    
    # 3. Data Cleaning & Preprocessing
    print("\nPreprocessing Data...")
    df_clean = data.clean_data(df)
    
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = data.preprocess_data(df_clean)
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")
    
    # 4. Model Training
    print("\nTraining Models...")
    trained_models = modeling.train_models(X_train, y_train)
    
    # 5. Evaluation
    print("\nEvaluating Models...")
    results = modeling.evaluate_models(trained_models, X_test, y_test)
    
    # Print metrics
    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"  MSE:  {metrics['MSE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  R2:   {metrics['R2']:.4f}")
        
    modeling.print_model_rankings(results, metric="R2", ascending=False)
    
    # 6. Plot Model Performance
    print("\nPlotting Model Performance...")
    for name, metrics in results.items():
        visualization.plot_actual_vs_predicted(y_test, metrics['Predictions'], name)
        visualization.plot_residuals(y_test, metrics['Predictions'], name)
        
    # 7. Save Best Model
    print("\nSaving Best Model...")
    modeling.save_best_model(trained_models, results, scaler_X, scaler_y, metric="R2")
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()
