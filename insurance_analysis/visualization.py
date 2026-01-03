import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def set_style():
    plt.rcParams['figure.figsize'] = (10, 6)
    sns.set_theme(style="whitegrid", palette="muted")

def save_plot(filename, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close()

def plot_distributions(df, columns):
    for col in columns:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        save_plot(f'dist_{col}.png')

def plot_categorical_analysis(df, cat_col, target_col):
    # Box plot
    plt.figure()
    sns.boxplot(x=cat_col, y=target_col, data=df)
    plt.title(f'{target_col} by {cat_col} (Box Plot)')
    save_plot(f'box_{cat_col}_vs_{target_col}.png')
    
    # Violin plot
    plt.figure()
    sns.violinplot(x=cat_col, y=target_col, data=df, inner='quartile')
    plt.title(f'{target_col} by {cat_col} (Violin Plot)')
    save_plot(f'violin_{cat_col}_vs_{target_col}.png')

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation (including encodable)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    save_plot('correlation_heatmap.png')

def plot_pairplot(df, hue=None):
    plot = sns.pairplot(df, hue=hue, diag_kind='kde', corner=True)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plot.savefig(os.path.join('plots', 'pairplot.png'))
    print(f"Saved plot: plots/pairplot.png")
    plt.close()

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure()
    
    # Flatten if needed
    y_test_flat = y_test.flatten() if hasattr(y_test, 'flatten') else y_test
    y_pred_flat = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.6)
    
    # Ideal line
    min_val = min(y_test_flat.min(), y_pred_flat.min())
    max_val = max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')
    
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.legend()
    save_plot(f'pred_vs_actual_{model_name.replace(" ", "_")}.png')

def plot_residuals(y_test, y_pred, model_name):
    residual = y_test - y_pred
    plt.figure()
    sns.histplot(residual, kde=True)
    plt.title(f'{model_name}: Residuals Distribution')
    plt.xlabel('Residual')
    save_plot(f'residuals_{model_name.replace(" ", "_")}.png')
