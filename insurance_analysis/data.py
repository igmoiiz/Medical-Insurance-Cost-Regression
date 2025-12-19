import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the insurance dataset from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the dataset by removing duplicates and converting boolean columns.
    """
    # Drop duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows.")
    
    # Convert bool to int
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
        
    return df

def preprocess_data(df, target_column='charges', test_size=0.2, random_state=42):
    """
    Preprocesses the data:
    1. One-hot encoding for categorical variables.
    2. Splitting into Train/Test sets.
    3. Scaling features (StandardScaler).
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    # Important: Fit scaler ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
