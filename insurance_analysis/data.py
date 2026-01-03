import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
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
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column].values.reshape(-1, 1) # Reshape for scaler
    
    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled.ravel(), y_test_scaled.ravel(), scaler_X, scaler_y
