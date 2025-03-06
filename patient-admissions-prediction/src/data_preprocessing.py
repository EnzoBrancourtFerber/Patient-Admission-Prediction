import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Define categorical and numerical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()

    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Apply transformations
    X = df.drop('target', axis=1)  # Assuming 'target' is the label column
    y = df['target']
    
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)