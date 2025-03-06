import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_val, y_val

def save_model(model, filename):
    joblib.dump(model, filename)

def main():
    # Load processed data
    data = pd.read_csv('data/processed/processed_data.csv')
    
    # Train the model
    model, X_val, y_val = train_model(data, target_column='admission_status')
    
    # Save the trained model
    save_model(model, 'models/model.pkl')

if __name__ == "__main__":
    main()