# train_models.py - Model Training Script
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

MODEL_FOLDER = 'models/'

def load_disease_data(disease_name):
    data_files = {
        "diabetes": "data/diabetes.csv",
        "heart": "data/heart_disease.csv",
        "liver": "data/liver_disease.csv"
    }
    
    df = pd.read_csv(data_files[disease_name])
    
    # Diabetes preprocessing
    if disease_name == "diabetes":
        zero_as_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_as_nan_cols] = df[zero_as_nan_cols].replace(0, np.nan)
        df[zero_as_nan_cols] = df[zero_as_nan_cols].fillna(df[zero_as_nan_cols].mean())
    
    # Heart disease preprocessing
    elif disease_name == "heart":
        df = df.replace('?', np.nan)
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['ca', 'thal']:
            df[col].fillna(df[col].mode()[0], inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
    
    # Liver disease preprocessing
    elif disease_name == "liver":
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df = df.rename(columns={'Dataset': 'target'})
        df['Albumin_and_Globulin_Ratio'].fillna(
            df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
    
    return df

def train_disease_model(disease_name):
    try:
        df = load_disease_data(disease_name)
        
        disease_config = {
            "diabetes": {
                "target": 'Outcome',
                "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                "model": LogisticRegression(max_iter=1000)
            },
            "heart": {
                "target": 'target',
                "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
                "model": RandomForestClassifier(n_estimators=100, random_state=42)
            },
            "liver": {
                "target": 'target',
                "features": ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                             'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                             'Albumin_and_Globulin_Ratio'],
                "model": LogisticRegression(max_iter=1000)
            }
        }
        
        config = disease_config[disease_name]
        X = df[config["features"]]
        y = df[config["target"]]
        
        # Preprocessing
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = config["model"]
        model.fit(X_train, y_train)
        
        # Save artifacts
        joblib.dump(model, os.path.join(MODEL_FOLDER, f'{disease_name}_model.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_FOLDER, f'{disease_name}_scaler.pkl'))
        joblib.dump(imputer, os.path.join(MODEL_FOLDER, f'{disease_name}_imputer.pkl'))
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{disease_name.capitalize()} Model - Accuracy: {acc:.4f}")
        
        return True
    
    except Exception as e:
        print(f"Error training {disease_name} model: {str(e)}")
        return False

if __name__ == "__main__":
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
        
    for disease in ['diabetes', 'heart', 'liver']:
        print(f"Training {disease} model...")
        train_disease_model(disease)
    print("All models trained and saved successfully!")