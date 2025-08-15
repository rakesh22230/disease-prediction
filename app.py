# app.py - Main Flask Application
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)
app.config['MODEL_FOLDER'] = 'models/'

# Feature descriptions for UI
FEATURE_INFO = {
    "diabetes": [
        ('Pregnancies', 'Number of times pregnant', 'number'),
        ('Glucose', 'Plasma glucose concentration (mg/dL)', 'number'),
        ('BloodPressure', 'Diastolic blood pressure (mm Hg)', 'number'),
        ('SkinThickness', 'Triceps skinfold thickness (mm)', 'number'),
        ('Insulin', '2-Hour serum insulin (μU/mL)', 'number'),
        ('BMI', 'Body mass index (kg/m²)', 'number'),
        ('DiabetesPedigreeFunction', 'Diabetes pedigree function', 'number'),
        ('Age', 'Age (years)', 'number')
    ],
    "heart": [
        ('age', 'Age (years)', 'number'),
        ('sex', 'Sex (1 = Male, 0 = Female)', 'select', [('0', 'Female'), ('1', 'Male')]),
        ('cp', 'Chest pain type (0-3)', 'select', [('0', 'Typical angina'), ('1', 'Atypical angina'), ('2', 'Non-anginal pain'), ('3', 'Asymptomatic')]),
        ('trestbps', 'Resting blood pressure (mm Hg)', 'number'),
        ('chol', 'Serum cholesterol (mg/dL)', 'number'),
        ('fbs', 'Fasting blood sugar > 120 mg/dL', 'select', [('0', 'False'), ('1', 'True')]),
        ('restecg', 'Resting ECG results (0-2)', 'select', [('0', 'Normal'), ('1', 'ST-T abnormality'), ('2', 'Left ventricular hypertrophy')]),
        ('thalach', 'Maximum heart rate achieved', 'number'),
        ('exang', 'Exercise induced angina', 'select', [('0', 'No'), ('1', 'Yes')]),
        ('oldpeak', 'ST depression induced by exercise', 'number'),
        ('slope', 'Slope of peak exercise ST segment', 'select', [('0', 'Upsloping'), ('1', 'Flat'), ('2', 'Downsloping')]),
        ('ca', 'Number of major vessels (0-3)', 'number'),
        ('thal', 'Thalassemia (1-3)', 'select', [('1', 'Normal'), ('2', 'Fixed defect'), ('3', 'Reversible defect')])
    ],
    "liver": [
        ('Age', 'Age (years)', 'number'),
        ('Gender', 'Gender', 'select', [('0', 'Female'), ('1', 'Male')]),
        ('Total_Bilirubin', 'Total Bilirubin (mg/dL)', 'number'),
        ('Direct_Bilirubin', 'Direct Bilirubin (mg/dL)', 'number'),
        ('Alkaline_Phosphotase', 'Alkaline Phosphatase (IU/L)', 'number'),
        ('Alamine_Aminotransferase', 'ALT (IU/L)', 'number'),
        ('Aspartate_Aminotransferase', 'AST (IU/L)', 'number'),
        ('Total_Protiens', 'Total Proteins (g/dL)', 'number'),
        ('Albumin', 'Albumin (g/dL)', 'number'),
        ('Albumin_and_Globulin_Ratio', 'Albumin/Globulin Ratio', 'number')
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        disease = request.form['disease']
        inputs = []
        
        # Collect form data
        for feature, *_ in FEATURE_INFO[disease]:
            value = request.form.get(feature)
            if value is None:
                continue
            try:
                inputs.append(float(value))
            except ValueError:
                inputs.append(0.0)  # Default value
        
        # Load models and predict
        try:
            model = joblib.load(os.path.join(app.config['MODEL_FOLDER'], f'{disease}_model.pkl'))
            scaler = joblib.load(os.path.join(app.config['MODEL_FOLDER'], f'{disease}_scaler.pkl'))
            imputer = joblib.load(os.path.join(app.config['MODEL_FOLDER'], f'{disease}_imputer.pkl'))
            
            # Preprocess input
            input_data = np.array(inputs).reshape(1, -1)
            input_data = imputer.transform(input_data)
            input_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1] * 100
            
            result = {
                'disease': disease.capitalize(),
                'prediction': 'Positive' if prediction == 1 else 'Negative',
                'confidence': round(proba, 2),
                'features': FEATURE_INFO[disease]
            }
            return render_template('predict.html', result=result)
        
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    # GET request - show form
    disease = request.args.get('disease', 'diabetes')
    return render_template('predict.html', features=FEATURE_INFO[disease], disease=disease)

if __name__ == '__main__':
    if not os.path.exists(app.config['MODEL_FOLDER']):
        os.makedirs(app.config['MODEL_FOLDER'])
    app.run(debug=True)