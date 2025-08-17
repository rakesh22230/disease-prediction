
# 🩺 Disease Prediction Web App

This project is a **Flask-based web app** that can predict if a person is likely to have **Diabetes, Heart Disease, or Liver Disease** using machine learning.


## 📌 Project Link 👉 https://github.com/rakesh22230/disease-prediction

## 📂 Project Structure

disease-prediction/
├── app.py # Main Flask application
├── train_models.py # Script to train machine learning models
├── requirements.txt # Python dependencies
├── Procfile # Deployment file for Heroku/Render
│
├── data/ # Dataset files
│ ├── diabetes.csv
│ ├── heart_disease.csv
│ └── liver_disease.csv
│
├── models/ # Trained ML models
│ ├── diabetes_model.pkl
│ ├── heart_model.pkl
│ └── liver_model.pkl
│
├── templates/ # HTML templates
│ ├── index.html
│ └── predict.html
│
└── static/css/ # Stylesheets
└── style.css

## 🚀 How to Run Locally

1. Install **Python 3.8+** → [Download Python](https://www.python.org/downloads/)

2. Clone this repository:
   ```bash
   git clone https://github.com/rakesh22230/disease-prediction.git
   cd disease-prediction

3.Install required libraries:

pip install -r requirements.txt

4.Run the app:

python app.py 

5. Open your browser → http://127.0.0.1:5000/


💡 How It Works :
1. Choose a disease (Diabetes / Heart / Liver).
2. Enter your health details.
3. Click Predict.
4.Get result → At Risk or Not at Risk.

🛠️ Built With
(i)   Python
(ii)  Flask
(iii) Scikit-learn
(iv)  HTML, CSS

🌍 Deployment
This project is deployed on : 👉 Render
