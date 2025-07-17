import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------
# Load Model and Scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Heart project\heart_disease_model.pkl', "rb") as model_file:
        model = pickle.load(model_file)
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Heart project\heart_disease_scaler.pkl', "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# ---------------------------
# Inference Function
# ---------------------------
def infer_heart_disease(model, scaler, input_features):
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    input_df = pd.DataFrame([input_features], columns=feature_names)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return "❤️ Disease Detected" if prediction[0] == 1 else "✅ No Disease"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🫀 Heart Disease Prediction App")
st.markdown("Enter the patient's details to check for potential heart disease risk.")

# File paths (update with your actual file locations)
model_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Heart project\heart_disease_model.pkl'
scaler_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Heart project\heart_disease_scaler.pkl'

# Load model and scaler
model, scaler = load_model_and_scaler(model_path, scaler_path)

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["male", "female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Cholesterol (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Encode sex
sex_encoded = 1 if sex == "male" else 0

# Predict button
if st.button("Predict"):
    input_data = [
        age, sex_encoded, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]
    result = infer_heart_disease(model, scaler, input_data)
    st.success(f"🩺 Prediction: {result}")
