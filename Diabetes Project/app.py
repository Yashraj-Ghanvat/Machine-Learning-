import streamlit as st
import pickle
import numpy as np

def infer_diabetes(model_path, scaler_path, input_features):
    """
    Perform inferencing on the diabetes dataset.

    Parameters:
    - model_path (str): Path to the saved model pickle file.
    - scaler_path (str): Path to the saved scaler pickle file.
    - input_features (list): List of input features in the order:
      [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
      DiabetesPedigreeFunction, Age]

    Returns:
    - str: "Diabetes Detected" if Outcome is 1, otherwise "No Diabetes".
    """
    try:
        # Load the trained model and scaler
        with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\diabetes_model.pickle', 'rb') as model_file:
            model = pickle.load(model_file)

        with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\scaler (1).pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Ensure the input features are in the correct format
        input_array = np.array([input_features]).reshape(1, -1)

        # Scale the input features
        scaled_input = scaler.transform(input_array)

        # Predict outcome
        prediction = model.predict(scaled_input)

        # Return result
        return "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App
st.title("Diabetes Prediction App")

# Sidebar inputs
st.sidebar.header("Input Features")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.sidebar.number_input("Insulin", min_value=0, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, step=1)

# Model and scaler paths
model_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\diabetes_model.pickle'
scaler_path = r'C:\Users\Admin\Desktop\Yashraj\ML\Diabetes Project\scaler (1).pkl'

# Predict button
if st.button("Predict Diabetes"):
    input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    result = infer_diabetes(model_path, scaler_path, input_features)
    st.write(f"Prediction: **{result}**")