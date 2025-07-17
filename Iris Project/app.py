import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load model and scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    with open("log_reg_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# ---------------------------
# Preprocess input
# ---------------------------
def preprocess_input(input_data, scaler):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    return input_scaled
# ---------------------------
# Predict species
# ---------------------------
def predict_iris_species(input_data):
    model, scaler = load_model_and_scaler()
    input_scaled = preprocess_input(input_data, scaler)
    prediction = model.predict(input_scaled)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return species_map[prediction[0]]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌸 Iris Species Predictor")

st.write("Enter the flower measurements below to predict the species:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict Species"):
    new_data = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width
    }
    species = predict_iris_species(new_data)
    st.success(f"The predicted Iris species is: **{species}**")
