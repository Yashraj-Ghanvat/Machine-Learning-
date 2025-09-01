import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------
# Load Saved Models
# ---------------------------
@st.cache_resource
def load_models(path=r'C:\Users\Admin\Desktop\Yashraj\ML\Intrusion Detecation\intrusion_detection_models.pkl'):
    """
    Loads the trained models dictionary from pickle file.
    """
    try:
        with open(path, 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please check the path.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# ---------------------------
# Prediction Function
# ---------------------------
def predict_intrusion(model, input_data, num_features=10):
    """
    Predicts intrusion ('normal' or 'anomaly').
    """
    # Ensure input is numpy array
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    # Validate input feature count
    if input_data.shape[1] != num_features:
        raise ValueError(f"Input data should have {num_features} features, but got {input_data.shape[1]}")

    predictions = model.predict(input_data)

    # If model outputs string labels directly
    if isinstance(predictions[0], str):
        return predictions.tolist()

    # Otherwise map numeric outputs
    labels = ['normal', 'anomaly']
    return [labels[int(p)] for p in predictions]

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="🛡️ Intrusion Detection System", layout="wide")
st.title("🛡️ Network Intrusion Detection System")
st.markdown("This app uses a **Decision Tree Classifier** trained on network traffic data to detect whether a connection is **Normal** or an **Anomaly (Intrusion)**.")

# Load models
models = load_models()
if models:
    # Select model
    model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
    model = models.get(model_choice)

    st.sidebar.header("🔧 Input Features")
    st.sidebar.markdown("Enter the 10 selected feature values:")

    # Create input fields
    feature_inputs = []
    for i in range(10):
        val = st.sidebar.number_input(f"Feature {i+1}", value=0.0, format="%.3f")
        feature_inputs.append(val)

    # Prediction
    if st.sidebar.button("🔍 Predict Intrusion"):
        try:
            result = predict_intrusion(model, feature_inputs, num_features=10)
            st.subheader("✅ Prediction Result")
            if result[0] == "anomaly":
                st.error("🚨 Intrusion Detected (Anomaly)", icon="🔥")
            else:
                st.success("✅ Normal Traffic", icon="✔️")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.warning("No models loaded. Please upload a valid `.pkl` file.")
