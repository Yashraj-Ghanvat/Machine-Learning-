import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn # Importing sklearn to check the version

# --- Page Configuration ---
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
try:
    # Use a relative path, which is more robust than a hardcoded absolute path.
    # This requires 'used_car_price_model.pkl' to be in the same folder as 'app.py'.
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Used Car Prediction\used_car_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'used_car_price_model.pkl' is in the same directory as this script.")
    st.stop()
except AttributeError as e:
    st.error(f"Version Mismatch Error: {e}")
    st.error(f"This app requires scikit-learn version 1.6.1. Please run 'pip install -r requirements.txt' in your terminal.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Application Title and Description ---
st.title("🚗 Used Car Price Predictor")
st.markdown("""
Welcome to the Used Car Price Predictor! This application uses a machine learning model 
to estimate the price of a used car based on its features. 

Please fill in the details of the car in the sidebar to get a price prediction.
""")
st.markdown("---")

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Car Details")

# Define lists for dropdown menus based on the notebook analysis
fuel_type_options = ['Gasoline', 'Diesel', 'Hybrid', 'E85 Flex Fuel', 'Plug-In Hybrid', 'not supported', '–']
accident_options = ['None reported', 'At least 1 accident or damage reported']
clean_title_options = ['Yes'] # Based on the notebook, after dropna(), only 'Yes' remains.

# Create input fields in the sidebar
with st.sidebar:
    brand = st.text_input("Brand", "Hyundai", help="e.g., Ford, Hyundai, Lexus")
    model_name = st.text_input("Model", "Palisade SEL", help="e.g., Utility Police Interceptor, Palisade SEL")
    model_year = st.number_input("Model Year", min_value=1980, max_value=2025, value=2021, step=1)
    milage = st.number_input("Milage (in miles)", min_value=0, value=34742, step=100)
    fuel_type = st.selectbox("Fuel Type", options=fuel_type_options, index=0)
    engine = st.text_input("Engine", "3.8L V6 24V GDI DOHC", help="e.g., 3.8L V6 24V GDI DOHC, 3.5 Liter DOHC")
    transmission = st.text_input("Transmission", "8-Speed Automatic", help="e.g., 6-Speed A/T, 8-Speed Automatic")
    ext_col = st.text_input("Exterior Color", "Moonlight Cloud", help="e.g., Black, Moonlight Cloud")
    int_col = st.text_input("Interior Color", "Gray", help="e.g., Black, Gray")
    accident = st.selectbox("Accident History", options=accident_options, index=1)
    clean_title = st.selectbox("Clean Title", options=clean_title_options, index=0)

# --- Prediction Logic ---
predict_button = st.sidebar.button("Predict Price", use_container_width=True)

st.subheader("Prediction Result")
if predict_button:
    # Create a DataFrame from the user's input
    input_data = {
        'brand': [brand],
        'model': [model_name],
        'model_year': [model_year],
        'milage': [milage],
        'fuel_type': [fuel_type],
        'engine': [engine],
        'transmission': [transmission],
        'ext_col': [ext_col],
        'int_col': [int_col],
        'accident': [accident],
        'clean_title': [clean_title]
    }
    input_df = pd.DataFrame(input_data)

    # Display the user input for confirmation
    st.write("#### Your Input:")
    st.dataframe(input_df)

    try:
        # Make a prediction
        prediction = model.predict(input_df)
        predicted_price = prediction[0]

        # Display the prediction
        st.markdown("---")
        st.success(f"**Estimated Price:**")
        st.markdown(f"<h2 style='text-align: center; color: #28a745;'>${predicted_price:,.2f}</h2>", unsafe_allow_html=True)
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please enter the car details in the sidebar and click 'Predict Price' to see the result.")

# --- Footer ---
st.markdown("---")
st.markdown(f"Developed for the Used Car Price Prediction project. (scikit-learn version: {sklearn.__version__})")

