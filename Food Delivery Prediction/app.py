import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler from the pickle file
@st.cache_resource
def load_model():
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Food Delivery Prediction\food_delivery_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['feature_names']

model, scaler, feature_names = load_model()

# Streamlit app
def main():
    st.title("Food Delivery Time Prediction")
    st.write("This app predicts the delivery time based on order details")
    
    # Input fields
    st.header("Order Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=10.0)
        preparation_time = st.number_input("Preparation Time (min)", min_value=0, max_value=60, value=15)
        courier_exp = st.number_input("Courier Experience (years)", min_value=0, max_value=20, value=3)
    
    with col2:
        weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy", "Windy"])
        traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        vehicle_type = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
    
    # Create input dataframe
    input_data = {
        'Order_ID': [999],  # Dummy value
        'Distance_km': [distance_km],
        'Preparation_Time_min': [preparation_time],
        'Courier_Experience_yrs': [courier_exp],
        'Weather_Clear': [0],
        'Weather_Foggy': [0],
        'Weather_Rainy': [0],
        'Weather_Snowy': [0],
        'Weather_Windy': [0],
        'Traffic_Level_Low': [0],
        'Traffic_Level_Medium': [0],
        'Time_of_Day_Evening': [0],
        'Time_of_Day_Morning': [0],
        'Time_of_Day_Night': [0],
        'Vehicle_Type_Car': [0],
        'Vehicle_Type_Scooter': [0],
        'Total_Estimated_Time': [preparation_time + distance_km],
        'Peak_Hours': [0],
        'Bad_Weather': [0],
        'Weather_Distance_Interaction': [0]
    }
    
    # Set one-hot encoded values
    input_data[f'Weather_{weather}'] = [1]
    input_data[f'Traffic_Level_{traffic_level}'] = [1]
    input_data[f'Time_of_Day_{time_of_day}'] = [1]
    
    if vehicle_type == "Car":
        input_data['Vehicle_Type_Car'] = [1]
    elif vehicle_type == "Scooter":
        input_data['Vehicle_Type_Scooter'] = [1]
    
    # Set Bad_Weather flag
    if weather != "Clear":
        input_data['Bad_Weather'] = [1]
        input_data['Weather_Distance_Interaction'] = [distance_km]
    
    # Set Peak_Hours flag
    if time_of_day in ["Evening", "Night"]:
        input_data['Peak_Hours'] = [1]
    
    # Create DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Ensure all feature columns are present and in correct order
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    # Scale numerical features
    numerical_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Make prediction
    if st.button("Predict Delivery Time"):
        prediction = model.predict(input_df)
        st.success(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")

if __name__ == "__main__":
    main()