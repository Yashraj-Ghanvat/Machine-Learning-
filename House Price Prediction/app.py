#Import required libraries 
import streamlit as st 
import numpy as np  
import pickle 
#  Function to make prediction using the saved model and scaler 
def predict_house_price(input_features): 
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\House Price Prediction\best_model.pkl', 'rb') as model_file:  # Load trained model 
        model = pickle.load(model_file) 
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\House Price Prediction\scaler (1).pkl', 'rb') as scaler_file:  # Load feature scaler 
        scaler = pickle.load(scaler_file) 
 
    input_array = np.array(input_features).reshape(1, -1)  # Convert input to 2D array 
 
    scaled_input = scaler.transform(input_array)  # Scale features using same scaler used in training 
    prediction = model.predict(scaled_input)  # Predict using model 
    return prediction[0] 
#  Streamlit App Title 
st.title('California Housing Price Prediction') 
 
# Sidebar for User Input 
st.sidebar.header('Input Features') 
 
# Collecting input values from the user via sidebar 
input_features = [ 
    st.sidebar.number_input('Median Income', min_value=0.0, step=0.1), 
    st.sidebar.number_input('House Age', min_value=0.0, step=0.1), 
    st.sidebar.number_input('Average Rooms', min_value=0.0, step=0.1), 
    st.sidebar.number_input('Average Bedrooms', min_value=0.0, step=0.1), 
    st.sidebar.number_input('Population', min_value=0.0, step=1.0), 
    st.sidebar.number_input('Average Occupancy', min_value=0.0, step=0.1), 
    st.sidebar.number_input('Latitude', min_value=32.0, max_value=42.0, step=0.01), 
    st.sidebar.number_input('Longitude', min_value=-124.0, max_value=-114.0, step=0.01) 
] 
# Trigger prediction when button is clicked 
if st.button('Predict'): 
    result = predict_house_price(input_features) 
    st.write(f'Predicted House Price: ${result * 100000:.2f}') 