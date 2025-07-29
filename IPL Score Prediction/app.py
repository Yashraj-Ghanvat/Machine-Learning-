# Import necessary libraries 
import streamlit as st  # For creating the web app interface 
import numpy as np      # For numerical operations 
import pickle           # For loading the trained machine learning model 

# Load the trained Linear Regression model using pickle 
with open('linear_regressor.pkl', 'rb') as file: 
    linear_regressor = pickle.load(file)  # Loading the serialized model 

# Define the list of consistent IPL teams for dropdown options and encoding 
teams = [ 
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders', 
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad' 
] 

# Function to prepare input and predict the final score 
def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5): 
    temp_array = []  # Temporary array to hold features 
    # One-hot encoding for batting team (8 values, only one will be 1) 
    for team in teams: 
        temp_array.append(1 if batting_team == team else 0) 
    # One-hot encoding for bowling team (8 values, only one will be 1) 
    for team in teams: 
        temp_array.append(1 if bowling_team == team else 0) 
    # Append match statistics (numerical values) 
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]) 
    # Convert the list into a 2D NumPy array as expected by the model 
    temp_array = np.array([temp_array]) 
    # Predict the final score using the trained model 
    predicted_score = int(linear_regressor.predict(temp_array)[0]) 
    return predicted_score 

# ---------------------- Streamlit UI ---------------------- 
# Set title of the web app 
st.title('IPL First Innings Score Predictor')
# Dropdown to select Batting Team 
batting_team = st.selectbox('Select Batting Team', teams) 
# Dropdown to select Bowling Team 
bowling_team = st.selectbox('Select Bowling Team', teams) 
# Input for number of overs completed (should be at least 5 overs) 
overs = st.number_input('Overs Completed (≥ 5)', min_value=5.0, max_value=20.0, step=0.1) 
# Input for current total runs scored 
runs = st.number_input('Current Runs', min_value=0, max_value=300, step=1) 
# Input for number of wickets fallen so far 
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1) 
# Input for number of runs scored in the last 5 overs 
runs_in_prev_5 = st.number_input('Runs in Last 5 Overs', min_value=0, max_value=100, step=1) 
# Input for number of wickets lost in the last 5 overs 
wickets_in_prev_5 = st.number_input('Wickets in Last 5 Overs', min_value=0, max_value=10, step=1) 
# Button to make prediction 
if st.button('Predict Score'): 
    final_score = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5) 
    # Display a predicted range to show some margin of error 
    st.success(f'Predicted Score Range: {final_score - 10} to {final_score + 5}')
