🚗 Used Car Price Prediction – Machine Learning Regression Project
📌 Project Overview

This project predicts the price of used cars based on various attributes such as car model, fuel type, mileage, engine capacity, and more.

The workflow includes:

Data loading & cleaning

Feature preprocessing (encoding + scaling)

Model training using multiple regression algorithms

Model evaluation (RMSE, MAE, R²)

Saving the best model for deployment

📂 Dataset

The dataset used_cars.csv contains car details with attributes like:

name → Car model name

year → Year of manufacture

selling_price → Target variable (car price)

km_driven → Kilometers driven

fuel → Fuel type (Petrol, Diesel, CNG, LPG, Electric)

seller_type → Type of seller (Dealer, Individual, Trustmark Dealer)

transmission → Transmission type (Manual, Automatic)

owner → Ownership history

Target variable:

Price (numeric, continuous → regression task)

⚙️ Project Workflow
1️⃣ Data Preprocessing

Handled missing values

Encoded categorical columns (fuel, seller_type, transmission, owner)

Scaled numerical features (year, km_driven, etc.) using StandardScaler

2️⃣ Model Training

Trained the following regression models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

3️⃣ Model Evaluation

Evaluation metrics used:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score (Coefficient of Determination)

4️⃣ Model Saving

Saved the best model and scaler using Pickle for later inference:

import pickle

# Save best model
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

📊 Results
Model	RMSE	MAE	R² Score
Linear Regression	~X.XX	~X.X	~0.70
Random Forest Regressor	~X.XX	~X.X	~0.90
XGBoost Regressor	~X.XX	~X.X	~0.92

(Exact numbers depend on dataset split and training)

🚀 Installation & Usage
🔹 Clone the Repository
git clone https://github.com/your-username/used-car-price-prediction.git
cd used-car-price-prediction

🔹 Install Dependencies
pip install -r requirements.txt

🔹 Run Jupyter Notebook
jupyter notebook "Used_Car_Price_Prediction.ipynb"

🔹 For Inference (Load Model & Predict)
import pickle
import numpy as np

# Load model & scaler
model = pickle.load(open("car_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Example new input [year, km_driven, fuel, seller_type, transmission, owner, ...]
new_car = np.array([[2018, 35000, 1, 0, 1, 0]])  
new_car_scaled = scaler.transform(new_car)
predicted_price = model.predict(new_car_scaled)

print("Predicted Car Price:", predicted_price[0])

📌 Future Improvements

Deploy as a Streamlit app for interactive predictions

Experiment with deep learning models (ANN, LSTM)

Use advanced feature engineering for better accuracy
