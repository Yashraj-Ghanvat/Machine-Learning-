import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load Model and Scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler(model_path='model.pkl', scaler_path='scaler.pkl'):
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Titanic Project\model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Titanic Project\scaler (1).pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# ---------------------------
# Preprocess Input
# ---------------------------
def preprocess_input(data_dict, scaler):
    df = pd.DataFrame([data_dict])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    drop_cols = ['Name', 'Ticket', 'Cabin']
    for col in drop_cols:
        df.drop(columns=col, errors='ignore', inplace=True)
    df.fillna(0, inplace=True)
    return scaler.transform(df)

# ---------------------------
# Inference Function
# ---------------------------
def predict_survival(input_data):
    model, scaler = load_model_and_scaler()
    input_scaled = preprocess_input(input_data, scaler)
    prediction = model.predict(input_scaled)
    return "Survived" if prediction[0] == 1 else "Did Not Survive"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict if they would survive the Titanic disaster:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

if st.button("Predict"):
    passenger_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }
    result = predict_survival(passenger_data)
    st.success(f"🧾 Prediction: {result}")
