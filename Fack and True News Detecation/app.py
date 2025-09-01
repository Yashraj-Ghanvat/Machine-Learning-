# app.py
import streamlit as st
import pandas as pd
import string
import nltk
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ===============================
# 1. Download stopwords (first run)
# ===============================
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ===============================
# 2. Preprocessing function
# ===============================
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ===============================
# 3. Load datasets
# ===============================
@st.cache_data
def load_and_prepare_data():
    true_df = pd.read_csv(r"C:\Users\Admin\Desktop\Yashraj\ML\Fack and True News Detecation\True.csv")
    fake_df = pd.read_csv(r"C:\Users\Admin\Desktop\Yashraj\ML\Fack and True News Detecation\Fake.csv")

    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    return df, X, y, vectorizer

df, X, y, vectorizer = load_and_prepare_data()

# ===============================
# 4. Train Models
# ===============================
@st.cache_resource
def train_models(_X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": LinearSVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ('lr', models["Logistic Regression"]),
            ('nb', models["Naive Bayes"]),
            ('rf', models["Random Forest"]),
            ('svm', models["SVM"])
        ],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)

    return models, ensemble

models, ensemble = train_models(X, y)

# ===============================
# 5. Streamlit UI
# ===============================
st.set_page_config(page_title="📰 Fake News Classifier", layout="wide")

st.title("📰 Fake News Detection System")
st.markdown("""
This app uses **Machine Learning models** (Logistic Regression, Naive Bayes, Random Forest, SVM, and an Ensemble Classifier) 
to predict whether a news article is **Fake** or **Real**.
""")

# Sidebar
st.sidebar.header("🔍 Input News Text")
user_input = st.sidebar.text_area("Paste news content here:", height=200)

model_choice = st.sidebar.selectbox(
    "Select Model for Prediction:",
    ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM", "Ensemble (Voting Classifier)"]
)

if st.sidebar.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text to classify.")
    else:
        # Preprocess and vectorize
        clean_input = clean_text(user_input)
        input_vector = vectorizer.transform([clean_input])

        if model_choice == "Ensemble (Voting Classifier)":
            prediction = ensemble.predict(input_vector)[0]
        else:
            prediction = models[model_choice].predict(input_vector)[0]

        # Display result
        st.subheader("🧾 Prediction Result")
        if prediction == 1:
            st.success("✅ This news is predicted as **REAL** 📰")
        else:
            st.error("🚨 This news is predicted as **FAKE** ❌")

# ===============================
# Footer
# ===============================
st.markdown("""
---
✅ Built with **Streamlit** | Fake News Classifier Project  
""")
