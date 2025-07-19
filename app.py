# app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Text Classification App")

# Input text
user_input = st.text_area("Enter your text here:")

# Predict
if st.button("Predict"):
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)
    st.success(f"Prediction: {prediction[0]}")
