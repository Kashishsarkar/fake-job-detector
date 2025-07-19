import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App title
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("Paste a job description below (e.g., title, description, requirements, etc.):")

# Input box
user_input = st.text_area("Job Posting Text")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = user_input.lower()
        cleaned = ''.join([c for c in cleaned if c.isalpha() or c.isspace()])
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("❌ This looks like a FAKE job posting.")
        else:
            st.success("✅ This looks like a REAL job posting.")
