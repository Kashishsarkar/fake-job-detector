import streamlit as st
import joblib
import re
from sklearn.preprocessing import normalize

# Load model and vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Title and subtitle
st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("🕵️‍♀️ Fake Job Posting Detector")
st.markdown("Check whether a job posting is real or fake using a trained ML model.")

# User input
user_input = st.text_area("Paste the job description here:")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][prediction]

        if prediction == 1:
            st.error(f"❌ This looks like a FAKE job posting.\n\n🔍 Confidence: {probability:.2%}")
        else:
            st.success(f"✅ This looks like a REAL job posting.\n\n🔍 Confidence: {probability:.2%}")
