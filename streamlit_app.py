import streamlit as st
import joblib

import os
from sentiment_model import SentimentAnalyzer

model_path = "nb_model.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.warning("⚠️ Model not found. Training now from sentiment_data.txt...")

    sa = SentimentAnalyzer()
    texts, labels = sa.load_data_from_txt("sentiment_data.txt")
    sa.train(texts, labels)
    sa.save()

    model = sa.model
    vectorizer = sa.vectorizer

    st.success("✅ Model trained and loaded!")


st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter a product review below to classify it as **positive** or **negative**.")

user_input = st.text_area("📝 Enter your review here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        st.write(f"🎯 **Sentiment:** {pred.capitalize()}")
        st.progress(int(prob[1] * 100))
    else:
        st.warning("Please enter a review to analyze.")