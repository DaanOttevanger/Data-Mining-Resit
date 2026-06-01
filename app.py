import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


MODEL_PATH = Path("models/best_fake_review_pipeline.pkl")


def clean_text(value):
    value = str(value).lower()
    value = re.sub(r"<.*?>", " ", value)
    value = re.sub(r"http\S+|www\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s']", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("Fake Review Detector")
st.write("Enter a review text and the model will predict whether it looks fake or real.")

if not MODEL_PATH.exists():
    st.error("Model file not found. First run the modelling notebook and save the best model.")
    st.stop()

model = joblib.load(MODEL_PATH)

review = st.text_area(
    "Review text",
    height=160,
    placeholder="Type or paste a review here..."
)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned_review = clean_text(review)
        prediction = model.predict([cleaned_review])[0]

        label = "fake" if prediction == 1 else "real"

        st.subheader("Prediction")
        st.write(f"The review is predicted as: **{label.upper()}**")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([cleaned_review])[0]
            fake_probability = probability[1]
            st.write(f"Estimated fake probability: **{fake_probability:.2%}**")
        elif hasattr(model, "decision_function"):
            score = model.decision_function([cleaned_review])[0]
            st.write(f"Decision score: **{score:.3f}**")
            st.caption("A higher score means the review is more likely to be fake.")

        st.caption("This is a model prediction and should not be treated as absolute proof.")
