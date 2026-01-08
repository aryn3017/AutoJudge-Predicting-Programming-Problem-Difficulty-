import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

# Load models
svm = joblib.load("models/svm_classifier.pkl")
reg = joblib.load("models/rf_regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_numeric_features(text):
    features = {}

    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["line_count"] = text.count("\n") + 1
    features["digit_count"] = len(re.findall(r"\d", text))

    symbol_map = {
        "+": "plus", "-": "minus", "*": "mul", "/": "div",
        "%": "mod", "^": "pow", "<=": "le", ">=": "ge", "==": "eq"
    }

    for sym, name in symbol_map.items():
        features[f"sym_{name}"] = text.count(sym)

    for col in numeric_cols:
        features.setdefault(col, 0)

    return np.array([features[col] for col in numeric_cols]).reshape(1, -1)

# UI
st.title("Programming Problem Difficulty Predictor")

description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"):
    full_text = clean_text(description + "\n" + input_desc + "\n" + output_desc)

    X_text = tfidf.transform([full_text])
    X_num = scaler.transform(extract_numeric_features(full_text))
    X = hstack([X_text, X_num])

    difficulty = svm.predict(X)[0]
    score = round(reg.predict(X.toarray())[0], 2)

    st.success(f"Difficulty Class: {difficulty}")
    st.info(f"Difficulty Score: {score}")
