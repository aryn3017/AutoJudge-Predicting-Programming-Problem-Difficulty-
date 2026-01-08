from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from scipy.sparse import hstack

app = Flask(__name__)

# Load models
svm = joblib.load("models/rf_classifier.pkl")
reg = joblib.load("models/rf_regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")

# ---------- Feature utilities ----------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_numeric_features(text):
    features = {}

    # Basic features
    features["char_count"] = len(text)
    features["word_count"] = len(text.split())
    features["line_count"] = text.count("\n") + 1
    features["digit_count"] = len(re.findall(r"\d", text))

    # Symbol features
    symbol_map = {
        "+": "plus", "-": "minus", "*": "mul", "/": "div",
        "%": "mod", "^": "pow", "<=": "le", ">=": "ge", "==": "eq"
    }

    for sym, name in symbol_map.items():
        features[f"sym_{name}"] = text.count(sym)

    # Keyword features (MUST MATCH TRAINING)
    keywords = [
        "dp", "graph", "tree", "greedy", "recursion",
        "union find", "dsu", "bfs", "dfs", "segment tree",
        # add ALL keywords you used in training
    ]

    for kw in keywords:
        features[f"kw_{kw.replace(' ', '_')}"] = text.count(kw)

    # Return in training order
    return np.array([features.get(col, 0) for col in numeric_cols])



# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    score = None

    if request.method == "POST":
        description = request.form["description"]
        input_desc = request.form["input"]
        output_desc = request.form["output"]

        full_text = clean_text(
            description + "\nInput:\n" + input_desc + "\nOutput:\n" + output_desc
        )

        # TF-IDF
        X_text = tfidf.transform([full_text])

        # Numeric features
        X_num = extract_numeric_features(full_text).reshape(1, -1)
        X_num = scaler.transform(X_num)

        # Combine
        X = hstack([X_text, X_num])

        # Predictions
        prediction = svm.predict(X)[0]
        score = round(reg.predict(X.toarray())[0], 2)

    return render_template(
        "index.html",
        prediction=prediction,
        score=score
    )

if __name__ == "__main__":
    app.run(debug=True)
