from flask import Flask, request, jsonify, render_template
import pickle
from pathlib import Path
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",quiet=True)

app=Flask(__name__)

model_path = Path(__file__).parent / "logistic_regression_tfidf.pkl"

with open(model_path, "rb") as f:
    saved_objects = pickle.load(f)

loaded_LR = saved_objects["model"]
loaded_vectorizer = saved_objects["vectorizer"]

english_stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct = string.punctuation

# preprocessing func
def preprocess_text(text: str) -> str:
    cleaned ="".join([ch for ch in text if ch not in punct])
    words = [w for w in cleaned.split() if w.lower() not in english_stopwords]
    return " ".join(words)

def lemmatize_text(text: str) -> str:
    """Lemmatize words."""
    return " ".join(lemmatizer.lematize(w) for w in text.split())

def full_pipeline(text: str) -> str:
    """Apply full preprocessing pipeline"""
    t = preprocess_text(text)
    t = lemmatize_text(text)
    return t
# prediction function
def predict_text(text: str):
    processed = full_pipeline(text)
    vector = loaded_vectorizer.transform([processed])

    pred = loaded_LR.predict(vector)[0]
    proba = loaded_LR.predict_proba(vector)[0][1]

    return int(pred), float(proba)

# Home page route (UI)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_label = None
    probability = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "")
        if user_text.strip():
            pred, proba = predict_text(user_text)

            if pred == 1:
                prediction_label = "Positive Review ⭐"
            else:
                prediction_label = "Negative Review ❌"

            probability = round(proba, 4)

    return render_template(
        "index.html",
        prediction_label=prediction_label,
        probability=probability,
        user_text=user_text,
    )
# API route (JSON)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    pred, proba = predict_text(text)

    return jsonify({
        "prediction": pred,
        "probability_positive": proba
    })

# Run app locally (remove debug=True for deployment)
if __name__ == "__main__":
    app.run(debug=True)
