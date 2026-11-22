from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).resolve().parent / "data"
MODEL_NAME = "TF-IDF + Logistic Regression"
METRIC_NAME = "Macro F1"


def read_movies(path):
    plots, genres = [], []
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split(" ::: ", 3)
            if len(fields) != 4 or not fields[2].strip() or not fields[3].strip():
                raise ValueError(f"{path.name}, line {number}: expected ID ::: Title ::: Genre ::: Plot.")
            plots.append(fields[3].strip())
            genres.append(fields[2].strip())
    return plots, genres


def train():
    plots, genres = read_movies(DATA / "train_data.txt")
    test_plots, test_genres = read_movies(DATA / "test_data_solution.txt")
    model = make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000,
                        min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
    )
    model.fit(plots, genres)
    predictions = model.predict(test_plots)
    return model, {
        "accuracy": accuracy_score(test_genres, predictions),
        "f1": f1_score(test_genres, predictions, average="macro", zero_division=0),
        "test_size": len(test_genres),
    }


def predict(model, plot):
    import re

    plot = str(plot).strip()
    if len(re.findall(r"\b[A-Za-z]{2,}\b", plot)) < 8 or len(plot) > 6000:
        raise ValueError("Enter a plot with at least 8 English words and at most 6,000 characters.")
    if model[0].transform([plot]).nnz == 0:
        raise ValueError("The model recognized no terms. Try adding more plot details.")
    scores = model.predict_proba([plot])[0]
    top = scores.argsort()[-3:][::-1]
    return [{"label": str(model.classes_[i]).title(), "score": float(scores[i])} for i in top]
