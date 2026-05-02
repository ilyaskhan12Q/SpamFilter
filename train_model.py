from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "spam.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "metrics.pkl"


@dataclass(frozen=True)
class TrainingMetrics:
    dataset_rows: int
    ham_rows: int
    spam_rows: int
    test_accuracy: float
    cross_val_accuracy_mean: float
    cross_val_accuracy_std: float
    classification_report: str


def load_dataset(path: Path = DATASET_PATH) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add the UCI SMS Spam Collection as spam.csv "
            "with columns: label,message."
        )

    labels: list[str] = []
    messages: list[str] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"label", "message"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("spam.csv must include columns named 'label' and 'message'.")

        for row in reader:
            label = row["label"].strip().lower()
            message = row["message"].strip()
            if label in {"ham", "spam"} and message:
                labels.append(label)
                messages.append(message)

    if not messages:
        raise ValueError("No valid rows found in spam.csv.")
    return messages, labels


def train_and_save() -> TrainingMetrics:
    messages, labels = load_dataset()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=False,
    )
    features = vectorizer.fit_transform(messages)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = MultinomialNB(alpha=0.2)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    cv_model = MultinomialNB(alpha=0.2)
    cv_scores = cross_val_score(cv_model, features, labels, cv=5, scoring="accuracy")

    # Save a final model trained on all available examples for best app inference.
    final_model = MultinomialNB(alpha=0.2)
    final_model.fit(features, labels)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics = TrainingMetrics(
        dataset_rows=len(labels),
        ham_rows=labels.count("ham"),
        spam_rows=labels.count("spam"),
        test_accuracy=float(accuracy),
        cross_val_accuracy_mean=float(cv_scores.mean()),
        cross_val_accuracy_std=float(cv_scores.std()),
        classification_report=classification_report(y_test, predictions, digits=4),
    )
    joblib.dump(asdict(metrics), METADATA_PATH)
    return metrics


def main() -> None:
    metrics = train_and_save()
    print("SpamFilter Pro training complete")
    print(f"Rows: {metrics.dataset_rows:,}")
    print(f"Ham / Spam: {metrics.ham_rows:,} / {metrics.spam_rows:,}")
    print(f"Test accuracy: {metrics.test_accuracy:.4%}")
    print(
        "5-fold CV accuracy: "
        f"{metrics.cross_val_accuracy_mean:.4%} +/- {metrics.cross_val_accuracy_std:.4%}"
    )
    print(metrics.classification_report)


if __name__ == "__main__":
    main()
