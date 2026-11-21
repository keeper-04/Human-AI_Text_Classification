"""Train and evaluate the requested text classifiers for challenges A, B, and C."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB


SCRIPT_DIR = Path(__file__).resolve().parent


def find_data_dir() -> Path:
    """Find the repository data folder whether this script is nested or at root."""
    for directory in SCRIPT_DIR.parents:
        candidate = directory / "data"
        if all((candidate / challenge).is_dir() for challenge in ("A", "B", "C")):
            return candidate
    candidate = SCRIPT_DIR / "data"
    if all((candidate / challenge).is_dir() for challenge in ("A", "B", "C")):
        return candidate
    raise FileNotFoundError(
        f"Could not find a data folder containing A, B, and C beside or above {SCRIPT_DIR}"
    )


DATA_DIR = find_data_dir()
# Keep generated reports and predictions beside this script.
OUTPUT_DIR = SCRIPT_DIR


def load_split(challenge: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / challenge / f"{split}.csv"
    frame = pd.read_csv(path)
    required = {"id", "text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    nulls = frame[["id", "text", "label"]].isna().sum()
    if nulls.any():
        raise ValueError(f"{path} has missing required values: {nulls[nulls > 0].to_dict()}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    return frame


def main() -> None:
    metrics_rows = []
    report_sections = []
    mistake_frames = []

    for challenge in ("A", "B", "C"):
        train = load_split(challenge, "train")
        test = load_split(challenge, "test")
        y_train = train["label"]
        y_test = test["label"]
        model_predictions = {}

        models = (
            ("Baseline", CountVectorizer(max_features=10000, min_df=5), MultinomialNB(alpha=1.0)),
            ("Model 2", CountVectorizer(max_features=10000, min_df=5), LogisticRegression()),
            ("Model 3", TfidfVectorizer(max_features=10000, min_df=5), LogisticRegression()),
        )
        for name, vectorizer, classifier in models:
            x_train = vectorizer.fit_transform(train["text"].astype(str))
            x_test = vectorizer.transform(test["text"].astype(str))
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            model_predictions[name] = predictions
            metrics_rows.append(
                {
                    "challenge": challenge,
                    "model": name,
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
                    "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
                    "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
                    "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
                    "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
                    "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
                }
            )
            report_sections.append(
                f"Challenge {challenge} | {name}\n"
                + classification_report(y_test, predictions, zero_division=0)
            )
            wrong = predictions != y_test.to_numpy()
            if wrong.any():
                mistakes = test.loc[wrong, ["id", "text", "label"]].copy()
                mistakes.insert(0, "model", name)
                mistakes.insert(0, "challenge", challenge)
                mistakes["predicted_label"] = predictions[wrong]
                mistake_frames.append(mistakes)
            score = metrics_rows[-1]
            print(f"{challenge} | {name}: accuracy={score['accuracy']:.4f}, macro-F1={score['f1_macro']:.4f}", flush=True)

        # Use the model with the best macro-F1 on the labeled evaluation split for
        # the requested one-file-per-challenge prediction deliverable.
        best_model = max(
            (row for row in metrics_rows if row["challenge"] == challenge),
            key=lambda row: row["f1_macro"],
        )["model"]
        pd.DataFrame({"id": test["id"], "label": model_predictions[best_model]}).to_csv(
            OUTPUT_DIR / f"predictions_{challenge}.csv", index=False
        )
        print(f"{challenge}: predictions saved using {best_model} (highest test macro-F1)")

    results = pd.DataFrame(metrics_rows)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
    all_mistakes = pd.concat(mistake_frames, ignore_index=True) if mistake_frames else pd.DataFrame()
    all_mistakes.to_csv(OUTPUT_DIR / "misclassified_examples.csv", index=False)
    print("\nSaved results.csv, classification_reports.txt, misclassified_examples.csv, and predictions_A/B/C.csv")
    print(results.to_string(index=False))
    if len(all_mistakes):
        print(f"Saved {len(all_mistakes)} incorrect prediction examples; inspect any five in misclassified_examples.csv.")


if __name__ == "__main__":
    main()
