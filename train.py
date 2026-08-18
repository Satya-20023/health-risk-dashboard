"""Train logistic regression on the UCI combined heart-disease dataset."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "heart_disease.csv"
MODEL = ROOT / "models" / "risk_model.joblib"
FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp_s",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "st_slope",
]


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA)
    frame = frame.dropna(subset=FEATURES + ["target"])
    frame["target"] = (frame["target"] > 0).astype(int)
    return frame


def train() -> dict:
    frame = load_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame["target"],
        test_size=0.2,
        random_state=42,
        stratify=frame["target"],
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    proba = pipe.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_rows": int(len(frame)),
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "features": FEATURES, "metrics": metrics}, MODEL)
    print(
        f"Accuracy: {metrics['accuracy']:.1%}  Precision: {metrics['precision']:.1%}  "
        f"ROC-AUC: {metrics['roc_auc']:.3f}"
    )
    return metrics


if __name__ == "__main__":
    train()
