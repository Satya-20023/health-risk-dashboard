"""Train logistic regression and report accuracy / precision."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import main as generate

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "patients.csv"
MODEL = ROOT / "models" / "risk_model.joblib"
FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "cholesterol",
    "glucose",
    "smoking",
    "exercise_hours",
    "family_history",
]


def train() -> None:
    if not DATA.exists():
        generate()
    frame = pd.read_csv(DATA)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame["high_risk"],
        test_size=0.2,
        random_state=42,
        stratify=frame["high_risk"],
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "features": FEATURES}, MODEL)
    print(f"Accuracy: {acc:.1%}  Precision: {prec:.1%}")
    print(f"Saved {MODEL}")


if __name__ == "__main__":
    train()
