"""Build a synthetic but structured patient dataset for training."""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1200
OUT = Path(__file__).parent / "data" / "patients.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    age = RNG.integers(22, 82, size=N)
    bmi = np.clip(RNG.normal(27.5, 5.4, size=N), 16, 48)
    systolic = np.clip(RNG.normal(128, 18, size=N), 90, 200)
    cholesterol = np.clip(RNG.normal(198, 42, size=N), 120, 340)
    glucose = np.clip(RNG.normal(108, 28, size=N), 70, 260)
    smoking = RNG.binomial(1, 0.28, size=N)
    exercise = np.clip(RNG.normal(3.2, 2.1, size=N), 0, 12)
    family_history = RNG.binomial(1, 0.31, size=N)

    score = (
        0.035 * (age - 40)
        + 0.09 * (bmi - 25)
        + 0.025 * (systolic - 120)
        + 0.012 * (cholesterol - 180)
        + 0.02 * (glucose - 100)
        + 1.15 * smoking
        - 0.18 * exercise
        + 0.95 * family_history
        + RNG.normal(0, 0.55, size=N)
    )
    risk = (score > 1.15).astype(int)

    frame = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi.round(1),
            "systolic_bp": systolic.round(0).astype(int),
            "cholesterol": cholesterol.round(0).astype(int),
            "glucose": glucose.round(0).astype(int),
            "smoking": smoking,
            "exercise_hours": exercise.round(1),
            "family_history": family_history,
            "high_risk": risk,
        }
    )
    frame.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(frame)} rows, high-risk rate={frame.high_risk.mean():.2%})")


if __name__ == "__main__":
    main()
