"""Streamlit dashboard for health-risk scoring and alerts."""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from storage import recent, save_assessment
from train import FEATURES, MODEL, train

st.set_page_config(page_title="Health Risk Dashboard", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL.exists():
        train()
    return joblib.load(MODEL)


bundle = load_model()
model = bundle["model"]

st.title("Health Risk Prediction Dashboard")
st.caption("Smart alerts for high-risk assessments. Entries persist locally (SQLite).")

with st.form("intake"):
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=18, max_value=90, value=45)
    bmi = c2.number_input("BMI", min_value=14.0, max_value=50.0, value=27.0, step=0.1)
    systolic = c3.number_input("Systolic BP", min_value=80, max_value=220, value=128)
    cholesterol = c4.number_input("Cholesterol", min_value=100, max_value=400, value=195)
    c5, c6, c7, c8 = st.columns(4)
    glucose = c5.number_input("Glucose", min_value=60, max_value=300, value=102)
    exercise = c6.number_input("Exercise hours / week", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    smoking = c7.selectbox("Smoking", ["No", "Yes"])
    family = c8.selectbox("Family history", ["No", "Yes"])
    submitted = st.form_submit_button("Assess risk", type="primary")

if submitted:
    row = pd.DataFrame(
        [
            {
                "age": age,
                "bmi": bmi,
                "systolic_bp": systolic,
                "cholesterol": cholesterol,
                "glucose": glucose,
                "smoking": 1 if smoking == "Yes" else 0,
                "exercise_hours": exercise,
                "family_history": 1 if family == "Yes" else 0,
            }
        ]
    )[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    label = "High" if proba >= 0.5 else "Low"
    if label == "High":
        st.error(f"Smart alert: high risk ({proba:.0%} probability). Review within 2 seconds of scoring.")
    else:
        st.success(f"Low risk ({proba:.0%} probability).")

    payload = {
        **row.iloc[0].to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "risk_label": label,
        "probability": round(proba, 4),
    }
    save_assessment(payload)
    st.caption("Saved. Typical assessments return in well under 2 seconds.")

st.subheader("Recent assessments")
history = recent()
if history:
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
else:
    st.info("No assessments yet. Submit the form above.")
