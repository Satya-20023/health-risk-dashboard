"""Interactive heart-risk dashboard trained on the UCI combined heart-disease set."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auth import is_signed_in, render_auth
from storage import recent, save_assessment
from train import DATA, FEATURES, MODEL, load_frame, train  # UCI heart-disease loader

st.set_page_config(
    page_title="Pulse — Health Risk Dashboard",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f3ead8", family="Outfit, sans-serif"),
    colorway=["#d4652f", "#e8a06a", "#8fbf9f", "#c9b99a"],
    margin=dict(t=40, r=20, l=20, b=30),
)

CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
ECG = {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy"}
SLOPE = {0: "Downsloping", 1: "Flat", 2: "Upsloping", 3: "Steep"}


@st.cache_resource
def load_bundle():
    if not MODEL.exists():
        train()
    return joblib.load(MODEL)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_frame()


def score_row(model, values: dict) -> tuple[float, str]:
    frame = pd.DataFrame([values])[FEATURES]
    proba = float(model.predict_proba(frame)[0, 1])
    if proba >= 0.7:
        return proba, "High"
    if proba >= 0.4:
        return proba, "Elevated"
    return proba, "Low"


bundle = load_bundle()
model = bundle["model"]
metrics = bundle.get("metrics") or {}
df = load_data()
user = render_auth("Pulse")

st.markdown(
    """
    <style>
      .hero-kicker { letter-spacing:.16em; text-transform:uppercase; color:#e8a06a; font-size:.78rem; }
      .tip { color:#cfc6b6; font-size:.95rem; }
      div[data-testid="stMetric"] { background:#1b1814; border:1px solid rgba(243,234,216,.12); padding:12px 16px; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-kicker">UCI Heart Disease · 1,190 patients</p>', unsafe_allow_html=True)
st.title("Pulse — health risk with smart alerts")
st.caption(
    "Logistic regression on the combined Cleveland / Hungarian / Switzerland / Long Beach / Statlog set. "
    "This is a demo, not a diagnosis."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Patients in training set", f"{metrics.get('n_rows', len(df)):,}")
m2.metric("Accuracy", f"{metrics.get('accuracy', 0):.0%}")
m3.metric("Precision", f"{metrics.get('precision', 0):.0%}")
m4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2f}")

tab_assess, tab_explore, tab_model, tab_log = st.tabs(
    ["Assess me", "Explore the cohort", "How the model thinks", "Recent alerts"]
)

with tab_assess:
    st.subheader("Interactive intake")
    st.markdown(
        '<p class="tip">Move the sliders — the gauge updates live. Submit to save an alert.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        age = st.slider("Age", 28, 80, 54)
        sex = st.radio("Sex", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male", horizontal=True)
        chest = st.select_slider("Chest pain type", options=[1, 2, 3, 4], value=2, format_func=lambda v: CP[v])
        bp = st.slider("Resting blood pressure (mmHg)", 80, 200, 130)
        chol = st.slider("Serum cholesterol (mg/dL)", 80, 400, 240)
        fbs = st.toggle("Fasting blood sugar > 120 mg/dL", value=False)
        ecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda v: ECG[v])
        hr = st.slider("Maximum heart rate achieved", 70, 210, 150)
        angina = st.toggle("Exercise-induced angina", value=False)
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope = st.select_slider("ST slope", options=[0, 1, 2, 3], value=1, format_func=lambda v: SLOPE[v])
    values = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest,
        "resting_bp_s": bp,
        "cholesterol": chol,
        "fasting_blood_sugar": int(fbs),
        "resting_ecg": ecg,
        "max_heart_rate": hr,
        "exercise_angina": int(angina),
        "oldpeak": oldpeak,
        "st_slope": slope,
    }
    proba, label = score_row(model, values)
    color = {"High": "#d4652f", "Elevated": "#e8a06a", "Low": "#8fbf9f"}[label]
    with right:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 42, "color": "#f3ead8"}},
                title={"text": f"{label} risk", "font": {"size": 20, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9a9286"},
                    "bar": {"color": color},
                    "bgcolor": "#1b1814",
                    "steps": [
                        {"range": [0, 40], "color": "#243028"},
                        {"range": [40, 70], "color": "#3a2c1c"},
                        {"range": [70, 100], "color": "#3a1e16"},
                    ],
                    "threshold": {"line": {"color": "#f3ead8", "width": 2}, "value": 50},
                },
            )
        )
        gauge.update_layout(height=320, **PLOT)
        st.plotly_chart(gauge, use_container_width=True)
        if label == "High":
            st.error("Smart alert: high predicted heart-disease risk. Talk with a clinician — this is not a diagnosis.")
            st.write("Typical next steps: resting ECG review, lipid panel, and a cardiology consult if symptoms persist.")
        elif label == "Elevated":
            st.warning("Elevated risk. Lifestyle and follow-up labs are worth discussing with primary care.")
        else:
            st.success("Low predicted risk in this model. Keep monitoring blood pressure, lipids, and activity.")
        if is_signed_in():
            if st.button("Save this assessment", type="primary"):
                save_assessment(values, label, round(proba, 4), username=user)
                st.toast("Saved to your account log.")
        else:
            st.caption("Sign in (sidebar) to save this result. Guest mode is view-only for history.")

with tab_explore:
    st.subheader("The 1,190-patient cohort")
    c1, c2 = st.columns(2)
    with c1:
        sex_pick = st.multiselect("Sex", [0, 1], default=[0, 1], format_func=lambda v: "Female" if v == 0 else "Male")
    with c2:
        age_rng = st.slider("Age range", int(df.age.min()), int(df.age.max()), (40, 70))
    view = df[df.sex.isin(sex_pick) & df.age.between(*age_rng)].copy()
    view["status"] = view.target.map({0: "No disease", 1: "Disease"})
    a, b = st.columns(2)
    fig = px.histogram(view, x="age", color="status", barmode="overlay", nbins=24, title="Age by outcome")
    fig.update_layout(**PLOT)
    a.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(
        view,
        x="max_heart_rate",
        y="oldpeak",
        color="status",
        size="cholesterol",
        hover_data=["age", "resting_bp_s"],
        title="Max heart rate vs ST depression",
        opacity=0.75,
    )
    fig.update_layout(**PLOT)
    b.plotly_chart(fig, use_container_width=True)
    c, d = st.columns(2)
    fig = px.box(view, x="status", y="cholesterol", color="status", title="Cholesterol")
    fig.update_layout(**PLOT, showlegend=False)
    c.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(view, x="chest_pain_type", color="status", barmode="group", title="Chest-pain type")
    fig.update_layout(**PLOT)
    d.plotly_chart(fig, use_container_width=True)
    st.dataframe(view[FEATURES + ["target"]], use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("What the logistic model weighs")
    clf = model.named_steps["clf"]
    weights = pd.DataFrame({"feature": FEATURES, "weight": clf.coef_[0]}).sort_values("weight")
    fig = px.bar(
        weights,
        x="weight",
        y="feature",
        orientation="h",
        title="Coefficient direction (positive → higher risk)",
        color="weight",
        color_continuous_scale=["#8fbf9f", "#f3ead8", "#d4652f"],
    )
    fig.update_layout(**PLOT, height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        **How to read this.** After scaling, a positive weight means that feature increases predicted risk.
        Exercise angina, ST depression, and chest-pain type usually dominate; higher max heart rate often
        points the other way. Source: [UCI Heart Disease](https://doi.org/10.24432/C52P4X).
        """
    )

with tab_log:
    if not is_signed_in():
        st.info("Guest mode: sign in to keep a private log of assessments.")
    else:
        rows = recent(username=user)
        if not rows:
            st.info("No saved assessments yet. Score someone on the first tab.")
        else:
            pretty = pd.DataFrame(rows)
            keep = [c for c in ["created_at", "risk_label", "probability", *FEATURES] if c in pretty.columns]
            st.dataframe(pretty[keep], use_container_width=True, hide_index=True)
