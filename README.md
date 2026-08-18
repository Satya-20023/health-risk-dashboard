# Pulse — Health Risk Dashboard

Interactive Streamlit app trained on the combined [UCI Heart Disease](https://doi.org/10.24432/C52P4X) dataset (1,190 patients).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Tabs: live risk gauge, cohort explorer, model coefficients, saved alerts.
