# Health Risk Prediction Dashboard with Smart Alerts

Streamlit dashboard that classifies health risk with logistic regression, stores entries locally, and raises alerts for high-risk cases.

This is the public demo from [Satya Narayana Raju Sagi](https://www.linkedin.com/in/raju-49047b258)'s résumé. Production storage can swap to MongoDB Atlas via `MONGO_URI`.

## Features

- Logistic regression risk scores (target: ~87% accuracy / ~92% precision on the bundled test split)
- Real-time form with smart alerts
- SQLite persistence (MongoDB Atlas compatible interface in `storage.py`)
- Feature preprocessing before scoring

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open http://localhost:8501
