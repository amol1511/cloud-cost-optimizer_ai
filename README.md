# Cloud Cost Optimization Tool with AI

This Streamlit app analyzes cloud cost CSV data, detects anomalies, forecasts spend, and integrates with Gemini AI to provide optimization insights.

## Features
- Upload or use sample cloud cost data.
- Summary of costs by service & resource.
- Anomaly detection (IsolationForest).
- Forecasting with Prophet.
- AI-powered recommendations using Google Gemini.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Cloud)
1. Push this folder to GitHub.
2. Create a new app at https://share.streamlit.io/
3. Set `GEMINI_API_KEY` in Secrets.
