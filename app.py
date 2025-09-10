import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import google.generativeai as genai
import os

st.set_page_config(page_title="💸 Cloud Cost Optimization Tool + AI", layout="wide")
st.title("💸 Cloud Cost Optimization Tool with AI Insights")

@st.cache_data
def load_data(path="sample_data.csv"):
    return pd.read_csv(path)

df = load_data()

st.subheader("Summary")
total_cost = df['cost_usd'].sum() if 'cost_usd' in df.columns else 0
st.metric("Total Monthly Cost", f"${total_cost:,.2f}")

if 'cost_usd' in df.columns:
    iso = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['cost_usd']].fillna(0))
    st.write("Anomalies:", df[df['anomaly']==-1].head())

# Prophet forecasting demo
if 'month' in df.columns:
    monthly = df.groupby('month', as_index=False)['cost_usd'].sum()
    monthly['ds'] = pd.to_datetime(monthly['month']+'-01')
    monthly = monthly.rename(columns={'cost_usd':'y'})

    if len(monthly.dropna()) >= 2:  # ✅ Only run if enough data
        from prophet import Prophet
        m = Prophet(yearly_seasonality=False)
        m.fit(monthly[['ds','y']])
        future = m.make_future_dataframe(periods=6, freq='M')
        fc = m.predict(future)
        st.subheader("📈 Forecast")
        st.line_chart(fc.set_index('ds')['yhat'])
    else:
        st.warning("⚠️ Not enough historical data to generate a forecast.")


api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if api_key:
    genai.configure(api_key=api_key)
    q = st.text_input("Ask Gemini about cost optimization:")
    if st.button("Ask Gemini") and q:
        model = genai.GenerativeModel("gemini-1.5-flash")
        res = model.generate_content(q)
        st.write(res.text)
else:
    st.warning("Set GEMINI_API_KEY in secrets to enable AI assistant.")
