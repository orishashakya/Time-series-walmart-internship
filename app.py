# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import pickle
import os
from datetime import datetime, timedelta

# ────────────────────────────────────────────────
# Config & paths
# ────────────────────────────────────────────────

st.set_page_config(page_title="Walmart Sales Forecaster", layout="wide")

DATA_PATH = r"E:\newpy\time-series-walmart\data\processed\walmart_features_imputed.csv"
GLOBAL_MODEL_PATH = "global_prophet_model.pkl"           # you'll save these later
STORE_MODELS_DIR  = "store_prophet_models"               # folder with one file per store

# ────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df

@st.cache_resource
def load_global_model():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        st.error("Global model not found. Please train and save it first.")
        st.stop()
    with open(GLOBAL_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_store_model(store_id):
    path = f"{STORE_MODELS_DIR}/prophet_store_{store_id}.pkl"
    if not os.path.exists(path):
        st.warning(f"Model for store {store_id} not found.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

# ────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────

st.sidebar.title("Walmart Sales Forecaster")

view_mode = st.sidebar.radio(
    "Forecast Level",
    options=["Global (all stores)", "Per Store"],
    index=0
)

weeks_forward = st.sidebar.slider(
    "Forecast horizon (weeks)",
    min_value=4,
    max_value=52,
    value=26,
    step=4
)

if view_mode == "Per Store":
    df = load_data()
    available_stores = sorted(df["Store"].unique())
    selected_store = st.sidebar.selectbox(
        "Select Store",
        options=available_stores,
        index=0
    )

show_uncertainty = st.sidebar.checkbox("Show uncertainty bands", value=True)

# ────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────

st.title("Walmart Weekly Sales Forecasting")
st.markdown("Built with **Facebook Prophet** • Data from Walmart Recruiting competition")

# Load appropriate model
if view_mode == "Global (all stores)":
    model = load_global_model()
    title_prefix = "Global"
    agg_level = "All Stores & Departments"
else:
    model = load_store_model(selected_store)
    if model is None:
        st.stop()
    title_prefix = f"Store {selected_store}"
    agg_level = f"Store {selected_store}"

st.subheader(f"{title_prefix} Forecast")

# ─── Generate forecast ────────────────────────────────────────────────

with st.spinner("Generating forecast..."):
    future = model.make_future_dataframe(periods=weeks_forward, freq='W-FRI')
    
    # If you used extra regressors, you must supply them here too
    # For simplicity we assume only IsHoliday was used and we ffill it
    if hasattr(model, 'extra_regressors') and 'IsHoliday' in model.extra_regressors:
        # This part depends on your historical data structure
        # Simplest version: assume last known value carries forward
        last_is_holiday = model.history['IsHoliday'].iloc[-1]
        future['IsHoliday'] = last_is_holiday   # naive — improve later if needed
    
    forecast = model.predict(future)

# ─── Plot ─────────────────────────────────────────────────────────────

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=model.history['ds'],
    y=model.history['y'],
    mode='lines',
    name='Historical',
    line=dict(color='royalblue')
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast',
    line=dict(color='red')
))

if show_uncertainty:
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.15)',
        name='80% Uncertainty'
    ))

fig.update_layout(
    title=f"{title_prefix} Weekly Sales – {agg_level}",
    xaxis_title="Date",
    yaxis_title="Weekly Sales ($)",
    hovermode="x unified",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ─── Download forecast ────────────────────────────────────────────────

csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
csv_data.columns = ['Date', 'Forecast', 'Lower_80', 'Upper_80']
csv_data['Date'] = csv_data['Date'].dt.strftime('%Y-%m-%d')

st.download_button(
    label="Download forecast as CSV",
    data=csv_data.to_csv(index=False).encode('utf-8'),
    file_name=f"{title_prefix.lower().replace(' ', '_')}_forecast.csv",
    mime="text/csv"
)

# ─── Footer / info ────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Note: This is a demonstration model. For production use, consider retraining periodically "
    "and adding more regressors (temperature, markdowns, fuel price, etc.)."
)