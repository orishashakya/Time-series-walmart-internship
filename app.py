# app.py
import streamlit as st
import plotly.graph_objects as go
import pickle
import os

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────

st.set_page_config(page_title="Walmart Sales Forecaster", layout="wide")

GLOBAL_MODEL_PATH = "global_prophet_model.pkl"
STORE_MODELS_DIR = "store_prophet_models"

# ────────────────────────────────────────────────
# Model Loaders
# ────────────────────────────────────────────────

@st.cache_resource
def load_global_model():
    if not os.path.exists(GLOBAL_MODEL_PATH):
        st.error("Global model file not found.")
        st.stop()
    with open(GLOBAL_MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_store_model(store_id):
    path = f"{STORE_MODELS_DIR}/prophet_store_{store_id}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def get_available_stores():
    if not os.path.exists(STORE_MODELS_DIR):
        return []

    files = os.listdir(STORE_MODELS_DIR)
    stores = []

    for f in files:
        if f.endswith(".pkl"):
            # expects format prophet_store_1.pkl
            store_id = f.replace("prophet_store_", "").replace(".pkl", "")
            stores.append(int(store_id))

    return sorted(stores)

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────

st.sidebar.title("Walmart Sales Forecaster")

view_mode = st.sidebar.radio(
    "Forecast Level",
    ["Global (all stores)", "Per Store"]
)

weeks_forward = st.sidebar.slider(
    "Forecast horizon (weeks)",
    4, 52, 26, 4
)

show_uncertainty = st.sidebar.checkbox(
    "Show uncertainty bands", value=True
)

if view_mode == "Per Store":
    available_stores = get_available_stores()

    if not available_stores:
        st.sidebar.error("No store models found.")
        st.stop()

    selected_store = st.sidebar.selectbox(
        "Select Store",
        available_stores
    )

# ────────────────────────────────────────────────
# Main Content
# ────────────────────────────────────────────────

st.title("Walmart Weekly Sales Forecasting")
st.markdown("Built with **Facebook Prophet**")

# Load model
if view_mode == "Global (all stores)":
    model = load_global_model()
    title_prefix = "Global"
else:
    model = load_store_model(selected_store)
    if model is None:
        st.error("Model for this store not found.")
        st.stop()
    title_prefix = f"Store {selected_store}"

st.subheader(f"{title_prefix} Forecast")

# ────────────────────────────────────────────────
# Forecast Generation
# ────────────────────────────────────────────────

with st.spinner("Generating forecast..."):
    future = model.make_future_dataframe(
        periods=weeks_forward,
        freq="W-FRI"
    )

    forecast = model.predict(future)

# ────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=model.history["ds"],
    y=model.history["y"],
    mode="lines",
    name="Historical"
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast"
))

# Uncertainty
if show_uncertainty:
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        fill="tonexty",
        name="Uncertainty"
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Weekly Sales ($)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# Download Forecast
# ────────────────────────────────────────────────

csv_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
csv_data.columns = ["Date", "Forecast", "Lower", "Upper"]
csv_data["Date"] = csv_data["Date"].dt.strftime("%Y-%m-%d")

st.download_button(
    "Download forecast as CSV",
    csv_data.to_csv(index=False).encode("utf-8"),
    file_name=f"{title_prefix.lower().replace(' ', '_')}_forecast.csv",
    mime="text/csv"
)

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Demo forecasting application using Prophet. "
    "For production systems, consider retraining periodically and adding more regressors."
)
st.write("Extra regressors:", model.extra_regressors)
