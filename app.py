import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Nairobi Flood Risk Intelligence Hub",
    layout="wide"
)

# Custom CSS for a clean, professional aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .risk-high { background-color: #da3633; color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    .risk-mod { background-color: #d29922; color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    .risk-low { background-color: #238636; color: white; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.title("Nairobi Ward-Level Flood Risk Intelligence Hub")
st.markdown("Advanced machine learning early warning system combining XGBoost predictive modeling with spatial ward and river analytics.")
st.markdown("---")

# 2. Load Model and Live Data Safely
@st.cache_resource
def load_model():
    if os.path.exists("flood_model.joblib"):
        return joblib.load("flood_model.joblib")
    return None

@st.cache_data
def load_forecast_data():
    if os.path.exists("nairobi_flood_risk_output.csv"):
        df = pd.read_csv("nairobi_flood_risk_output.csv")
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    return None

model = load_model()
df_preds = load_forecast_data()

# 3. Sidebar Navigation & Mode Selector
st.sidebar.header("Control Panel")
app_mode = st.sidebar.radio("Select Operation Mode", ["Live 7-Day Forecast", "Custom Date & What-If Simulator"])

if model is None:
    st.error("Model file 'flood_model.joblib' not found. Please train your model first.")
elif df_preds is None and app_mode == "Live 7-Day Forecast":
    st.error("Prediction output file not found. Please run your batch inference script first.")

# ==========================================
# MODE 1: LIVE 7-DAY FORECAST INSPECTION
# ==========================================
if app_mode == "Live 7-Day Forecast" and df_preds is not None:
    st.subheader("Live Open-Meteo 7-Day Forecast Window")
    
    min_date = df_preds['date'].min()
    max_date = df_preds['date'].max()
    
    selected_date = st.sidebar.date_input(
        "Select Forecast Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    filtered_row = df_preds[df_preds['date'] == selected_date]

    if not filtered_row.empty:
        row = filtered_row.iloc[0]
        
        # Display Metrics in Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Rainfall", f"{row['rainfall_mm']:.1f} mm")
        col2.metric("3-Day Cumulative", f"{row['rain_3day_sum']:.1f} mm")
        col3.metric("Flood Probability", f"{row['flood_probability'] * 100:.1f}%")
        col4.metric("Risk Status", row['risk_tier'])

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visual Banner
        if "HIGH" in row['risk_tier']:
            st.markdown('<div class="risk-high">RED ALERT: High flood probability detected. Evacuate low-lying river areas.</div>', unsafe_allow_html=True)
        elif "MODERATE" in row['risk_tier']:
            st.markdown('<div class="risk-mod">WARNING: Moderate risk conditions. Exercise caution near drainage lines.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">NORMAL: Stable weather conditions with low flood risk.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Complete Weekly Risk Overview Table")
    st.dataframe(df_preds[['date', 'rainfall_mm', 'rain_3day_sum', 'flood_probability', 'risk_tier']], use_container_width=True)

# ==========================================
# MODE 2: CUSTOM DATE & WHAT-IF SIMULATOR
# ==========================================
elif app_mode == "Custom Date & What-If Simulator":
    st.subheader("Custom Date & What-If Scenario Simulator")
    st.markdown("Test any date or simulate extreme weather conditions by entering custom rainfall metrics below to instantly calculate model predictions.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        custom_date = st.date_input("Choose Any Target Date")
        custom_rain = st.number_input("Simulated Daily Rainfall (mm)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
        custom_rain_3day = st.number_input("Simulated 3-Day Cumulative Rainfall (mm)", min_value=0.0, max_value=400.0, value=50.0, step=1.0)

    with col_b:
        custom_rain_7day = st.number_input("Simulated 7-Day Cumulative Rainfall (mm)", min_value=0.0, max_value=600.0, value=80.0, step=1.0)
        month = custom_date.month
        custom_anomaly = custom_rain - 5.0

    if st.button("Run Custom Prediction Simulation", type="primary"):
        input_data = pd.DataFrame([{
            'rainfall_mm': custom_rain,
            'rain_3day_sum': custom_rain_3day,
            'rain_7day_sum': custom_rain_7day,
            'rain_anomaly': custom_anomaly,
            'month': month
        }])

        prob = model.predict_proba(input_data)[0][1]
        
        if prob >= 0.70:
            tier = "HIGH RISK (Red Alert)"
            box_class = "risk-high"
        elif prob >= 0.40:
            tier = "MODERATE RISK (Warning)"
            box_class = "risk-mod"
        else:
            tier = "LOW RISK (Normal)"
            box_class = "risk-low"

        st.markdown(f"### Simulation Results for: **{custom_date}**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Rainfall", f"{custom_rain} mm")
        m2.metric("Calculated Flood Probability", f"{prob * 100:.1f}%")
        m3.metric("Assigned Risk Tier", tier)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{box_class}">Simulation Outcome: {tier}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Academic Research Project | Nairobi Ward Flood Risk Engine")