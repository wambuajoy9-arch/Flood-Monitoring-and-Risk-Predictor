import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import pydeck as pdk

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="Nairobi Flood Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional Custom CSS (Clean dark theme, glassmorphism cards, no emojis)
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    .stSidebar { background-color: #111827; }
    
    /* Custom Metric Cards */
    .metric-container {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk Alert Banners */
    .alert-high {
        background-color: #991b1b;
        color: #ffffff;
        padding: 16px;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .alert-mod {
        background-color: #b45309;
        color: #ffffff;
        padding: 16px;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .alert-low {
        background-color: #065f46;
        color: #ffffff;
        padding: 16px;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Title Section
st.title("Nairobi Ward-Level Flood Risk Intelligence Hub")
st.markdown("Operational decision-support dashboard driven by XGBoost machine learning, meteorological live feeds, and GIS ward boundary mappings.")
st.markdown("---")

# 4. Safe Resource & Data Loading
@st.cache_resource
def load_ml_model():
    if os.path.exists("flood_model.joblib"):
        return joblib.load("flood_model.joblib")
    return None

@st.cache_data
def load_predictions():
    if os.path.exists("nairobi_flood_risk_output.csv"):
        df = pd.read_csv("nairobi_flood_risk_output.csv")
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    return None

model = load_ml_model()
df_preds = load_predictions()

# 5. Sidebar Control Panel with Weather-Themed Terminology
st.sidebar.header("Operational Controls")
st.sidebar.markdown("Configure analytical views and meteorological simulation parameters.")

app_mode = st.sidebar.selectbox(
    "Select Operating Module",
    ["Meteorological Forecast Feed", "Hydrological Simulation Lab", "Spatial Ward Risk Mapping"]
)

if model is None:
    st.error("System Error: Model artifact 'flood_model.joblib' not detected. Please verify your working directory.")


# MODULE 1: METEOROLOGICAL FORECAST FEED (Rolling 7-Day Window)

if app_mode == "Meteorological Forecast Feed" and df_preds is not None:
    st.subheader("Open-Meteo 7-Day Atmospheric & Precipitation Feed")
    
    min_date = df_preds['date'].min()
    max_date = df_preds['date'].max()
    
    target_date = st.sidebar.date_input(
        "Inspection Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    row_data = df_preds[df_preds['date'] == target_date]

    if not row_data.empty:
        record = row_data.iloc[0]
        
        # Metric columns layout
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precipitation Rate", f"{record['rainfall_mm']:.1f} mm")
        col2.metric("3-Day Cumulative", f"{record['rain_3day_sum']:.1f} mm")
        col3.metric("Model Probability", f"{record['flood_probability'] * 100:.1f}%")
        col4.metric("Risk Classification", record['risk_tier'])

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Status Alert Banner
        if "HIGH" in record['risk_tier']:
            st.markdown('<div class="alert-high">CRITICAL ALERT: High flood likelihood projected. Immediate drainage monitoring required in vulnerable sectors.</div>', unsafe_allow_html=True)
        elif "MODERATE" in record['risk_tier']:
            st.markdown('<div class="alert-mod">ADVISORY WARNING: Elevated moisture accumulation. Exercise vigilance near river basins.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-low">SYSTEM NORMAL: Atmospheric conditions stable. Nominal risk levels recorded.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Weekly Forecast Ledger")
    st.dataframe(df_preds[['date', 'rainfall_mm', 'rain_3day_sum', 'flood_probability', 'risk_tier']], use_container_width=True)


# MODULE 2: HYDROLOGICAL SIMULATION LAB (What-If Custom Predictor)

elif app_mode == "Hydrological Simulation Lab":
    st.subheader("Custom What-If Precipitation Scenario Lab")
    st.markdown("Bypass temporal API restrictions to simulate custom rainfall accumulation events across any timeline.")

    col_left, col_right = st.columns(2)
    
    with col_left:
        sim_date = st.date_input("Simulation Target Date")
        sim_rain = st.number_input("Daily Precipitation (mm)", min_value=0.0, max_value=250.0, value=30.0, step=1.0)
        sim_3day = st.number_input("3-Day Cumulative Sum (mm)", min_value=0.0, max_value=500.0, value=75.0, step=1.0)

    with col_right:
        sim_7day = st.number_input("7-Day Cumulative Sum (mm)", min_value=0.0, max_value=800.0, value=120.0, step=1.0)
        month_val = sim_date.month
        anomaly_val = sim_rain - 10.0

    if st.button("Execute Model Simulation", type="primary"):
        input_vector = pd.DataFrame([{
            'rainfall_mm': sim_rain,
            'rain_3day_sum': sim_3day,
            'rain_7day_sum': sim_7day,
            'rain_anomaly': anomaly_val,
            'month': month_val
        }])

        probability = model.predict_proba(input_vector)[0][1]
        
        if probability >= 0.70:
            classification = "HIGH RISK (Red Alert)"
            box_style = "alert-high"
        elif probability >= 0.40:
            classification = "MODERATE RISK (Warning)"
            box_style = "alert-mod"
        else:
            classification = "LOW RISK (Normal)"
            box_style = "alert-low"

        st.markdown(f"### Simulation Outcome for: **{sim_date}**")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Input Rainfall", f"{sim_rain} mm")
        m_col2.metric("Calculated Risk Score", f"{probability * 100:.1f}%")
        m_col3.metric("Assigned Tier", classification)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{box_style}">Simulation Result: {classification}</div>', unsafe_allow_html=True)


# MODULE 3: SPATIAL WARD RISK MAPPING

elif app_mode == "Spatial Ward Risk Mapping":
    st.subheader("Nairobi Ward-Level GIS Infrastructure View")
    st.markdown("Geospatial visualization framework rendering Nairobi's 85 administrative wards and river networks.")

    geojson_path = "nairobi_wards.geojson"
    
    if os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
            
        # Center coordinates for Nairobi
        nairobi_latitude = -1.2921
        nairobi_longitude = 36.8219
        
        # Pydeck map layer configuration
        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            stroked=True,
            filled=True,
            get_fill_color=[30, 58, 138, 140],
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=nairobi_latitude,
            longitude=nairobi_longitude,
            zoom=10.5,
            pitch=30
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"html": "<b>Ward Boundary Analysis</b>"}
        )

        st.pydeck_chart(r)
    else:
        st.warning("Spatial layer file `nairobi_wards.geojson` not located in the project directory. Map rendering unavailable.")

# Footer Section
st.markdown("---")
st.markdown("Academic Research Initiative | Advanced Spatial Flood Risk Prediction Engine")