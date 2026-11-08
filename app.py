import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import pydeck as pdk

# 1. Page Configuration
st.set_page_config(
    page_title="Nairobi Weather & Flood Watch",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Weather App CSS Theme
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #090d16; border-right: 1px solid #1e293b; }
    .weather-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .weather-safe {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.2rem; font-weight: 600;
    }
    .weather-warning {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.2rem; font-weight: 600;
    }
    .weather-danger {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white; padding: 20px; border-radius: 15px; text-align: center; font-size: 1.2rem; font-weight: 600;
    }
    h1, h2, h3 { color: #f8fafc; }
    </style>
""", unsafe_allow_html=True)

# 3. Header Title
st.title("Nairobi Atmospheric & Flood Watch")
st.markdown("Live meteorological conditions and ward-level flood risk intelligence.")
st.markdown("---")

# 4. Load Model and Data Safely
@st.cache_resource
def load_model():
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

model = load_model()
df_preds = load_predictions()

# 5. Sidebar Navigation Controls
st.sidebar.markdown("### Weather Station")
app_mode = st.sidebar.radio(
    "",
    ["Live 7-Day Forecast", "Scenario Simulator", "Ward GIS Map"]
)

if model is None:
    st.error("Model file 'flood_model.joblib' missing.")


# MODE 1: LIVE FORECAST WEATHER CARDS

if app_mode == "Live 7-Day Forecast" and df_preds is not None:
    st.subheader("7-Day Forecast Outlook")
    
    min_date = df_preds['date'].min()
    max_date = df_preds['date'].max()
    
    selected_date = st.sidebar.date_input(
        "Choose Forecast Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    row = df_preds[df_preds['date'] == selected_date]

    if not row.empty:
        data = row.iloc[0]
        prob = data['flood_probability']
        
        if prob >= 0.70:
            status_class = "weather-danger"
            condition_text = "Severe Storm & Flood Warning"
            face_icon = "(✖╭╮✖) High Risk Alert"
        elif prob >= 0.40:
            status_class = "weather-warning"
            condition_text = "Heavy Showers / Elevated Risk"
            face_icon = "(⊙_⊙) Moderate Advisory"
        else:
            status_class = "weather-safe"
            condition_text = "Clear Skies / Stable Conditions"
            face_icon = "(^◡^) Safe & Normal"

        st.markdown(f"""
            <div class="weather-card">
                <h2>{selected_date.strftime('%A, %B %d, %Y')}</h2>
                <h1 style="font-size: 3rem; margin: 10px 0;">{face_icon}</h1>
                <p style="font-size: 1.3rem; color: #cbd5e1;">Condition: <b>{condition_text}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rainfall Amount", f"{data['rainfall_mm']:.1f} mm")
        c2.metric("3-Day Accumulation", f"{data['rain_3day_sum']:.1f} mm")
        c3.metric("Flood Probability", f"{prob * 100:.1f}%")
        c4.metric("Risk Status", data['risk_tier'])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{status_class}">Operational Status: {data["risk_tier"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Weekly Timeline Ledger")
    st.dataframe(df_preds[['date', 'rainfall_mm', 'rain_3day_sum', 'flood_probability', 'risk_tier']], use_container_width=True)


# MODE 2: SCENARIO SIMULATOR

elif app_mode == "Scenario Simulator":
    st.subheader("Custom Weather & Storm Simulator")
    st.markdown("Test custom rainfall events to see how the system reacts under different weather scenarios.")

    col1, col2 = st.columns(2)
    with col1:
        sim_date = st.date_input("Simulation Date")
        rain_in = st.number_input("Daily Rainfall (mm)", 0.0, 200.0, 30.0)
        rain_3 = st.number_input("3-Day Total (mm)", 0.0, 400.0, 75.0)
    with col2:
        rain_7 = st.number_input("7-Day Total (mm)", 0.0, 600.0, 120.0)
        sim_month = sim_date.month

    if st.button("Run Simulation", type="primary"):
        input_df = pd.DataFrame([{
            'rainfall_mm': rain_in,
            'rain_3day_sum': rain_3,
            'rain_7day_sum': rain_7,
            'rain_anomaly': rain_in - 5.0,
            'month': sim_month
        }])
        
        prob = model.predict_proba(input_df)[0][1]
        
        if prob >= 0.70:
            box_cls = "weather-danger"
            sim_face = "(✖╭╮✖) Severe Storm Alert"
        elif prob >= 0.40:
            box_cls = "weather-warning"
            sim_face = "(⊙_⊙) Moderate Weather Advisory"
        else:
            box_cls = "weather-safe"
            sim_face = "(^◡^) Clear & Stable"

        st.markdown(f"""
            <div class="weather-card">
                <h2>Simulation for {sim_date}</h2>
                <h1 style="font-size: 3rem; margin: 10px 0;">{sim_face}</h1>
                <p style="font-size: 1.2rem;">Calculated Flood Probability: <b>{prob * 100:.1f}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{box_cls}">Simulation Outcome Score: {prob * 100:.1f}% Risk Probability</div>', unsafe_allow_html=True)


# MODE 3: WARD GIS MAP (Self-Contained & Error-Proof)

elif app_mode == "Ward GIS Map":
    st.subheader("Nairobi Regional Risk Radar & Ward Nodes")
    st.markdown("Interactive geospatial monitoring grid across Nairobi sub-counties.")

    geojson_path = "nairobi_wards.geojson"
    
    if os.path.exists(geojson_path):
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
        layer = pdk.Layer(
            "GeoJsonLayer", geojson_data, stroked=True, filled=True,
            get_fill_color=[56, 189, 248, 100], get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1, pickable=True
        )
    else:
        # Fallback: Generate built-in Nairobi regional weather nodes if GeoJSON is missing
        fallback_data = pd.DataFrame([
            {"name": "Westlands / CBD", "lat": -1.2863, "lon": 36.8172, "risk": "Normal"},
            {"name": "Kibera / Langata", "lat": -1.3138, "lon": 36.7845, "risk": "Moderate"},
            {"name": "Mathare / Eastleigh", "lat": -1.2686, "lon": 36.8536, "risk": "Normal"},
            {"name": "Embakasi", "lat": -1.3032, "lon": 36.9081, "risk": "Normal"},
            {"name": "Kasarani", "lat": -1.2215, "lon": 36.8906, "risk": "Normal"}
        ])
        layer = pdk.Layer(
            "ScatterplotLayer",
            fallback_data,
            get_position='[lon, lat]',
            get_radius=1200,
            get_fill_color=[56, 189, 248, 180],
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            pickable=True
        )

    view_state = pdk.ViewState(latitude=-1.2921, longitude=36.8219, zoom=11, pitch=25)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"html": "<b>Nairobi Monitoring Node</b>"})
    st.pydeck_chart(deck, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Nairobi Meteorological & Flood Watch Hub")