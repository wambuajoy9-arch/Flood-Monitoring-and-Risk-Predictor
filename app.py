import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pydeck as pdk

# 1. Page Configuration
st.set_page_config(
    page_title="Nairobi Weather & Flood Watch",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Weather App Card CSS 
st.markdown("""
    <style>
    .stApp { background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Main Weather Widget Card */
    .weather-widget {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 40px;
        border-radius: 30px;
        box-shadow: 0 20px 30px -10px rgba(29, 78, 216, 0.4);
        position: relative;
        overflow: hidden;
    }
    .weather-widget-warning {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        box-shadow: 0 20px 30px -10px rgba(180, 83, 9, 0.4);
    }
    .weather-widget-danger {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        box-shadow: 0 20px 30px -10px rgba(220, 38, 38, 0.4);
    }

    /* Bottom Forecast Row Pills */
    .forecast-row {
        display: flex;
        justify-content: space-between;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .forecast-pill-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 12px 18px;
        border-radius: 16px;
        text-align: center;
        flex: 1;
        margin: 0 6px;
    }

    /* Stat Pills */
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 18px;
        text-align: center;
    }

    h1, h2, h3 { color: #f8fafc; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# 3. Header Title
st.title("Nairobi Atmospheric & Flood Watch")
st.markdown("Consumer-grade meteorological intelligence and watershed risk tracking.")
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

# 5. Sidebar Weather Station Controls
st.sidebar.markdown("### Weather Station")
app_mode = st.sidebar.radio(
    "",
    ["Live 7-Day Forecast", "Scenario Simulator", "Ward GIS Map"]
)

if model is None:
    st.error("System Error: Model artifact 'flood_model.joblib' missing.")


# MODE 1: WEATHER APP CARD INTERFACE

if app_mode == "Live 7-Day Forecast" and df_preds is not None:
    st.subheader("Live Atmospheric Outlook")
    
    min_date = df_preds['date'].min()
    max_date = df_preds['date'].max()
    
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    row = df_preds[df_preds['date'] == selected_date]

    if not row.empty:
        data = row.iloc[0]
        prob = data['flood_probability'] * 100
        
        # Determine widget background theme and weather visuals
        if prob >= 60:
            widget_class = "weather-widget weather-widget-danger"
            condition_label = "Severe Storm & High Flood Risk"
            weather_icon = "⛈️"
        elif prob >= 30:
            widget_class = "weather-widget weather-widget-warning"
            condition_label = "Heavy Showers & Moderate Advisory"
            weather_icon = "🌦️"
        else:
            widget_class = "weather-widget"
            condition_label = "Clear Skies & Stable Conditions"
            weather_icon = "⛅"

        # Render Main Weather App Card Layout
        st.markdown(f"""
            <div class="{widget_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">{selected_date.strftime('%A, %B %d, %Y')}</p>
                        <h1 style="font-size: 4.5rem; margin: 5px 0; font-weight: 800;">{data['rainfall_mm']:.1f}<span style="font-size: 2rem;"> mm</span></h1>
                        <p style="font-size: 1.4rem; font-weight: 600; margin-top: 5px;">{condition_label}</p>
                        <p style="font-size: 1rem; background: rgba(0,0,0,0.2); display: inline-block; padding: 4px 14px; border-radius: 12px; margin-top: 8px;">
                            Flood Risk Probability: <b>{prob:.1f}% ({data['risk_tier']})</b>
                        </p>
                    </div>
                    <div style="text-align: right; font-size: 7rem; line-height: 1;">
                        {weather_icon}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stat Breakdown Cards underneath
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><h4>Daily Rainfall</h4><p style="font-size: 1.5rem; font-weight: bold; color: #38bdf8;">{data["rainfall_mm"]:.1f} mm</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><h4>3-Day Accumulation</h4><p style="font-size: 1.5rem; font-weight: bold; color: #818cf8;">{data["rain_3day_sum"]:.1f} mm</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><h4>Model Confidence</h4><p style="font-size: 1.5rem; font-weight: bold; color: #34d399;">{prob:.1f}%</p></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><h4>Status Tier</h4><p style="font-size: 1.1rem; font-weight: bold; margin-top: 5px; color: {"#ef4444" if prob>=60 else "#f59e0b" if prob>=30 else "#10b981"};">{data["risk_tier"]}</p></div>', unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Weekly Forecast Overview")
    
    # Render modern weekly pills row
    cols = st.columns(len(df_preds))
    for idx, (_, d_row) in enumerate(df_preds.iterrows()):
        with cols[idx]:
            p_val = d_row['flood_probability'] * 100
            day_name = pd.to_datetime(d_row['date']).strftime('%a')
            icon_symbol = "⛈️" if p_val >= 60 else "🌦️" if p_val >= 30 else "⛅"
            st.markdown(f"""
                <div class="forecast-pill-box">
                    <p style="font-weight: bold; margin-bottom: 2px; color: #e2e8f0;">{day_name}</p>
                    <p style="font-size: 1.5rem; margin: 5px 0;">{icon_symbol}</p>
                    <p style="font-size: 0.9rem; font-weight: bold; color: #ffffff;">{d_row['rainfall_mm']:.1f}mm</p>
                    <p style="font-size: 0.75rem; color: {'#fca5a5' if p_val >= 60 else '#fde047' if p_val >= 30 else '#6ee7b7'}; font-weight: 600;">{p_val:.0f}% Risk</p>
                </div>
            """, unsafe_allow_html=True)


# MODE 2: SCENARIO SIMULATOR

elif app_mode == "Scenario Simulator":
    st.subheader("Custom Storm Simulator")
    st.markdown("Test custom rainfall parameters to evaluate simulated hydrological response.")

    col1, col2 = st.columns(2)
    with col1:
        sim_date = st.date_input("Simulation Date")
        rain_in = st.number_input("Daily Rainfall (mm)", 0.0, 200.0, 35.0)
        rain_3 = st.number_input("3-Day Accumulation (mm)", 0.0, 400.0, 80.0)
    with col2:
        rain_7 = st.number_input("7-Day Accumulation (mm)", 0.0, 600.0, 130.0)
        sim_month = sim_date.month

    if st.button("Run Simulation Model", type="primary"):
        input_df = pd.DataFrame([{
            'rainfall_mm': rain_in,
            'rain_3day_sum': rain_3,
            'rain_7day_sum': rain_7,
            'rain_anomaly': rain_in - 5.0,
            'month': sim_month
        }])
        
        prob = model.predict_proba(input_df)[0][1] * 100
        
        if prob >= 60:
            sim_widget = "weather-widget weather-widget-danger"
            sim_cond = "Severe Storm Simulation Alert"
            sim_icon = "⛈️"
        elif prob >= 30:
            sim_widget = "weather-widget weather-widget-warning"
            sim_cond = "Moderate Weather Advisory Simulation"
            sim_icon = "🌦️"
        else:
            sim_widget = "weather-widget"
            sim_cond = "Stable Conditions Simulation"
            sim_icon = "⛅"

        st.markdown(f"""
            <div class="{sim_widget}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">Simulation for {sim_date}</p>
                        <h1 style="font-size: 4.5rem; margin: 5px 0; font-weight: 800;">{rain_in:.1f}<span style="font-size: 2rem;"> mm</span></h1>
                        <p style="font-size: 1.4rem; font-weight: 600; margin-top: 5px;">{sim_cond}</p>
                        <p style="font-size: 1rem; background: rgba(0,0,0,0.2); display: inline-block; padding: 4px 14px; border-radius: 12px; margin-top: 8px;">
                            Calculated Flood Probability: <b>{prob:.1f}%</b>
                        </p>
                    </div>
                    <div style="text-align: right; font-size: 7rem; line-height: 1;">
                        {sim_icon}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# MODE 3: DYNAMIC WARD GIS MAP

elif app_mode == "Ward GIS Map":
    st.subheader("Nairobi Regional Risk Radar & Ward Nodes")
    st.markdown("Interactive spatial risk layers color-coded by model flood probability and river basin proximity.")

    ward_nodes = pd.DataFrame([
        {"ward": "Mathare / Eastleigh", "lat": -1.2686, "lon": 36.8536, "risk_mult": 1.2, "river": "Mathare River"},
        {"ward": "Kibera / Woodley", "lat": -1.3138, "lon": 36.7845, "risk_mult": 1.3, "river": "Ngong River"},
        {"ward": "Embakasi / Pipeline", "lat": -1.3032, "lon": 36.9081, "risk_mult": 1.1, "river": "Nairobi River Basin"},
        {"ward": "Westlands / Parklands", "lat": -1.2683, "lon": 36.8115, "risk_mult": 0.6, "river": "Higher Elevation"},
        {"ward": "CBD / Starehe", "lat": -1.2863, "lon": 36.8172, "risk_mult": 0.8, "river": "Nairobi River Basin"},
        {"ward": "Kasarani / Mwiki", "lat": -1.2215, "lon": 36.8906, "risk_mult": 0.7, "river": "Gathara-ini River"},
        {"ward": "Dagoretti / Riruta", "lat": -1.2989, "lon": 36.7538, "risk_mult": 0.5, "river": "Higher Elevation"}
    ])

    current_prob = 0.05
    if df_preds is not None and not df_preds.empty:
        current_prob = df_preds['flood_probability'].iloc[0]

    def calculate_node_color(row):
        adjusted_prob = min(current_prob * row['risk_mult'], 1.0)
        if adjusted_prob >= 0.60:
            return [239, 68, 68, 220]    # Red
        elif adjusted_prob >= 0.30:
            return [245, 158, 11, 220]  # Orange
        else:
            return [16, 185, 129, 200]   # Green

    ward_nodes['color'] = ward_nodes.apply(calculate_node_color, axis=1)
    ward_nodes['calculated_risk'] = ward_nodes['risk_mult'].apply(
        lambda m: f"{min(current_prob * m * 100, 100):.1f}% Risk"
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        ward_nodes,
        get_position='[lon, lat]',
        get_radius=1600,
        get_fill_color='color',
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
        pickable=True
    )

    view_state = pdk.ViewState(latitude=-1.2850, longitude=36.8350, zoom=11.2, pitch=35)
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>Ward:</b> {ward}<br/><b>Basin:</b> {river}<br/><b>Status:</b> {calculated_risk}"}
    )

    st.pydeck_chart(deck, use_container_width=True)

    st.markdown("""
        <div style="display: flex; gap: 30px; justify-content: center; margin-top: 15px; font-weight: 500;">
            <span style="color: #34d399;">● Normal / Safe Zone</span>
            <span style="color: #fbbf24;">● Moderate Warning Zone</span>
            <span style="color: #f87171;">● High Risk Flood Zone</span>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; color: #64748b;'>Nairobi Meteorological & Flood Watch Hub</p>", unsafe_allow_html=True)