import streamlit as st
import pandas as pd
import json

# 1. Page Configuration
st.set_page_config(
    page_title="Nairobi Flood Risk & Early Warning System",
    page_icon="🌊",
    layout="wide"
)

st.title(" Nairobi Ward-Level Flood Risk Monitoring System")
st.markdown("Global deployment instance powered by XGBoost machine learning & Open-Meteo live weather forecasts.")

# 2. Load Data Safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("nairobi_flood_risk_output.csv")
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception as e:
        return None

df_preds = load_data()

if df_preds is None:
    st.error("Prediction output file not found! Please run your batch inference script first.")
else:
    # 3. Sidebar: Interactive Date Picker & Controls
    st.sidebar.header("Control Panel")
    st.sidebar.markdown("Select a date to inspect forecast risks:")
    
    min_date = df_preds['date'].min()
    max_date = df_preds['date'].max()
    
    # Interactive date selection widget
    selected_date = st.sidebar.date_input(
        "Target Forecast Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    # Filter data based on user input date
    filtered_row = df_preds[df_preds['date'] == selected_date]

    # 4. Main Dashboard Layout
    st.markdown(f"### Forecast Analysis for: **{selected_date}**")

    if not filtered_row.empty:
        row = filtered_row.iloc[0]
        
        # Metric Cards Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Rainfall", f"{row['rainfall_mm']:.1f} mm")
        col2.metric("3-Day Cumulative", f"{row['rain_3day_sum']:.1f} mm")
        col3.metric("Flood Probability", f"{row['flood_probability'] * 100:.1f}%")
        col4.metric("Risk Status", row['risk_tier'])

        # Visual alert box depending on risk level
        if "HIGH" in row['risk_tier']:
            st.error("⚠️ **RED ALERT:** High probability of flooding detected for this date. Monitor local drainage channels.")
        elif "MODERATE" in row['risk_tier']:
            st.warning("⚠️ **WARNING:** Moderate risk conditions. Exercise caution in flood-prone informal settlements.")
        else:
            st.success("✅ **NORMAL:** Weather conditions are stable with low flood probability.")

    else:
        st.warning("No forecast data available for the selected date.")

    # 5. Full 7-Day Forecast Overview Table
    st.markdown("---")
    st.subheader("7-Day Operational Forecast Table")
    st.dataframe(df_preds[['date', 'rainfall_mm', 'rain_3day_sum', 'flood_probability', 'risk_tier']], use_container_width=True)

    # 6. Model Metadata Footer
    st.markdown("---")
    st.markdown("**System Architecture:** XGBoost Binary Classification | Spatial Layers: 85 Nairobi Wards & 730 Rivers")