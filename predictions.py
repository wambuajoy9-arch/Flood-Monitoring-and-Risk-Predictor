import pandas as pd
import numpy as np
import joblib
import os

print(" Starting Batch Inference Pipeline for Nairobi Flood Risk Prediction...")

# 1. Load the trained model and live forecast data
model_path = "flood_model.joblib"
forecast_path = "nairobi_live_forecast.csv"

model = joblib.load(model_path)
df_forecast = pd.read_csv(forecast_path)
df_forecast['date'] = pd.to_datetime(df_forecast['date'])

if 'rain_mm' in df_forecast.columns and 'rainfall_mm' not in df_forecast.columns:
    df_forecast = df_forecast.rename(columns={'rain_mm': 'rainfall_mm'})

# 2. Extract the exact feature set used during training (matching 'rainfall_mm')
features = ['rainfall_mm', 'rain_3day_sum', 'rain_7day_sum', 'rain_anomaly', 'month']
X_live = df_forecast[features]

# 3. Generate Predictions & Probabilities
df_forecast['flood_probability'] = model.predict_proba(X_live)[:, 1]
df_forecast['predicted_flood_alert'] = model.predict(X_live)

# 4. Assign Qualitative Risk Tiers based on Probability
def assign_risk_tier(prob):
    if prob >= 0.70:
        return "HIGH RISK (Red Alert)"
    elif prob >= 0.40:
        return "MODERATE RISK (Warning)"
    else:
        return "LOW RISK (Normal)"

df_forecast['risk_tier'] = df_forecast['flood_probability'].apply(assign_risk_tier)

# 5. Export Structured Output for Monitoring & Dashboard Consumption
output_filename = "nairobi_flood_risk_output.csv"
df_forecast.to_csv(output_filename, index=False)

print(f"\nSuccessfully generated predictions! Output saved to '{output_filename}':\n")
print(df_forecast[['date', 'rainfall_mm', 'rain_3day_sum', 'flood_probability', 'risk_tier']])