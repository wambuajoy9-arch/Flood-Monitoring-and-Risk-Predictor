import requests
import pandas as pd
import numpy as np

def fetch_nairobi_forecast():
    """Fetches 7-day live precipitation forecast for Nairobi via Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -1.286389,
        "longitude": 36.817223,
        "daily": ["precipitation_sum"],
        "timezone": "Africa/Nairobi"
    }
    
    print("Calling Open-Meteo Live Forecast API...")
    response = requests.get(url, params=params)
    data = response.json()
    
    # Format into DataFrame
    df_forecast = pd.DataFrame({
        'date': data['daily']['time'],
        'rain_mm': data['daily']['precipitation_sum']
    })
    
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    
    # Engineer rolling features on live forecast
    df_forecast['rain_3day_sum'] = df_forecast['rain_mm'].rolling(window=3, min_periods=1).sum()
    df_forecast['rain_7day_sum'] = df_forecast['rain_mm'].rolling(window=7, min_periods=1).sum()
    
    df_forecast['month'] = df_forecast['date'].dt.month
    # Monthly mean fallback baseline
    df_forecast['rain_anomaly'] = df_forecast['rain_mm'] - 5.0 

    print("Live 7-Day Forecast Retrieved:")
    print(df_forecast[['date', 'rain_mm', 'rain_3day_sum']])
    
    df_forecast.to_csv("nairobi_live_forecast.csv", index=False)
    return df_forecast

if __name__ == "__main__":
    fetch_nairobi_forecast()