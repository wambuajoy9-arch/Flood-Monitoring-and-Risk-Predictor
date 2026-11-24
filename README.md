# Nairobi Flood Monitoring & Risk Predictor

A flood risk monitoring and early-warning web application for Nairobi, combining historical rainfall patterns, spatial ward/river data, machine learning models, and a weather-app style Streamlit dashboard.

## Project Overview

Flooding in Nairobi often develops from multi-day rainfall accumulation on already-saturated ground rather than a single isolated shower. This project frames flood risk estimation as a **binary classification problem** (`1` = flood risk, `0` = normal/baseline conditions) using daily rainfall metrics, cumulative rainfall windows, rainfall anomalies, seasonal context, and spatial layers for Nairobi.

Because severe flood events are rare compared to normal days, predicted probabilities on calm or low-rain days are naturally small (e.g., around `0.5%`) and increase as heavier or multi-day rainfall develops.

## Key Features

- **Machine Learning Benchmarking:** Trains and evaluates `XGBClassifier` and `RandomForestClassifier` using ROC-AUC scores.
- **Live 7-Day Forecast Ingestion:** Pulls daily precipitation forecasts for Nairobi from the Open-Meteo API and engineers rolling rainfall features (`3-day` and `7-day` sums, `month`, and `rain_anomaly`).
- **Geospatial Context:** Integrates Nairobi ward boundaries (`Nairobi_wards.geojson`), river networks (`Nairobi_rivers.geojson`), elevation data (`output_SRTMGL1.tif`), and administrative boundary layers.
- **Weather-App Style Dashboard:** Built with Streamlit and PyDeck, featuring dynamic risk cards, forecast views, scenario simulation, and interactive map layers.

## Project Structure

```text
Flood-Monitoring-and-Risk-Predictor/
├── DATA/
│   ├── extracted_shp/               # Unpacked administrative shapefiles
│   ├── FL20240426KEN_SHP.zip        # Kenya administrative boundary shapefile archive
│   ├── ken-rainfall-subnat-full (1).csv # Subnational rainfall dataset
│   └── output_SRTMGL1.tif           # SRTM Digital Elevation Model (DEM) GeoTIFF
├── app.py                           # Streamlit dashboard application
├── flood_model.joblib               # Saved trained model artifact
├── forecasting.ipynb                # Data prep, training, feature engineering & ROC-AUC plots
├── live_forecast_api.py             # Fetches 7-day Open-Meteo forecast & engineers features
├── model_comparison_bar_chart.png   # ROC-AUC bar chart comparison
├── model_roc_curves.png             # ROC curve overlay plot
├── nairobi_flood_risk_output.csv    # Generated flood risk prediction outputs
├── nairobi_live_forecast.csv        # Processed 7-day live forecast data
├── Nairobi_rivers.geojson           # Nairobi rivers spatial layer
├── Nairobi_wards.geojson            # Nairobi wards spatial boundaries
├── predictions.py                   # Model inference script for generating risk outputs
├── Project_Template.md              # Coursework/project template reference
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies