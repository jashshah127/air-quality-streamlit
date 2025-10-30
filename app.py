"""
Streamlit Dashboard for Air Quality Index Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.predict import AQIPredictor
from src.data import AirQualityDataGenerator

# Page config
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_predictor():
    try:
        return AQIPredictor()
    except:
        st.error("Model not found! Run: python src/train.py")
        st.stop()

predictor = load_predictor()

# Title
st.title("🌍 Air Quality Index Dashboard")
st.markdown("**Predict and visualize air quality using machine learning**")

# Sidebar inputs
st.sidebar.header("📊 Input Parameters")
pm25 = st.sidebar.slider("PM2.5 (μg/m³)", 0, 500, 50)
pm10 = st.sidebar.slider("PM10 (μg/m³)", 0, 600, 75)
no2 = st.sidebar.slider("NO₂ (ppb)", 0, 200, 30)
so2 = st.sidebar.slider("SO₂ (ppb)", 0, 100, 15)
co = st.sidebar.slider("CO (ppm)", 0.0, 10.0, 1.0, 0.1)
o3 = st.sidebar.slider("O₃ (ppb)", 0, 150, 40)

st.sidebar.markdown("---")
st.sidebar.header("🌡️ Environmental")
temperature = st.sidebar.slider("Temperature (°C)", -10, 45, 20)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 60)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)

# Predict
if st.sidebar.button("🔮 Predict AQI"):
    features = [pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed]
    aqi = predictor.predict(features)
    info = predictor.get_aqi_info(aqi)
    
    st.markdown(f"## AQI: {aqi:.0f} - {info['category']}")
    st.markdown(f"**Health**: {info['health']}")
    st.markdown(f"**Recommendation**: {info['recommendation']}")

# Visualizations
st.header("📈 Current Levels")

col1, col2 = st.columns(2)

with col1:
    pollutant_data = pd.DataFrame({
        'Pollutant': ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'CO', 'O₃'],
        'Level': [pm25, pm10, no2, so2, co*10, o3]
    })
    fig1 = px.bar(pollutant_data, x='Pollutant', y='Level', 
                  color='Level', color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    env_data = pd.DataFrame({
        'Parameter': ['Temp', 'Humidity', 'Wind'],
        'Value': [temperature, humidity, wind_speed]
    })
    fig2 = px.bar(env_data, x='Parameter', y='Value',
                  color='Value', color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)

# Model performance
st.header("🎯 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Test RMSE", f"{predictor.metrics['test_rmse']:.2f}")
col2.metric("Test MAE", f"{predictor.metrics['test_mae']:.2f}")
col3.metric("R² Score", f"{predictor.metrics['test_r2']:.4f}")