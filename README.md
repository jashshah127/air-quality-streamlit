# Air Quality Index Prediction Dashboard

Interactive Streamlit dashboard for predicting Air Quality Index using machine learning.

## Overview

MLOps project demonstrating interactive dashboards with Streamlit for environmental data analysis.

**Features:**
- Real-time AQI prediction from pollutant levels
- Interactive visualizations with Plotly
- Health recommendations based on air quality
- Random Forest Regressor model

## Quick Start

### 1. Clone and Setup
```
git clone https://github.com/jashshah127/air-quality-streamlit.git
cd air-quality-streamlit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model
```
cd src
python train.py
cd ..
```

### 3. Run Dashboard
```
streamlit run app.py
```

Dashboard opens at http://localhost:8501

## Project Structure
```
air-quality-streamlit/
├── app.py              # Main Streamlit dashboard
├── src/
│   ├── data.py        # Data generation
│   ├── train.py       # Model training
│   └── predict.py     # Predictions
├── model/             # Trained model files
├── requirements.txt
└── README.md
```

## Features

### Input Parameters
- PM2.5, PM10 (Particulate Matter)
- NO2, SO2, CO, O3 (Gases)
- Temperature, Humidity, Wind Speed

### Output
- AQI value (0-500)
- Health category (Good/Moderate/Unhealthy/Hazardous)
- Health impact and recommendations
- Interactive visualizations

## AQI Categories

| Range | Category | Color |
|-------|----------|-------|
| 0-50 | Good | Green |
| 51-100 | Moderate | Yellow |
| 101-150 | Unhealthy for Sensitive Groups | Orange |
| 151-200 | Unhealthy | Red |
| 201-300 | Very Unhealthy | Purple |
| 301+ | Hazardous | Maroon |

## Model Performance

- Model: Random Forest Regressor
- Test RMSE: ~12
- Test R²: ~0.89
- Top Features: CO, Wind Speed, PM10

## Assignment Requirements Met

- Different dataset: Air quality environmental data
- Different model: Random Forest Regressor (regression task)
- Enhanced features: Interactive dashboard with visualizations
- Health integration: AQI categories and recommendations

## Technical Stack

- Python 3.9+
- Streamlit 1.28.0
- scikit-learn 1.3.2
- Plotly 5.18.0
- pandas, numpy

## Usage Example

1. Adjust pollutant sliders in sidebar
2. Set environmental conditions
3. Click "Predict AQI" button
4. View AQI value with health category
5. Explore visualizations in tabs

## Author

**Jash Shah**
- GitHub: @jashshah127
- Course: MLOps - Northeastern University
- Date: October 30, 2025 - Thursday

## Repository

https://github.com/jashshah127/air-quality-streamlit
