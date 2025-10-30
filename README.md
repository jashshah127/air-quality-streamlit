# Air Quality Index Prediction Dashboard

An interactive Streamlit dashboard for predicting and visualizing Air Quality Index (AQI) using machine learning.

## Project Overview

This project demonstrates MLOps practices with interactive dashboards, featuring:
- **Interactive Dashboard**: Real-time AQI prediction with Streamlit
- **ML Model**: Random Forest Regressor for AQI prediction
- **Visualizations**: Interactive charts with Plotly
- **Health Recommendations**: AQI-based health guidance
- **Environmental Application**: Public health and air quality monitoring

## Project Structure

```
air-quality-streamlit/
├── app.py                  # Main Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── data.py            # Data generation
│   ├── train.py           # Model training
│   └── predict.py         # Prediction utilities
├── model/                 # Trained model artifacts
│   ├── aqi_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── feature_names.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/jashshah127/air-quality-streamlit.git
cd air-quality-streamlit

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd src
python train.py
cd ..
```

Expected output:
- Test RMSE: ~20-30
- Test R²: ~0.85-0.95
- Creates model files in `model/` directory

### 3. Run the Dashboard

```bash
streamlit run app.py
```

Dashboard will open automatically in your browser at: **http://localhost:8501**

## Features

### Interactive Prediction
- **Pollutant Sliders**: Adjust PM2.5, PM10, NO₂, SO₂, CO, O₃ levels
- **Environmental Conditions**: Set temperature, humidity, wind speed
- **Real-time AQI**: Instant predictions with health categorization
- **Color-coded Results**: Visual AQI categories (Green → Red → Purple)

### Visualizations
- **Pollutant Bar Charts**: Compare current pollutant levels
- **AQI Gauge**: Interactive gauge showing current air quality
- **Environmental Parameters**: Temperature, humidity, wind visualization
- **Feature Importance**: Which factors most affect AQI

### Health Guidance
- **AQI Categories**: Good, Moderate, Unhealthy, Hazardous
- **Health Impact**: Specific health effects for each category
- **Recommendations**: Activity recommendations based on air quality

## AQI Categories

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0-50 | Good | Green | Air quality is satisfactory |
| 51-100 | Moderate | Yellow | Acceptable air quality |
| 101-150 | Unhealthy for Sensitive Groups | Orange | Sensitive groups affected |
| 151-200 | Unhealthy | Red | Everyone may experience effects |
| 201-300 | Very Unhealthy | Purple | Health alert conditions |
| 301+ | Hazardous | Maroon | Emergency conditions |

## Model Details

### Input Features (9 total)
1. **PM2.5** - Fine Particulate Matter (μg/m³)
2. **PM10** - Coarse Particulate Matter (μg/m³)
3. **NO₂** - Nitrogen Dioxide (ppb)
4. **SO₂** - Sulfur Dioxide (ppb)
5. **CO** - Carbon Monoxide (ppm)
6. **O₃** - Ozone (ppb)
7. **Temperature** (°C)
8. **Humidity** (%)
9. **Wind Speed** (km/h)

### Model Performance
- **Model Type**: Random Forest Regressor
- **Test RMSE**: ~25 (typical)
- **Test R²**: ~0.90
- **Training Samples**: 2000

### Top Important Features
1. PM2.5 (Fine particles)
2. PM10 (Coarse particles)
3. NO₂ (Nitrogen dioxide)

## Key Differences from Original Lab

### Modifications Made
- **Different Dataset**: Air quality environmental data (vs original)
- **Different Model**: Random Forest Regressor for regression task
- **Problem Type**: Regression (predict continuous AQI) vs classification
- **Application**: Environmental/public health focus
- **Enhanced Visualizations**: Interactive Plotly charts, AQI gauge
- **Health Integration**: AQI categories with health recommendations

### Dashboard Features
- Real-time prediction with interactive sliders
- Multi-tab interface (Visualizations, Performance, About)
- Color-coded AQI categories
- Health impact information
- Feature importance analysis
- Responsive design

## Usage Examples

### Example 1: Good Air Quality
```
PM2.5: 20, PM10: 30, NO₂: 15, SO₂: 5, CO: 0.5, O₃: 30
Temperature: 22°C, Humidity: 50%, Wind: 15 km/h
→ AQI: ~35 (Good) ✅
```

### Example 2: Moderate Air Quality
```
PM2.5: 60, PM10: 85, NO₂: 40, SO₂: 20, CO: 2.0, O₃: 60
Temperature: 28°C, Humidity: 70%, Wind: 5 km/h
→ AQI: ~95 (Moderate) ⚠️
```

### Example 3: Unhealthy Air Quality
```
PM2.5: 150, PM10: 200, NO₂: 80, SO₂: 40, CO: 5.0, O₃: 100
Temperature: 35°C, Humidity: 80%, Wind: 2 km/h
→ AQI: ~180 (Unhealthy) 🚨
```

## Technical Stack

- **Python**: 3.9+
- **Dashboard**: Streamlit 1.28.0
- **ML Library**: scikit-learn 1.3.2
- **Visualization**: Plotly 5.18.0
- **Data Processing**: pandas, numpy

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

## Troubleshooting

### Model Not Found
```
### Port Already in Use
```
ERROR: Port 8501 is already in use
```
**Solution**: `streamlit run app.py --server.port 8502`

### Module Import Errors
**Solution**: Activate virtual environment and reinstall requirements

## Future Enhancements

- Real-time data integration from air quality APIs
- Historical AQI trends and forecasting
- Location-based predictions
- Multiple city comparisons
- Alert system for unhealthy conditions
- Export predictions to CSV

## License

Educational project for MLOps course at Northeastern University.

## Author

**Jash Shah**
- GitHub: [@jashshah127](https://github.com/jashshah127)
- Course: MLOps - Northeastern University
- Date: October 30, 2025

---

**Repository**: https://github.com/jashshah127/air-quality-streamlit