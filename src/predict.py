"""
Prediction utilities for AQI
"""
import joblib
import numpy as np
import json
import os

class AQIPredictor:
    """Predict AQI from pollutant levels"""
    
    def __init__(self, model_dir="model"):
        self.model = joblib.load(os.path.join(model_dir, "aqi_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        
        with open(os.path.join(model_dir, "feature_names.json")) as f:
            self.feature_names = json.load(f)
        
        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.metrics = json.load(f)
    
    def predict(self, features):
        """Predict AQI from features"""
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        aqi = self.model.predict(X_scaled)[0]
        return float(aqi)
    
    def get_aqi_info(self, aqi):
        """Get AQI category and health info"""
        if aqi <= 50:
            return {
                'category': 'Good',
                'color': '#00E400',
                'health': 'Air quality is satisfactory',
                'recommendation': 'Enjoy outdoor activities!'
            }
        elif aqi <= 100:
            return {
                'category': 'Moderate',
                'color': '#FFFF00',
                'health': 'Acceptable air quality',
                'recommendation': 'Unusually sensitive people should consider limiting prolonged outdoor exertion'
            }
        elif aqi <= 150:
            return {
                'category': 'Unhealthy for Sensitive Groups',
                'color': '#FF7E00',
                'health': 'Members of sensitive groups may experience health effects',
                'recommendation': 'Sensitive groups should reduce prolonged outdoor exertion'
            }
        elif aqi <= 200:
            return {
                'category': 'Unhealthy',
                'color': '#FF0000',
                'health': 'Everyone may begin to experience health effects',
                'recommendation': 'Everyone should reduce prolonged outdoor exertion'
            }
        elif aqi <= 300:
            return {
                'category': 'Very Unhealthy',
                'color': '#8F3F97',
                'health': 'Health alert: everyone may experience serious health effects',
                'recommendation': 'Everyone should avoid all outdoor exertion'
            }
        else:
            return {
                'category': 'Hazardous',
                'color': '#7E0023',
                'health': 'Health warnings of emergency conditions',
                'recommendation': 'Everyone should remain indoors'
            }