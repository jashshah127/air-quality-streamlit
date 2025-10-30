"""
Data generation for Air Quality Index prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirQualityDataGenerator:
    """Generate synthetic air quality data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
            'temperature', 'humidity', 'wind_speed'
        ]
        self.feature_info = {
            'PM2.5': {'unit': 'μg/m³', 'range': (0, 500), 'desc': 'Fine Particulate Matter'},
            'PM10': {'unit': 'μg/m³', 'range': (0, 600), 'desc': 'Coarse Particulate Matter'},
            'NO2': {'unit': 'ppb', 'range': (0, 200), 'desc': 'Nitrogen Dioxide'},
            'SO2': {'unit': 'ppb', 'range': (0, 100), 'desc': 'Sulfur Dioxide'},
            'CO': {'unit': 'ppm', 'range': (0, 10), 'desc': 'Carbon Monoxide'},
            'O3': {'unit': 'ppb', 'range': (0, 150), 'desc': 'Ozone'},
            'temperature': {'unit': '°C', 'range': (-10, 45), 'desc': 'Temperature'},
            'humidity': {'unit': '%', 'range': (10, 100), 'desc': 'Relative Humidity'},
            'wind_speed': {'unit': 'km/h', 'range': (0, 50), 'desc': 'Wind Speed'}
        }
    
    def generate_data(self, n_samples=2000, random_state=42):
        """
        Generate synthetic air quality dataset
        
        Args:
            n_samples: Number of samples
            random_state: Random seed
            
        Returns:
            DataFrame with features and AQI
        """
        logger.info(f"Generating {n_samples} air quality samples...")
        np.random.seed(random_state)
        
        # Generate pollutant concentrations
        data = {
            'PM2.5': np.random.gamma(2, 20, n_samples),
            'PM10': np.random.gamma(2, 30, n_samples),
            'NO2': np.random.gamma(2, 15, n_samples),
            'SO2': np.random.gamma(2, 8, n_samples),
            'CO': np.random.gamma(2, 1, n_samples),
            'O3': np.random.gamma(2, 20, n_samples),
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'wind_speed': np.random.gamma(2, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Clip values to realistic ranges
        df['PM2.5'] = np.clip(df['PM2.5'], 0, 500)
        df['PM10'] = np.clip(df['PM10'], 0, 600)
        df['NO2'] = np.clip(df['NO2'], 0, 200)
        df['SO2'] = np.clip(df['SO2'], 0, 100)
        df['CO'] = np.clip(df['CO'], 0, 10)
        df['O3'] = np.clip(df['O3'], 0, 150)
        df['temperature'] = np.clip(df['temperature'], -10, 45)
        df['wind_speed'] = np.clip(df['wind_speed'], 0, 50)
        
        # Calculate AQI based on pollutants (simplified formula)
        df['AQI'] = self._calculate_aqi(df)
        
        logger.info(f"Data generated successfully")
        logger.info(f"AQI range: {df['AQI'].min():.1f} - {df['AQI'].max():.1f}")
        
        return df
    
    def _calculate_aqi(self, df):
        """Calculate AQI from pollutant levels"""
        # Simplified AQI calculation (weighted average)
        aqi = (
            df['PM2.5'] * 0.4 +
            df['PM10'] * 0.3 +
            df['NO2'] * 0.5 +
            df['SO2'] * 0.6 +
            df['CO'] * 15 +
            df['O3'] * 0.4 -
            df['wind_speed'] * 2
        )
        
        # Ensure AQI is positive and realistic
        aqi = np.clip(aqi, 0, 500)
        
        return aqi
    
    def get_aqi_category(self, aqi_value):
        """Get AQI category and color"""
        if aqi_value <= 50:
            return 'Good', '#00E400'
        elif aqi_value <= 100:
            return 'Moderate', '#FFFF00'
        elif aqi_value <= 150:
            return 'Unhealthy for Sensitive Groups', '#FF7E00'
        elif aqi_value <= 200:
            return 'Unhealthy', '#FF0000'
        elif aqi_value <= 300:
            return 'Very Unhealthy', '#8F3F97'
        else:
            return 'Hazardous', '#7E0023'
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        df = self.generate_data(random_state=random_state)
        
        X = df[self.feature_names]
        y = df['AQI']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }