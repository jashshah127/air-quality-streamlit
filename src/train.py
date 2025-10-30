"""
Train Air Quality Index prediction model
"""
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
from datetime import datetime
from data import AirQualityDataGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIModelTrainer:
    """Train AQI prediction model"""
    
    def __init__(self, model_dir="../model"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_model(self, n_estimators=100, random_state=42):
        """Train the model"""
        logger.info("Starting training...")
        
        data_gen = AirQualityDataGenerator()
        data = data_gen.prepare_data()
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            random_state=random_state
        )
        
        logger.info("Training Random Forest Regressor...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        self._evaluate(X_train, X_test, y_train, y_test)
        
        logger.info("Training completed!")
    
    def _evaluate(self, X_train, X_test, y_train, y_test):
        """Evaluate model"""
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.metrics = {
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_.astype(float))),
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestRegressor'
        }
        
        print(f"\nTraining RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test R²: {test_r2:.4f}")
        
        print("\nTop 5 Important Features:")
        sorted_features = sorted(self.metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features[:5]:
            print(f"  {feat}: {imp:.4f}")
    
    def save_model(self):
        """Save model artifacts"""
        joblib.dump(self.model, os.path.join(self.model_dir, "aqi_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        
        with open(os.path.join(self.model_dir, "metrics.json"), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        with open(os.path.join(self.model_dir, "feature_names.json"), 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info("Model saved successfully")

def main():
    trainer = AQIModelTrainer()
    trainer.train_model()
    trainer.save_model()
    print("\n" + "="*50)
    print("Model training completed!")
    print("="*50)

if __name__ == "__main__":
    main()