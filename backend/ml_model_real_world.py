"""
Real-world optimized document forgery detection model
"""
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import logging
from PIL import Image, ImageEnhance
import pytesseract

logger = logging.getLogger(__name__)

class RealWorldModel:
    def __init__(self):
        self.model_dir = "models"
        self.rf_model = None
        self.scaler = None
        self.rf_metrics = {}
        # Threshold for deciding forged vs authentic.
        # Extremely conservative (0.99): only documents with *very* strong
        # forged signal are marked as forged. This is tuned to avoid
        # marking genuine real-world IDs as forged.
        self.forgery_threshold = 0.99
        
        # Load trained models
        self._load_models()
        
        logger.info("Real-world model initialized")
        logger.info(f"Forgery threshold: {self.forgery_threshold}")
    
    def _load_models(self):
        """Load real-world optimized Random Forest model"""
        try:
            # Load Random Forest model
            rf_path = os.path.join(self.model_dir, "real_world_rf_model.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("Loaded real-world Random Forest model")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "real_world_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded real-world feature scaler")
            
            # Load RF metrics
            rf_metrics_path = os.path.join(self.model_dir, "real_world_rf_metrics.pkl")
            if os.path.exists(rf_metrics_path):
                with open(rf_metrics_path, 'rb') as f:
                    self.rf_metrics = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading real-world models: {e}")
    
    def extract_features(self, image_path):
        """Extract real-world optimized features"""
        try:
            # Import the same feature extraction function
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from train_real_world_model import extract_real_world_features
            
            features = extract_real_world_features(image_path)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def predict(self, image_path):
        """Predict if document is forged using real-world optimized models"""
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return 0, 0.5, {"error": "Could not extract features"}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction from Random Forest only
            rf_proba = self.rf_model.predict_proba(features_scaled)[0]
            forged_confidence = float(rf_proba[1])
            
            # Make prediction
            if forged_confidence >= self.forgery_threshold:
                prediction = 1  # Forged
            else:
                prediction = 0  # Authentic
            
            # Create detailed results
            details = {
                "rf_prediction": int(np.argmax(rf_proba)),
                "rf_confidence": float(rf_proba[1]),
                "ensemble_confidence": float(forged_confidence),
                "threshold": self.forgery_threshold,
                "rf_metrics": self.rf_metrics,
                "features_used": len(features[0]) if len(features.shape) > 1 else len(features),
                "model_type": "real_world_optimized"
            }
            
            return prediction, forged_confidence, details
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            prediction = 0
            confidence = 0.5
            details = {
                "error": str(e),
                "fallback": True,
                "fallback_reason": "Error in prediction - defaulting to authentic",
                "rf_metrics": self.rf_metrics
            }
            return prediction, confidence, details
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_type": "Real-World Optimized",
            "models": ["Random Forest"],
            "threshold": self.forgery_threshold,
            "features": "Real-world optimized (image quality, texture, OCR)",
            "rf_metrics": self.rf_metrics,
            "training_data": "FantasyID + Real-world optimization",
            "preprocessing": "Grayscale + Noise reduction + Contrast enhancement + Sharpening"
        }

def get_real_world_model():
    """Get real-world optimized model instance"""
    return RealWorldModel()
