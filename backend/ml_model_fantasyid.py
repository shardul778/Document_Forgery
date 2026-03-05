"""
FantasyID-trained document forgery detection model
Uses models trained on real ID card dataset
"""
import os
import numpy as np
import cv2
import pickle
import logging
from PIL import Image, ImageEnhance
import pytesseract
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FantasyIDModel:
    def __init__(self):
        self.model_dir = "models"
        self.rf_model = None
        self.scaler = None
        self.rf_metrics = {}
        # Threshold for forged vs authentic on FantasyID-style IDs.
        # Extremely conservative (0.99): only documents with *very* high
        # forged probability are flagged as forged. This heavily reduces
        # false positives on authentic IDs like Aadhaar.
        self.forgery_threshold = 0.99
        
        # Load trained models
        self._load_models()
        
        logger.info("FantasyID model initialized")
        logger.info(f"Forgery threshold: {self.forgery_threshold}")
    
    def _load_models(self):
        """Load FantasyID-trained Random Forest model"""
        try:
            # Load Random Forest model
            rf_path = os.path.join(self.model_dir, "fantasyid_rf_model.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("Loaded FantasyID Random Forest model")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "fantasyid_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded FantasyID feature scaler")
            else:
                # Fallback to old scaler
                old_scaler_path = os.path.join(self.model_dir, "scaler.pkl")
                if os.path.exists(old_scaler_path):
                    with open(old_scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info("Loaded fallback feature scaler")

            # Load metrics (Random Forest only)
            rf_metrics_path = os.path.join(self.model_dir, "fantasyid_rf_metrics.pkl")
            if os.path.exists(rf_metrics_path):
                with open(rf_metrics_path, 'rb') as f:
                    self.rf_metrics = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading FantasyID models: {e}")
    
    def extract_features(self, image_path):
        """Extract features matching the training pipeline"""
        try:
            # Import the same feature extraction function used in training
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from train_simple_effective import extract_simple_features
            
            features = extract_simple_features(image_path)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def predict(self, image_path):
        """Predict if document is forged using FantasyID models"""
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return 0, 0.5, {"error": "Could not extract features"}
            
            # Ensure features is 2D
            features = features.reshape(1, -1)
            
            # Check if model is loaded
            if self.rf_model is None or self.scaler is None:
                logger.warning("FantasyID RF model not loaded - using fallback")
                return 0, 0.5, {"fallback": True, "reason": "Models not loaded"}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction from Random Forest only
            rf_proba = self.rf_model.predict_proba(features_scaled)[0]
            forged_confidence = float(rf_proba[1])
            
            # Apply threshold
            prediction = 1 if forged_confidence >= self.forgery_threshold else 0
            
            # Prepare details
            details = {
                "model_type": "FantasyID",
                "rf_prediction": int(np.argmax(rf_proba)),
                "rf_confidence": float(rf_proba[1]),
                "ensemble_confidence": float(forged_confidence),
                "threshold": self.forgery_threshold,
                "rf_metrics": self.rf_metrics,
                "features_used": len(features[0])
            }
            
            return prediction, forged_confidence, details
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.5, {"error": str(e)}
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_type": "FantasyID",
            "trained": self.rf_model is not None,
            "rf_metrics": self.rf_metrics,
            "threshold": self.forgery_threshold,
            "dataset": "FantasyID (Real ID cards)"
        }

# Global instance
_fantasyid_model = None

def get_fantasyid_model():
    """Get global FantasyID model instance"""
    global _fantasyid_model
    if _fantasyid_model is None:
        _fantasyid_model = FantasyIDModel()
    return _fantasyid_model
