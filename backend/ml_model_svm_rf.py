"""
Document Forgery Detection Model - SVM and Random Forest Implementation
Objective 3: Classify documents using SVM and Random Forest algorithms
"""
import numpy as np
import pickle
import logging
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentForgerySVM_RF:
    """SVM and Random Forest model for document forgery detection"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model_dir_path = Path(model_dir)
        self.model_dir_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.svm_model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0,
            gamma='scale'
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Model metrics (will be updated after training)
        self.svm_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        self.rf_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        # Ensemble weights (can be adjusted based on performance)
        self.svm_weight = 0.5
        self.rf_weight = 0.5
        
        # Forgery threshold (made very conservative to reduce false positives)
        self.forgery_threshold = 0.95
        
        # Load trained models if available
        self._load_models()
        
        logger.info("SVM and Random Forest models initialized")
        logger.info(f"Forgery threshold: {self.forgery_threshold}")
    
    def _load_models(self):
        """Load trained models, scaler, and metrics if available"""
        svm_path = os.path.join(self.model_dir, "svm_model.pkl")
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        svm_metrics_path = os.path.join(self.model_dir, "svm_metrics.pkl")
        rf_metrics_path = os.path.join(self.model_dir, "rf_metrics.pkl")
        
        models_loaded = False
        
        # Load SVM model
        if os.path.exists(svm_path):
            try:
                with open(svm_path, 'rb') as f:
                    self.svm_model = pickle.load(f)
                logger.info(f"Loaded SVM model from {svm_path}")
                models_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load SVM model: {e}")
        
        # Load Random Forest model
        if os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info(f"Loaded Random Forest model from {rf_path}")
                models_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load Random Forest model: {e}")
        
        # Load scaler
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
        
        # Load metrics
        if os.path.exists(svm_metrics_path):
            try:
                with open(svm_metrics_path, 'rb') as f:
                    self.svm_metrics = pickle.load(f)
                logger.info(f"Loaded SVM metrics: {self.svm_metrics}")
            except Exception as e:
                logger.warning(f"Failed to load SVM metrics: {e}")
        
        if os.path.exists(rf_metrics_path):
            try:
                with open(rf_metrics_path, 'rb') as f:
                    self.rf_metrics = pickle.load(f)
                logger.info(f"Loaded RF metrics: {self.rf_metrics}")
            except Exception as e:
                logger.warning(f"Failed to load RF metrics: {e}")
        
        if not models_loaded:
            logger.info("No trained models found. Models will use default parameters.")
            logger.info("Run 'python train_svm_rf_model.py' to train the models.")
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict if document is forged using ensemble of SVM and Random Forest
        
        Args:
            features: numpy array of shape (80,) containing document features
            
        Returns:
            tuple: (prediction, confidence, details)
                - prediction: 0 (authentic) or 1 (forged)
                - confidence: confidence score (0.0 to 1.0)
                - details: dict with individual model predictions and metrics
        """
        try:
            # Ensure features is 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Check if models are properly trained
            if not (os.path.exists(os.path.join(self.model_dir, "svm_model.pkl")) and 
                   os.path.exists(os.path.join(self.model_dir, "rf_model.pkl"))):
                logger.warning("Models not trained - using conservative prediction")
                return 0, 0.6, {"fallback": "not_trained", "reason": "Models not trained"}
            
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Get predictions from both models
            svm_proba = self.svm_model.predict_proba(features_scaled)[0]
            rf_proba = self.rf_model.predict_proba(features_scaled)[0]
            
            # Ensemble prediction (weighted average)
            ensemble_proba = (self.svm_weight * svm_proba + self.rf_weight * rf_proba)
            
            # Get confidence for forged class (class 1)
            forged_confidence = ensemble_proba[1]
            
            # Apply threshold with safety margin and bias towards authentic
            if forged_confidence >= 0.98:  # Very high confidence required
                prediction = 1
            elif forged_confidence <= 0.20:  # Low confidence = authentic
                prediction = 0
            else:
                # For medium confidence, be very conservative (bias towards authentic)
                prediction = 1 if forged_confidence >= self.forgery_threshold else 0
            
            # Prepare details
            details = {
                "svm_prediction": int(np.argmax(svm_proba)),
                "svm_confidence": float(svm_proba[1]),
                "rf_prediction": int(np.argmax(rf_proba)),
                "rf_confidence": float(rf_proba[1]),
                "ensemble_confidence": float(forged_confidence),
                "threshold": self.forgery_threshold,
                "svm_metrics": self.svm_metrics,
                "rf_metrics": self.rf_metrics,
                "features_used": len(features[0]),
                "prediction_logic": "conservative" if 0.15 < forged_confidence < 0.95 else "direct"
            }
            
            return prediction, forged_confidence, details
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Conservative fallback - assume authentic unless very strong evidence
            prediction = 0  # Default to authentic
            confidence = 0.5  # Neutral confidence
            details = {
                "error": str(e),
                "fallback": True,
                "fallback_reason": "Error in prediction - defaulting to authentic",
                "svm_metrics": self.svm_metrics,
                "rf_metrics": self.rf_metrics
            }
            return prediction, confidence, details
    
    def get_model_info(self) -> dict:
        """Get information about the models"""
        return {
            "svm_model": {
                "type": "Support Vector Machine",
                "kernel": self.svm_model.kernel,
                "C": self.svm_model.C,
                "gamma": self.svm_model.gamma,
                "probability": self.svm_model.probability,
                "metrics": self.svm_metrics
            },
            "rf_model": {
                "type": "Random Forest",
                "n_estimators": self.rf_model.n_estimators,
                "max_depth": self.rf_model.max_depth,
                "min_samples_split": self.rf_model.min_samples_split,
                "min_samples_leaf": self.rf_model.min_samples_leaf,
                "metrics": self.rf_metrics
            },
            "ensemble": {
                "svm_weight": self.svm_weight,
                "rf_weight": self.rf_weight,
                "threshold": self.forgery_threshold
            },
            "trained": os.path.exists(os.path.join(self.model_dir, "svm_model.pkl")) and 
                     os.path.exists(os.path.join(self.model_dir, "rf_model.pkl"))
        }

# Global model instance
_model_instance = None

def get_document_forgery_model():
    """Get the global model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = DocumentForgerySVM_RF()
    return _model_instance
