"""
Debug script to understand prediction behavior
"""
import numpy as np
from ml_model_svm_rf import get_document_forgery_model
from document_analyzer import DocumentAnalyzer
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_prediction():
    """Debug the prediction process"""
    logger.info("=== DEBUG PREDICTION ===")
    
    # Load model
    model = get_document_forgery_model()
    logger.info(f"Model threshold: {model.forgery_threshold}")
    logger.info(f"Models trained: {model.get_model_info()['trained']}")
    
    # Load training metrics
    try:
        with open('models/svm_metrics.pkl', 'rb') as f:
            svm_metrics = pickle.load(f)
        with open('models/rf_metrics.pkl', 'rb') as f:
            rf_metrics = pickle.load(f)
        logger.info(f"SVM Metrics: {svm_metrics}")
        logger.info(f"RF Metrics: {rf_metrics}")
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
    
    # Create a sample feature vector (like a real document)
    sample_features = np.random.normal(0.3, 0.15, 80)  # Authentic-like features
    sample_features = np.clip(sample_features, 0, 1)
    
    logger.info(f"Sample features shape: {sample_features.shape}")
    logger.info(f"Sample features mean: {np.mean(sample_features):.3f}")
    logger.info(f"Sample features std: {np.std(sample_features):.3f}")
    
    # Test prediction
    try:
        prediction, confidence, details = model.predict(sample_features)
        logger.info(f"Prediction: {prediction}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Details: {details}")
        
        # Check individual model predictions
        if 'svm_confidence' in details:
            logger.info(f"SVM confidence: {details['svm_confidence']:.3f}")
        if 'rf_confidence' in details:
            logger.info(f"RF confidence: {details['rf_confidence']:.3f}")
        if 'ensemble_confidence' in details:
            logger.info(f"Ensemble confidence: {details['ensemble_confidence']:.3f}")
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction()
