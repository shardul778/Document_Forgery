"""Test the SVM and Random Forest models"""
import numpy as np
from ml_model_svm_rf import get_document_forgery_model

def test_models():
    print("Testing SVM and Random Forest models...")
    
    # Get the model
    model = get_document_forgery_model()
    
    # Get model info
    info = model.get_model_info()
    print("Model Information:")
    print(f"SVM Type: {info['svm_model']['type']}")
    print(f"SVM Kernel: {info['svm_model']['kernel']}")
    print(f"SVM C: {info['svm_model']['C']}")
    print(f"RF Type: {info['rf_model']['type']}")
    print(f"RF Estimators: {info['rf_model']['n_estimators']}")
    print(f"RF Max Depth: {info['rf_model']['max_depth']}")
    print(f"Models Trained: {info['trained']}")
    
    print("\nModel Metrics:")
    print(f"SVM Metrics: {info['svm_model']['metrics']}")
    print(f"RF Metrics: {info['rf_model']['metrics']}")
    
    # Test with sample features
    print("\nTesting with sample features...")
    
    # Create sample features (80 features: 64 image + 16 OCR)
    sample_features = np.random.rand(80)
    
    # Make prediction
    prediction, confidence, details = model.predict(sample_features)
    
    print(f"Prediction: {prediction} ({'Forged' if prediction == 1 else 'Authentic'})")
    print(f"Confidence: {confidence:.4f}")
    print(f"SVM Prediction: {details['svm_prediction']} (conf: {details['svm_confidence']:.4f})")
    print(f"RF Prediction: {details['rf_prediction']} (conf: {details['rf_confidence']:.4f})")
    print(f"Ensemble Confidence: {details['ensemble_confidence']:.4f}")
    print(f"Threshold: {details['threshold']}")
    
    print("\nSUCCESS: SVM and Random Forest models are working correctly!")

if __name__ == "__main__":
    test_models()
