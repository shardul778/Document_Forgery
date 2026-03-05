"""
Debug FantasyID model predictions
"""
import numpy as np
from ml_model_fantasyid import get_fantasyid_model
import os

def debug_fantasyid():
    """Debug why forged images show as authentic"""
    print("=== DEBUGGING FANTASYID MODEL ===")
    
    model = get_fantasyid_model()
    
    # Test authentic image
    authentic_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID\train\bonafide\huawei\arabic-003_03.jpg"
    if os.path.exists(authentic_path):
        pred, conf, details = model.predict(authentic_path)
        print(f"AUTHENTIC IMAGE:")
        print(f"  Prediction: {pred} (0=authentic, 1=forged)")
        print(f"  Confidence: {conf:.3f}")
        print(f"  SVM Conf: {details.get('svm_confidence', 'N/A'):.3f}")
        print(f"  RF Conf: {details.get('rf_confidence', 'N/A'):.3f}")
        print(f"  Ensemble: {details.get('ensemble_confidence', 'N/A'):.3f}")
        print()
    
    # Test forged image
    forged_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID\train\attack\digital_1\huawei\arabic-003_03-d1.jpg"
    if os.path.exists(forged_path):
        pred, conf, details = model.predict(forged_path)
        print(f"FORGED IMAGE:")
        print(f"  Prediction: {pred} (0=authentic, 1=forged)")
        print(f"  Confidence: {conf:.3f}")
        print(f"  SVM Conf: {details.get('svm_confidence', 'N/A'):.3f}")
        print(f"  RF Conf: {details.get('rf_confidence', 'N/A'):.3f}")
        print(f"  Ensemble: {details.get('ensemble_confidence', 'N/A'):.3f}")
        print(f"  Threshold: {details.get('threshold', 'N/A')}")
        print()
    
    print("=== ANALYSIS ===")
    print("If forged image confidence < 0.7, model needs:")
    print("1. Lower threshold")
    print("2. Better training")
    print("3. More discriminative features")

if __name__ == "__main__":
    debug_fantasyid()
