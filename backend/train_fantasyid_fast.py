"""
Fast FantasyID training with sample data and optimized features
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_basic_features(image_path):
    """Extract basic features quickly"""
    try:
        # Load and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic statistics
        features = [
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.min(gray),
            np.max(gray)
        ]
        
        # Histogram (8 bins for speed)
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Basic OCR features
        try:
            text = pytesseract.image_to_string(gray, config='--psm 6')
            text_length = len(text)
            word_count = len(text.split())
            digit_count = sum(c.isdigit() for c in text)
            
            features.extend([
                text_length,
                word_count,
                digit_count,
                len(set(text))  # unique characters
            ])
        except:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    except:
        return None

def main():
    # Set Tesseract path
    if os.name == 'nt':
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Load dataset
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    
    logger.info("Loading FantasyID dataset...")
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    
    # Use subset for faster training
    train_subset = train_df.sample(n=min(500, len(train_df)), random_state=42)
    test_subset = test_df.sample(n=min(200, len(test_df)), random_state=42)
    
    all_df = pd.concat([train_subset, test_subset], ignore_index=True)
    logger.info(f"Processing {len(all_df)} images...")
    
    features = []
    labels = []
    
    for idx, row in all_df.iterrows():
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        feat = extract_basic_features(image_path)
        if feat is not None:
            features.append(feat)
            labels.append(1 if row['is_attack'] else 0)
    
    if len(features) == 0:
        logger.error("No features extracted!")
        return
    
    X = np.array(features)
    y = np.array(labels)
    
    logger.info(f"Extracted {len(X)} samples with {len(X[0])} features")
    logger.info(f"Authentic: {np.sum(y == 0)}, Forged: {np.sum(y == 1)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    logger.info("Training SVM...")
    svm = SVC(probability=True, random_state=42, C=1.0, gamma='scale')
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred)
    }
    logger.info(f"SVM Metrics: {svm_metrics}")
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred)
    }
    logger.info(f"RF Metrics: {rf_metrics}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    with open('models/fantasyid_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('models/fantasyid_rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('models/fantasyid_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/fantasyid_svm_metrics.pkl', 'wb') as f:
        pickle.dump(svm_metrics, f)
    with open('models/fantasyid_rf_metrics.pkl', 'wb') as f:
        pickle.dump(rf_metrics, f)
    
    logger.info("FantasyID models saved successfully!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
