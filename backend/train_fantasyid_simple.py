"""
Simplified FantasyID training with grayscale, enhancement, OCR, and basic features
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import pytesseract
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """Apply grayscale conversion and enhancement"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for enhancement
        enhanced = cv2.equalizeHist(gray)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        
        # Edge detection to highlight forgery artifacts
        edges = cv2.Canny(clahe_enhanced, 50, 150)
        
        return gray, enhanced, clahe_enhanced, edges
    except Exception as e:
        logger.error(f"Error preprocessing {image_path}: {e}")
        return None

def extract_image_features(gray, enhanced, edges):
    """Extract key image features"""
    try:
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(enhanced),
            np.std(enhanced),
            np.var(enhanced),
            np.min(enhanced),
            np.max(enhanced)
        ])
        
        # Histogram features (16 bins)
        hist = cv2.calcHist([enhanced], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # Edge features
        edge_density = np.sum(edges > 0) / edges.size
        edge_variance = np.var(edges)
        features.extend([edge_density, edge_variance])
        
        # Texture features using Local Binary Pattern approximation
        def simple_lbp(img):
            h, w = img.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = img[i, j]
                    code = 0
                    for k in range(8):
                        x = i + [0, -1, -1, -1, 0, 1, 1, 1][k]
                        y = j + [1, 1, 0, -1, -1, -1, 0, 1][k]
                        if img[x, y] >= center:
                            code |= (1 << k)
                    lbp[i-1, j-1] = code
            return lbp
        
        lbp = simple_lbp(enhanced)
        lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        features.extend(lbp_hist)
        
        # Gradient features
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])
        
        return np.array(features)
    except Exception as e:
        logger.error(f"Error extracting image features: {e}")
        return np.zeros(40)  # Return zeros if error

def extract_ocr_features(image_path):
    """Extract OCR features for tampering detection"""
    try:
        img = Image.open(image_path)
        gray_img = img.convert('L')
        
        # Enhance contrast for better OCR
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced_img = enhancer.enhance(2.0)
        
        # Extract text
        text = pytesseract.image_to_string(enhanced_img, config='--psm 6')
        
        # Calculate features
        text_length = len(text)
        word_count = len(text.split())
        char_count = len(text.replace(' ', ''))
        
        # Word statistics
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        word_lengths = [len(word) for word in words]
        word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
        
        # Character analysis
        unique_chars = len(set(text))
        char_diversity = unique_chars / char_count if char_count > 0 else 0
        
        digit_count = sum(c.isdigit() for c in text)
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        uppercase_count = sum(c.isupper() for c in text)
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        punct_count = sum(c in '.,;:!?-()[]{}' for c in text)
        punct_ratio = punct_count / char_count if char_count > 0 else 0
        
        # OCR confidence
        try:
            data = pytesseract.image_to_data(enhanced_img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            ocr_confidence = np.mean(confidences) / 100 if confidences else 0.5
        except:
            ocr_confidence = 0.5
        
        # Inconsistency indicators
        suspicious_patterns = 0
        if text_length > 0:
            # Check for repeated characters
            for char in set(text):
                if text.count(char) > text_length * 0.3:
                    suspicious_patterns += 1
            # Check for unusual spacing
            if '  ' in text:
                suspicious_patterns += 1
        
        suspicious_patterns = min(suspicious_patterns, 3) / 3
        
        # Consistency scores
        text_consistency = 1.0 - (word_length_variance / 50)
        text_consistency = max(0, min(1, text_consistency))
        
        anomaly_score = (suspicious_patterns + (1 - text_consistency) + 
                       (1 - ocr_confidence)) / 3
        
        quality_score = (ocr_confidence + text_consistency) / 2
        
        return np.array([
            text_length, word_count, avg_word_length, char_diversity,
            digit_ratio, uppercase_ratio, punct_ratio, ocr_confidence,
            word_length_variance, suspicious_patterns, text_consistency,
            anomaly_score, quality_score
        ])
    except Exception as e:
        logger.error(f"Error extracting OCR features: {e}")
        return np.zeros(13)

def load_fantasyid_dataset(dataset_path):
    """Load and process FantasyID dataset"""
    logger.info("Loading FantasyID dataset...")
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    
    # Combine for processing
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(all_df)} images...")
    
    for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        # Preprocess
        result = preprocess_image(image_path)
        if result is None:
            continue
        
        gray, enhanced, clahe_enhanced, edges = result
        
        # Extract features
        img_features = extract_image_features(gray, enhanced, edges)
        ocr_features = extract_ocr_features(image_path)
        
        # Combine
        combined = np.concatenate([img_features, ocr_features])
        features.append(combined)
        labels.append(1 if row['is_attack'] else 0)
    
    return np.array(features), np.array(labels)

def main():
    # Set Tesseract path
    if os.name == 'nt':
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Load dataset
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    X, y = load_fantasyid_dataset(dataset_path)
    
    if len(X) == 0:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(X)} samples with {len(X[0])} features")
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
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
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
    
    logger.info("Models saved successfully!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
