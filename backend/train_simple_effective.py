"""
Simple but effective FantasyID training
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

def extract_simple_features(image_path):
    """Extract simple but effective features"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic enhancement
        enhanced = cv2.equalizeHist(gray)
        
        features = []
        
        # 1. Basic statistics
        features.extend([
            np.mean(enhanced),
            np.std(enhanced),
            np.var(enhanced),
            np.min(enhanced),
            np.max(enhanced)
        ])
        
        # 2. Edge features (very important for forgery)
        edges = cv2.Canny(enhanced, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        features.extend([edge_density, edge_mean, edge_std])
        
        # 3. Histogram features
        hist = cv2.calcHist([enhanced], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 4. Texture features (simple but effective)
        # Local Binary Pattern approximation
        def simple_lbp(img):
            h, w = img.shape
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = img[i, j]
                    code = 0
                    # 8 neighbors
                    neighbors = [
                        img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                        img[i, j-1], img[i, j+1],
                        img[i+1, j-1], img[i+1, j], img[i+1, j+1]
                    ]
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    lbp[i-1, j-1] = code
            return lbp
        
        lbp = simple_lbp(enhanced)
        lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        features.extend(lbp_hist)
        
        # 5. Noise features
        # Gaussian noise estimation
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        noise_map = cv2.filter2D(enhanced, -1, kernel)
        noise_var = np.var(noise_map)
        features.append(noise_var)
        
        # 6. Gradient features
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)
        features.extend([grad_mean, grad_std, grad_max])
        
        # 7. OCR features (critical for ID cards)
        try:
            pil_img = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_img = enhancer.enhance(2.0)
            
            text = pytesseract.image_to_string(enhanced_img, config='--psm 6')
            
            # Basic text features
            text_length = len(text)
            word_count = len(text.split())
            digit_count = sum(c.isdigit() for c in text)
            unique_chars = len(set(text))
            
            # OCR confidence
            try:
                data = pytesseract.image_to_data(enhanced_img, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                ocr_confidence = np.mean(confidences) / 100 if confidences else 0.5
            except:
                ocr_confidence = 0.5
            
            # Forgery indicators in text
            words = text.split()
            word_lengths = [len(word) for word in words]
            word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
            
            # Check for suspicious patterns
            suspicious_patterns = 0
            if text_length > 0:
                # Repeated characters (common in forgeries)
                char_counts = {}
                for char in text:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char, count in char_counts.items():
                    if count > text_length * 0.15:  # 15% threshold
                        suspicious_patterns += 1
                
                # Inconsistent spacing
                if '  ' in text or '\t' in text:
                    suspicious_patterns += 1
                
                # Mixed case issues
                has_upper = any(c.isupper() for c in text)
                has_lower = any(c.islower() for c in text)
                if has_upper and has_lower and word_count > 3:
                    upper_words = sum(1 for word in words if word.isupper())
                    lower_words = sum(1 for word in words if word.islower())
                    if abs(upper_words - lower_words) > word_count * 0.2:
                        suspicious_patterns += 1
            
            suspicious_patterns = min(suspicious_patterns, 3) / 3
            
            # Text quality score
            text_quality = (ocr_confidence + (1 - word_length_variance/30) + 
                           (1 - suspicious_patterns)) / 3
            
            features.extend([
                text_length, word_count, digit_count, unique_chars,
                ocr_confidence, word_length_variance, suspicious_patterns,
                text_quality
            ])
            
        except:
            # Default OCR features
            features.extend([0, 0, 0, 0, 0.5, 0, 0, 0.5])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def main():
    # Set Tesseract path
    if os.name == 'nt':
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Load dataset
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    
    # Create balanced dataset
    authentic = train_df[train_df['is_attack'] == False].sample(n=300, random_state=42)
    forged = train_df[train_df['is_attack'] == True].sample(n=300, random_state=42)
    
    combined = pd.concat([authentic, forged], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(combined)} images...")
    
    for idx, row in tqdm(combined.iterrows(), total=len(combined)):
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        feat = extract_simple_features(image_path)
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
    
    # Train SVM with strong regularization
    logger.info("Training SVM...")
    svm = SVC(probability=True, random_state=42, C=0.1, gamma='scale', kernel='rbf')
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
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8,
                             min_samples_split=5, min_samples_leaf=2)
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
    
    logger.info("Simple effective models saved!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
