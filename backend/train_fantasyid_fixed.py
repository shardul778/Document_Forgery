"""
Fixed FantasyID training with robust features
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

def extract_robust_features(image_path):
    """Extract robust features that distinguish authentic vs forged"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        
        features = []
        
        # 1. Edge features (key for forgery detection)
        edges_low = cv2.Canny(clahe_enhanced, 30, 80)
        edges_high = cv2.Canny(clahe_enhanced, 80, 150)
        
        edge_density_low = np.sum(edges_low > 0) / edges_low.size
        edge_density_high = np.sum(edges_high > 0) / edges_high.size
        edge_ratio = edge_density_high / (edge_density_low + 0.001)
        
        features.extend([edge_density_low, edge_density_high, edge_ratio])
        
        # 2. Texture features using GLCM (simpler than LBP)
        from skimage.feature import graycomatrix, graycoprops
        
        try:
            glcm = graycomatrix(
                clahe_enhanced,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )
            # graycoprops returns 2D arrays; take scalar values so the
            # final feature vector is 1D and np.array(features) is valid.
            contrast = float(graycoprops(glcm, "contrast")[0, 0])
            homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
            energy = float(graycoprops(glcm, "energy")[0, 0])
            correlation = float(graycoprops(glcm, "correlation")[0, 0])
            features.extend([contrast, homogeneity, energy, correlation])
        except Exception:
            # Fallback simple texture features
            texture_var = float(np.var(clahe_enhanced))
            texture_std = float(np.std(clahe_enhanced))
            texture_range = float(np.max(clahe_enhanced) - np.min(clahe_enhanced))
            features.extend([texture_var, texture_std, texture_range, texture_var])
        
        # 3. Noise analysis (forgery detection)
        # Laplacian for noise detection
        laplacian = cv2.Laplacian(clahe_enhanced, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        features.append(noise_variance)
        
        # 4. Gradient features (detect manipulation)
        grad_x = cv2.Sobel(clahe_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(clahe_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)
        grad_skew = np.mean((gradient_magnitude - grad_mean)**3) / (grad_std**3 + 0.001)
        
        features.extend([grad_mean, grad_std, grad_max, grad_skew])
        
        # 5. Statistical features
        stats = [
            np.mean(clahe_enhanced),
            np.std(clahe_enhanced),
            np.var(clahe_enhanced),
            np.min(clahe_enhanced),
            np.max(clahe_enhanced),
            np.median(clahe_enhanced),
            np.percentile(clahe_enhanced, 25),
            np.percentile(clahe_enhanced, 75)
        ]
        features.extend(stats)
        
        # 6. Histogram features
        hist = cv2.calcHist([clahe_enhanced], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 7. OCR features with forgery indicators
        try:
            pil_img = Image.fromarray(clahe_enhanced)
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
            
            # Check for common forgery patterns
            suspicious_patterns = 0
            if text_length > 0:
                # Unusual character repetition
                char_counts = {}
                for char in text:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char, count in char_counts.items():
                    if count > text_length * 0.2:  # 20% threshold
                        suspicious_patterns += 1
                
                # Inconsistent spacing
                if '  ' in text or '\t' in text:
                    suspicious_patterns += 1
                
                # Mixed case inconsistency
                has_upper = any(c.isupper() for c in text)
                has_lower = any(c.islower() for c in text)
                if has_upper and has_lower and word_count > 5:
                    # Check case consistency
                    upper_words = sum(1 for word in words if word.isupper())
                    lower_words = sum(1 for word in words if word.islower())
                    if abs(upper_words - lower_words) > word_count * 0.3:
                        suspicious_patterns += 1
            
            suspicious_patterns = min(suspicious_patterns, 4) / 4
            
            # Text quality score
            text_quality = (ocr_confidence + (1 - word_length_variance/50) + 
                           (1 - suspicious_patterns)) / 3
            
            features.extend([
                text_length, word_count, digit_count, unique_chars,
                ocr_confidence, word_length_variance, suspicious_patterns,
                text_quality
            ])
            
        except:
            # Default OCR features if extraction fails
            features.extend([0, 0, 0, 0, 0.5, 0, 0, 0.5])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def load_balanced_dataset(dataset_path):
    """Load balanced dataset with equal authentic/forged samples"""
    logger.info("Loading balanced FantasyID dataset...")
    
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    
    # Get balanced samples
    # Use a moderate number per class so training stays fast even with
    # expensive OCR / texture features.
    max_per_class = 250
    authentic_df = train_df[train_df['is_attack'] == False]
    forged_df = train_df[train_df['is_attack'] == True]
    authentic_samples = authentic_df.sample(
        n=min(max_per_class, len(authentic_df)), random_state=42
    )
    forged_samples = forged_df.sample(
        n=min(max_per_class, len(forged_df)), random_state=42
    )
    
    balanced_df = pd.concat([authentic_samples, forged_samples], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(balanced_df)} images...")
    
    for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        feat = extract_robust_features(image_path)
        if feat is not None:
            features.append(feat)
            labels.append(1 if row['is_attack'] else 0)
    
    return np.array(features), np.array(labels)

def main():
    # Set Tesseract path
    if os.name == 'nt':
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Load balanced dataset
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    X, y = load_balanced_dataset(dataset_path)
    
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
    
    # Train optimized SVM
    logger.info("Training optimized SVM...")
    svm = SVC(probability=True, random_state=42, C=5.0, gamma='scale', kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred)
    }
    logger.info(f"Optimized SVM Metrics: {svm_metrics}")
    
    # Train optimized Random Forest
    logger.info("Training optimized Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=12,
                             min_samples_split=4, min_samples_leaf=2, max_features='sqrt')
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred)
    }
    logger.info(f"Optimized RF Metrics: {rf_metrics}")
    
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
    
    logger.info("Optimized FantasyID models saved successfully!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
