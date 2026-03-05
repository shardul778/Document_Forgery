"""
Improved FantasyID training with better discriminative features
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

def extract_discriminative_features(image_path):
    """Extract features that better distinguish authentic vs forged"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_enhanced = clahe.apply(gray)
        
        features = []
        
        # 1. Edge-based features (forgery detection)
        edges = cv2.Canny(clahe_enhanced, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        features.extend([edge_density, edge_mean, edge_std])
        
        # 2. Texture features (LBP for forgery detection)
        def compute_lbp(img, radius=2, n_points=16):
            h, w = img.shape
            lbp = np.zeros((h-2*radius, w-2*radius), dtype=np.uint8)
            for i in range(radius, h-radius):
                for j in range(radius, w-radius):
                    center = img[i, j]
                    code = 0
                    for k in range(n_points):
                        x = i + radius * np.cos(2*np.pi*k/n_points)
                        y = j + radius * np.sin(2*np.pi*k/n_points)
                        if img[int(x), int(y)] >= center:
                            code |= (1 << k)
                    lbp[i-radius, j-radius] = code
            return lbp
        
        lbp = compute_lbp(clahe_enhanced)
        lbp_hist, _ = np.histogram(lbp, bins=16, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        features.extend(lbp_hist[:8])  # First 8 LBP features
        
        # 3. Frequency domain features (detect manipulation)
        f_transform = np.fft.fft2(clahe_enhanced)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High-frequency content (forgery indicator)
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        high_freq_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
        high_freq_energy = np.sum(high_freq_region**2)
        total_energy = np.sum(magnitude**2)
        high_freq_ratio = high_freq_energy / total_energy
        features.append(high_freq_ratio)
        
        # 4. Noise analysis (forgery leaves different noise patterns)
        def estimate_noise(image):
            H, W = image.shape
            M = [[1, -2, 1],
                   [-2, 4, -2],
                   [1, -2, 1]]
            sigma = np.sum(np.array(M)**2)
            noise_variance = np.sum(cv2.filter2D(image, -1, np.array(M))**2) / sigma
            return noise_variance
        
        noise_var = estimate_noise(clahe_enhanced)
        features.append(noise_var)
        
        # 5. Gradient features (detect inconsistencies)
        grad_x = cv2.Sobel(clahe_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(clahe_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        gradient_max = np.max(gradient_magnitude)
        features.extend([gradient_mean, gradient_std, gradient_max])
        
        # 6. Statistical features
        features.extend([
            np.mean(clahe_enhanced),
            np.std(clahe_enhanced),
            np.var(clahe_enhanced),
            np.min(clahe_enhanced),
            np.max(clahe_enhanced),
            np.median(clahe_enhanced)
        ])
        
        # 7. Histogram features
        hist = cv2.calcHist([clahe_enhanced], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 8. OCR features (text inconsistencies)
        try:
            pil_img = Image.fromarray(clahe_enhanced)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_img = enhancer.enhance(2.0)
            
            text = pytesseract.image_to_string(enhanced_img, config='--psm 6')
            
            text_length = len(text)
            word_count = len(text.split())
            digit_count = sum(c.isdigit() for c in text)
            unique_chars = len(set(text))
            
            # Text quality indicators
            try:
                data = pytesseract.image_to_data(enhanced_img, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                ocr_confidence = np.mean(confidences) / 100 if confidences else 0.5
            except:
                ocr_confidence = 0.5
            
            # Inconsistency indicators
            words = text.split()
            word_lengths = [len(word) for word in words]
            word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
            
            # Suspicious patterns
            suspicious_patterns = 0
            if text_length > 0:
                # Repeated characters
                for char in set(text):
                    if text.count(char) > text_length * 0.3:
                        suspicious_patterns += 1
                # Double spaces
                if '  ' in text:
                    suspicious_patterns += 1
            
            suspicious_patterns = min(suspicious_patterns, 3) / 3
            
            features.extend([
                text_length, word_count, digit_count, unique_chars,
                ocr_confidence, word_length_variance, suspicious_patterns
            ])
            
        except:
            features.extend([0, 0, 0, 0, 0.5, 0, 0])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def load_balanced_dataset(dataset_path):
    """Load balanced dataset with equal authentic/forged samples"""
    logger.info("Loading balanced FantasyID dataset...")
    
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    
    # Get equal samples from each class
    authentic_samples = train_df[train_df['is_attack'] == False].sample(n=300, random_state=42)
    forged_samples = train_df[train_df['is_attack'] == True].sample(n=300, random_state=42)
    
    # Combine
    balanced_df = pd.concat([authentic_samples, forged_samples], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(balanced_df)} images...")
    
    for idx, row in tqdm(balanced_df.iterrows(), total=len(balanced_df)):
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        feat = extract_discriminative_features(image_path)
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
    
    # Train SVM with better parameters
    logger.info("Training improved SVM...")
    svm = SVC(probability=True, random_state=42, C=10, gamma='scale', kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred)
    }
    logger.info(f"Improved SVM Metrics: {svm_metrics}")
    
    # Train Random Forest with better parameters
    logger.info("Training improved Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, 
                             min_samples_split=5, min_samples_leaf=2)
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred)
    }
    logger.info(f"Improved RF Metrics: {rf_metrics}")
    
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
    
    logger.info("Improved FantasyID models saved successfully!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
