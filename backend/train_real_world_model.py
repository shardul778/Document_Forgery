"""
Train model for real-world document forgery detection
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

def extract_real_world_features(image_path):
    """Extract features optimized for real-world documents"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for real documents
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        features = []
        
        # 1. Image quality features (important for real documents)
        # Sharpness measure
        laplacian_var = cv2.Laplacian(sharpened, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        # Noise level
        noise = cv2.subtract(sharpened, enhanced)
        noise_level = np.std(noise)
        features.append(noise_level)
        
        # Contrast measure
        contrast = enhanced.std()
        features.append(contrast)
        
        # Brightness measure
        brightness = enhanced.mean()
        features.append(brightness)
        
        # 2. Edge features (forgery detection)
        edges_low = cv2.Canny(enhanced, 30, 100)
        edges_high = cv2.Canny(enhanced, 100, 200)
        
        edge_density_low = np.sum(edges_low > 0) / edges_low.size
        edge_density_high = np.sum(edges_high > 0) / edges_high.size
        edge_ratio = edge_density_high / (edge_density_low + 0.001)
        
        features.extend([edge_density_low, edge_density_high, edge_ratio])
        
        # 3. Texture features (detect manipulation)
        # Local Binary Pattern approximation
        def compute_lbp(img, radius=1):
            h, w = img.shape
            lbp = np.zeros((h-2*radius, w-2*radius), dtype=np.uint8)
            for i in range(radius, h-radius):
                for j in range(radius, w-radius):
                    center = img[i, j]
                    code = 0
                    neighbors = [
                        img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                        img[i, j-1], img[i, j+1],
                        img[i+1, j-1], img[i+1, j], img[i+1, j+1]
                    ]
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    lbp[i-radius, j-radius] = code
            return lbp
        
        lbp = compute_lbp(enhanced)
        lbp_hist, _ = np.histogram(lbp, bins=16, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        features.extend(lbp_hist[:8])  # First 8 LBP features
        
        # 4. Frequency domain features (detect digital manipulation)
        f_transform = np.fft.fft2(enhanced)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High-frequency content (forgery indicator)
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # High frequency region
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_h-30:center_h+30, center_w-30:center_w+30] = 1
        high_freq_energy = np.sum(magnitude * (1 - high_freq_mask))
        total_energy = np.sum(magnitude)
        high_freq_ratio = high_freq_energy / total_energy
        features.append(high_freq_ratio)
        
        # 5. Statistical features
        stats = [
            np.mean(enhanced),
            np.std(enhanced),
            np.var(enhanced),
            np.min(enhanced),
            np.max(enhanced),
            np.median(enhanced),
            np.percentile(enhanced, 25),
            np.percentile(enhanced, 75)
        ]
        features.extend(stats)
        
        # 6. Histogram features
        hist = cv2.calcHist([enhanced], [0], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # 7. Gradient features (detect inconsistencies)
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)
        grad_skew = np.mean((gradient_magnitude - grad_mean)**3) / (grad_std**3 + 0.001)
        
        features.extend([grad_mean, grad_std, grad_max, grad_skew])
        
        # 8. OCR features with forgery detection
        try:
            # Multiple preprocessing for OCR
            pil_img = Image.fromarray(enhanced)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_img = enhancer.enhance(2.0)
            
            # Enhance sharpness
            sharpener = ImageEnhance.Sharpness(enhanced_img)
            sharp_img = sharpener.enhance(2.0)
            
            text = pytesseract.image_to_string(sharp_img, config='--psm 6')
            
            # Basic text features
            text_length = len(text)
            word_count = len(text.split())
            digit_count = sum(c.isdigit() for c in text)
            unique_chars = len(set(text))
            
            # OCR confidence
            try:
                data = pytesseract.image_to_data(sharp_img, output_type=pytesseract.Output.DICT)
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
                # Character repetition (common in forgeries)
                char_counts = {}
                for char in text:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                for char, count in char_counts.items():
                    if count > text_length * 0.2:  # 20% threshold
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
                    if abs(upper_words - lower_words) > word_count * 0.3:
                        suspicious_patterns += 1
                
                # Unusual characters
                unusual_chars = sum(1 for c in text if ord(c) > 127)
                if unusual_chars > text_length * 0.1:
                    suspicious_patterns += 1
            
            suspicious_patterns = min(suspicious_patterns, 4) / 4
            
            # Text quality score
            text_quality = (ocr_confidence + (1 - word_length_variance/30) + 
                           (1 - suspicious_patterns)) / 3
            
            # Document type indicators
            has_numbers = digit_count > 0
            has_letters = any(c.isalpha() for c in text)
            has_mixed = has_numbers and has_letters
            
            features.extend([
                text_length, word_count, digit_count, unique_chars,
                ocr_confidence, word_length_variance, suspicious_patterns,
                text_quality, int(has_numbers), int(has_letters), int(has_mixed)
            ])
            
        except:
            # Default OCR features if extraction fails
            features.extend([0, 0, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def load_fantasyid_dataset(dataset_path):
    """Load FantasyID dataset for training"""
    logger.info("Loading FantasyID dataset...")
    
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    
    # Create balanced dataset
    authentic_samples = train_df[train_df['is_attack'] == False].sample(n=400, random_state=42)
    forged_samples = train_df[train_df['is_attack'] == True].sample(n=400, random_state=42)
    
    combined = pd.concat([authentic_samples, forged_samples], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(combined)} images...")
    
    for idx, row in tqdm(combined.iterrows(), total=len(combined)):
        image_path = os.path.join(dataset_path, row['path'])
        
        if not os.path.exists(image_path):
            continue
        
        feat = extract_real_world_features(image_path)
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
    
    # Load dataset
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    X, y = load_fantasyid_dataset(dataset_path)
    
    if len(X) == 0:
        logger.error("No features extracted!")
        return
    
    logger.info(f"Extracted {len(X)} samples with {len(X[0])} features")
    logger.info(f"Authentic: {np.sum(y == 0)}, Forged: {np.sum(y == 1)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM with conservative parameters
    logger.info("Training conservative SVM...")
    svm = SVC(probability=True, random_state=42, C=1.0, gamma='scale', kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred)
    }
    logger.info(f"Conservative SVM Metrics: {svm_metrics}")
    
    # Train Random Forest with conservative parameters
    logger.info("Training conservative Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10,
                             min_samples_split=10, min_samples_leaf=5, max_features='sqrt')
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred)
    }
    logger.info(f"Conservative RF Metrics: {rf_metrics}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    with open('models/real_world_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('models/real_world_rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('models/real_world_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/real_world_svm_metrics.pkl', 'wb') as f:
        pickle.dump(svm_metrics, f)
    with open('models/real_world_rf_metrics.pkl', 'wb') as f:
        pickle.dump(rf_metrics, f)
    
    logger.info("Real-world optimized models saved!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
