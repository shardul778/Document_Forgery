"""
Train robust model for real-world document forgery detection
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
    """Extract robust features optimized for real-world documents"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing pipeline
        # 1. Noise reduction with multiple methods
        denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        denoised_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Contrast enhancement with multiple methods
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_clahe = clahe.apply(denoised_bilateral)
        
        # Histogram equalization
        enhanced_hist = cv2.equalizeHist(denoised_gaussian)
        
        # 3. Sharpening with different kernels
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened_clahe = cv2.filter2D(enhanced_clahe, -1, kernel_sharp)
        
        kernel_mild = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened_hist = cv2.filter2D(enhanced_hist, -1, kernel_mild)
        
        features = []
        
        # 1. Image Quality Features (Critical for real documents)
        # Sharpness measures
        laplacian_var_clahe = cv2.Laplacian(sharpened_clahe, cv2.CV_64F).var()
        laplacian_var_hist = cv2.Laplacian(sharpened_hist, cv2.CV_64F).var()
        features.extend([laplacian_var_clahe, laplacian_var_hist])
        
        # Noise levels
        noise_bilateral = cv2.subtract(denoised_bilateral, gray)
        noise_gaussian = cv2.subtract(denoised_gaussian, gray)
        noise_level_bilateral = np.std(noise_bilateral)
        noise_level_gaussian = np.std(noise_gaussian)
        features.extend([noise_level_bilateral, noise_level_gaussian])
        
        # Contrast and brightness measures
        contrast_clahe = enhanced_clahe.std()
        contrast_hist = enhanced_hist.std()
        brightness_clahe = enhanced_clahe.mean()
        brightness_hist = enhanced_hist.mean()
        features.extend([contrast_clahe, contrast_hist, brightness_clahe, brightness_hist])
        
        # 2. Advanced Edge Features
        # Multi-level edge detection
        edges_low = cv2.Canny(enhanced_clahe, 30, 100)
        edges_medium = cv2.Canny(enhanced_clahe, 50, 150)
        edges_high = cv2.Canny(enhanced_clahe, 100, 200)
        
        edge_density_low = np.sum(edges_low > 0) / edges_low.size
        edge_density_medium = np.sum(edges_medium > 0) / edges_medium.size
        edge_density_high = np.sum(edges_high > 0) / edges_high.size
        
        # Edge consistency (forged docs have inconsistent edges)
        edge_ratio_1 = edge_density_medium / (edge_density_low + 0.001)
        edge_ratio_2 = edge_density_high / (edge_density_medium + 0.001)
        
        # Edge statistics
        edge_mean_low = np.mean(edges_low)
        edge_std_low = np.std(edges_low)
        edge_mean_high = np.mean(edges_high)
        edge_std_high = np.std(edges_high)
        
        features.extend([
            edge_density_low, edge_density_medium, edge_density_high,
            edge_ratio_1, edge_ratio_2,
            edge_mean_low, edge_std_low, edge_mean_high, edge_std_high
        ])
        
        # 3. Texture Analysis (Multiple methods)
        # Local Binary Pattern
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
        
        lbp_clahe = compute_lbp(enhanced_clahe)
        lbp_hist_clahe, _ = np.histogram(lbp_clahe, bins=16, range=(0, 256))
        lbp_hist_clahe = lbp_hist_clahe.astype(float) / lbp_hist_clahe.sum()
        
        # GLCM features
        try:
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(enhanced_clahe, distances=[1, 2], angles=[0, np.pi/4], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
        except:
            # Fallback texture features
            texture_var = np.var(enhanced_clahe)
            texture_std = np.std(enhanced_clahe)
            texture_range = np.max(enhanced_clahe) - np.min(enhanced_clahe)
            contrast = texture_var / 1000
            homogeneity = 1 - (texture_std / 255)
            energy = 1 / (1 + texture_var)
            correlation = 0.5
        
        features.extend(lbp_hist_clahe[:8])  # First 8 LBP features
        features.extend([contrast, homogeneity, energy, correlation])
        
        # 4. Frequency Domain Analysis
        # FFT analysis for digital manipulation
        f_transform = np.fft.fft2(enhanced_clahe)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High-frequency content analysis
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # Different frequency regions
        low_freq_mask = np.zeros_like(magnitude)
        low_freq_mask[center_h-20:center_h+20, center_w-20:center_w+20] = 1
        
        mid_freq_mask = np.zeros_like(magnitude)
        mid_freq_mask[center_h-40:center_h+40, center_w-40:center_w+40] = 1
        mid_freq_mask[center_h-20:center_h+20, center_w-20:center_w+20] = 0
        
        high_freq_mask = 1 - low_freq_mask - mid_freq_mask
        
        low_freq_energy = np.sum(magnitude * low_freq_mask)
        mid_freq_energy = np.sum(magnitude * mid_freq_mask)
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude)
        
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
        
        features.extend([low_freq_ratio, mid_freq_ratio, high_freq_ratio])
        
        # 5. Statistical Features (Comprehensive)
        stats_clahe = [
            np.mean(enhanced_clahe),
            np.std(enhanced_clahe),
            np.var(enhanced_clahe),
            np.min(enhanced_clahe),
            np.max(enhanced_clahe),
            np.median(enhanced_clahe),
            np.percentile(enhanced_clahe, 25),
            np.percentile(enhanced_clahe, 75),
            np.percentile(enhanced_clahe, 10),
            np.percentile(enhanced_clahe, 90)
        ]
        
        stats_hist = [
            np.mean(enhanced_hist),
            np.std(enhanced_hist),
            np.var(enhanced_hist),
            np.min(enhanced_hist),
            np.max(enhanced_hist),
            np.median(enhanced_hist),
            np.percentile(enhanced_hist, 25),
            np.percentile(enhanced_hist, 75),
            np.percentile(enhanced_hist, 10),
            np.percentile(enhanced_hist, 90)
        ]
        
        features.extend(stats_clahe)
        features.extend(stats_hist)
        
        # 6. Histogram Features (Detailed)
        hist_clahe = cv2.calcHist([enhanced_clahe], [0], None, [32], [0, 256])
        hist_clahe = hist_clahe.flatten() / hist_clahe.sum()
        
        hist_hist = cv2.calcHist([enhanced_hist], [0], None, [32], [0, 256])
        hist_hist = hist_hist.flatten() / hist_hist.sum()
        
        features.extend(hist_clahe)
        features.extend(hist_hist)
        
        # 7. Gradient Features (Enhanced)
        grad_x = cv2.Sobel(enhanced_clahe, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced_clahe, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)
        grad_min = np.min(gradient_magnitude)
        grad_median = np.median(gradient_magnitude)
        
        # Gradient skewness and kurtosis
        grad_skew = np.mean((gradient_magnitude - grad_mean)**3) / (grad_std**3 + 0.001)
        grad_kurtosis = np.mean((gradient_magnitude - grad_mean)**4) / (grad_std**4 + 0.001)
        
        features.extend([grad_mean, grad_std, grad_max, grad_min, grad_median, grad_skew, grad_kurtosis])
        
        # 8. Advanced OCR Features
        try:
            # Multiple preprocessing for OCR
            pil_img_clahe = Image.fromarray(enhanced_clahe)
            pil_img_hist = Image.fromarray(enhanced_hist)
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(pil_img_clahe)
            enhanced_img = enhancer.enhance(2.0)
            
            sharpener = ImageEnhance.Sharpness(enhanced_img)
            sharp_img = sharpener.enhance(2.0)
            
            text_clahe = pytesseract.image_to_string(sharp_img, config='--psm 6')
            text_hist = pytesseract.image_to_string(pil_img_hist, config='--psm 6')
            
            # Combine text from both methods
            combined_text = text_clahe + " " + text_hist
            
            # Basic text features
            text_length = len(combined_text)
            word_count = len(combined_text.split())
            digit_count = sum(c.isdigit() for c in combined_text)
            unique_chars = len(set(combined_text))
            
            # OCR confidence
            try:
                data = pytesseract.image_to_data(sharp_img, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                ocr_confidence = np.mean(confidences) / 100 if confidences else 0.5
            except:
                ocr_confidence = 0.5
            
            # Advanced forgery indicators in text
            words = combined_text.split()
            word_lengths = [len(word) for word in words]
            word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
            word_length_mean = np.mean(word_lengths) if word_lengths else 0
            
            # Character analysis
            char_counts = {}
            for char in combined_text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Suspicious patterns
            suspicious_patterns = 0
            
            # Character repetition (common in forgeries)
            for char, count in char_counts.items():
                if count > text_length * 0.15:  # 15% threshold
                    suspicious_patterns += 1
            
            # Inconsistent spacing
            if '  ' in combined_text or '\t' in combined_text:
                suspicious_patterns += 1
            
            # Mixed case issues
            has_upper = any(c.isupper() for c in combined_text)
            has_lower = any(c.islower() for c in combined_text)
            if has_upper and has_lower and word_count > 3:
                upper_words = sum(1 for word in words if word.isupper())
                lower_words = sum(1 for word in words if word.islower())
                if abs(upper_words - lower_words) > word_count * 0.3:
                    suspicious_patterns += 1
            
            # Unusual characters
            unusual_chars = sum(1 for c in combined_text if ord(c) > 127)
            if unusual_chars > text_length * 0.1:
                suspicious_patterns += 1
            
            # Numeric patterns (important for ID cards)
            numeric_sequences = 0
            for word in words:
                if word.isdigit() and len(word) >= 4:
                    numeric_sequences += 1
            
            # Text quality indicators
            text_quality = (ocr_confidence + (1 - word_length_variance/30) + 
                           (1 - suspicious_patterns/4)) / 3
            
            # Document type indicators
            has_numbers = digit_count > 0
            has_letters = any(c.isalpha() for c in combined_text)
            has_mixed = has_numbers and has_letters
            has_long_words = any(len(word) > 10 for word in words)
            
            features.extend([
                text_length, word_count, digit_count, unique_chars,
                ocr_confidence, word_length_variance, word_length_mean,
                suspicious_patterns, text_quality, numeric_sequences,
                int(has_numbers), int(has_letters), int(has_mixed), int(has_long_words)
            ])
            
        except:
            # Default OCR features if extraction fails
            features.extend([0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def load_fantasyid_dataset(dataset_path):
    """Load FantasyID dataset with robust features"""
    logger.info("Loading FantasyID dataset with robust features...")
    
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    
    # Create balanced dataset with more samples
    authentic_samples = train_df[train_df['is_attack'] == False].sample(n=500, random_state=42)
    forged_samples = train_df[train_df['is_attack'] == True].sample(n=500, random_state=42)
    
    combined = pd.concat([authentic_samples, forged_samples], ignore_index=True)
    
    features = []
    labels = []
    
    logger.info(f"Processing {len(combined)} images with robust features...")
    
    for idx, row in tqdm(combined.iterrows(), total=len(combined)):
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
    
    # Train SVM with robust parameters
    logger.info("Training robust SVM...")
    svm = SVC(probability=True, random_state=42, C=5.0, gamma='scale', kernel='rbf')
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred),
        'recall': recall_score(y_test, svm_pred),
        'f1_score': f1_score(y_test, svm_pred)
    }
    logger.info(f"Robust SVM Metrics: {svm_metrics}")
    
    # Train Random Forest with robust parameters
    logger.info("Training robust Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15,
                             min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred)
    }
    logger.info(f"Robust RF Metrics: {rf_metrics}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    with open('models/robust_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    with open('models/robust_rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('models/robust_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/robust_svm_metrics.pkl', 'wb') as f:
        pickle.dump(svm_metrics, f)
    with open('models/robust_rf_metrics.pkl', 'wb') as f:
        pickle.dump(rf_metrics, f)
    
    logger.info("Robust models saved successfully!")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
