"""
Train document forgery detection model on FantasyID dataset
Uses grayscale conversion, enhancement, OCR, and texture analysis
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
from sklearn.model_selection import GridSearchCV
import pickle
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FantasyIDTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.svm_model = None
        self.rf_model = None
        
    def preprocess_image(self, image_path):
        """Apply grayscale conversion and enhancement to highlight forgery artifacts"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for enhancement
            enhanced = cv2.equalizeHist(gray)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_enhanced = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(clahe_enhanced, (3, 3), 0)
            
            # Edge detection to highlight forgery artifacts
            edges = cv2.Canny(blurred, 50, 150)
            
            # Texture analysis using Gabor filters
            def apply_gabor_filters(img):
                filters = []
                for theta in [0, 45, 90, 135]:
                    for frequency in [0.1, 0.3, 0.5]:
                        kern = cv2.getGaborKernel((15, 15), 3, np.radians(theta), 
                                                frequency, 0.5, 0, ktype=cv2.CV_32F)
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
                        filters.append(fimg)
                return filters
            
            gabor_filters = apply_gabor_filters(clahe_enhanced)
            
            return {
                'original': img,
                'gray': gray,
                'enhanced': enhanced,
                'clahe': clahe_enhanced,
                'blurred': blurred,
                'edges': edges,
                'gabor_filters': gabor_filters
            }
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def extract_image_features(self, processed_img):
        """Extract comprehensive image features for forgery detection"""
        try:
            if processed_img is None:
                return None
                
            features = []
            
            # Basic statistical features from enhanced image
            enhanced = processed_img['enhanced']
            features.extend([
                np.mean(enhanced),
                np.std(enhanced),
                np.var(enhanced),
                np.min(enhanced),
                np.max(enhanced),
                np.median(enhanced)
            ])
            
            # Histogram features
            hist = cv2.calcHist([enhanced], [0], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend(hist)
            
            # Edge density features
            edges = processed_img['edges']
            edge_density = np.sum(edges > 0) / edges.size
            edge_variance = np.var(edges)
            features.extend([edge_density, edge_variance])
            
            # Texture features from Gabor filters
            gabor_filters = processed_img['gabor_filters']
            for gabor_img in gabor_filters[:4]:  # Use first 4 filters
                features.extend([
                    np.mean(gabor_img),
                    np.std(gabor_img),
                    np.var(gabor_img)
                ])
            
            # Local Binary Pattern (LBP) features for texture
            def compute_lbp(img, radius=1, n_points=8):
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(img, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2)
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-7)
                return hist
            
            try:
                lbp_hist = compute_lbp(enhanced)
                features.extend(lbp_hist[:10])  # Take first 10 LBP features
            except:
                # Fallback if skimage not available
                features.extend([0] * 10)
            
            # Noise analysis
            noise = cv2.absdiff(processed_img['gray'], processed_img['blurred'])
            noise_features = [
                np.mean(noise),
                np.std(noise),
                np.var(noise)
            ]
            features.extend(noise_features)
            
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
            return None
    
    def extract_ocr_features(self, image_path):
        """Extract textual content and analyze inconsistencies for tampering detection"""
        try:
            # Load and preprocess for OCR
            img = Image.open(image_path)
            
            # Convert to grayscale for better OCR
            gray_img = img.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_img)
            enhanced_img = enhancer.enhance(2.0)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(enhanced_img, config='--psm 6')
            
            # Calculate text features
            text_length = len(text)
            word_count = len(text.split())
            char_count = len(text.replace(' ', ''))
            
            # Average word length
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Character diversity
            unique_chars = len(set(text))
            char_diversity = unique_chars / char_count if char_count > 0 else 0
            
            # Digit ratio (important for ID cards)
            digit_count = sum(c.isdigit() for c in text)
            digit_ratio = digit_count / char_count if char_count > 0 else 0
            
            # Uppercase ratio
            uppercase_count = sum(c.isupper() for c in text)
            uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
            
            # Punctuation ratio
            punct_count = sum(c in '.,;:!?-()[]{}' for c in text)
            punct_ratio = punct_count / char_count if char_count > 0 else 0
            
            # OCR confidence (if available)
            try:
                data = pytesseract.image_to_data(enhanced_img, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                ocr_confidence = np.mean(confidences) / 100 if confidences else 0.5
            except:
                ocr_confidence = 0.5
            
            # Word length variance (inconsistency indicator)
            word_lengths = [len(word) for word in words]
            word_length_variance = np.var(word_lengths) if len(word_lengths) > 1 else 0
            
            # Suspicious patterns (common in forgeries)
            suspicious_patterns = 0
            if text_length > 0:
                # Check for repeated characters
                for char in set(text):
                    if text.count(char) > text_length * 0.3:
                        suspicious_patterns += 1
                
                # Check for unusual spacing
                if '  ' in text:  # Double spaces
                    suspicious_patterns += 1
                
                # Check for mixed case inconsistencies
                has_upper = any(c.isupper() for c in text)
                has_lower = any(c.islower() for c in text)
                if has_upper and has_lower and text.count(' ') > 5:
                    # Inconsistent capitalization in longer texts
                    suspicious_patterns += 1
            
            suspicious_patterns = min(suspicious_patterns, 5) / 5  # Normalize to 0-1
            
            # Text consistency score
            text_consistency = 1.0 - (word_length_variance / 100)  # Normalize
            text_consistency = max(0, min(1, text_consistency))
            
            # Extraction success rate (based on confidence)
            extraction_success = ocr_confidence
            
            # Formatting consistency (ID cards usually have consistent format)
            formatting_consistency = 1.0 if word_count > 5 and digit_count > 0 else 0.5
            
            # Anomaly score (combination of suspicious indicators)
            anomaly_score = (suspicious_patterns + (1 - text_consistency) + 
                           (1 - ocr_confidence)) / 3
            
            # Quality score (overall text quality)
            quality_score = (ocr_confidence + text_consistency + 
                          extraction_success) / 3
            
            return np.array([
                text_length,
                word_count,
                avg_word_length,
                char_diversity,
                digit_ratio,
                uppercase_ratio,
                punct_ratio,
                ocr_confidence,
                word_length_variance,
                suspicious_patterns,
                text_consistency,
                extraction_success,
                formatting_consistency,
                anomaly_score,
                quality_score
            ])
            
        except Exception as e:
            logger.error(f"Error extracting OCR features from {image_path}: {e}")
            # Return default values
            return np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    def load_dataset(self):
        """Load FantasyID dataset and extract features"""
        logger.info("Loading FantasyID dataset...")
        
        # Load CSV files
        train_df = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.dataset_path, 'test.csv'))
        
        # Combine train and test for feature extraction
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        
        features = []
        labels = []
        
        logger.info(f"Processing {len(all_df)} images...")
        
        for idx, row in tqdm(all_df.iterrows(), total=len(all_df)):
            image_path = os.path.join(self.dataset_path, row['path'])
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                continue
            
            # Extract image features
            img_features = self.extract_image_features(processed_img)
            if img_features is None:
                continue
            
            # Extract OCR features
            ocr_features = self.extract_ocr_features(image_path)
            
            # Combine features
            combined_features = np.concatenate([img_features, ocr_features])
            
            features.append(combined_features)
            labels.append(1 if row['is_attack'] else 0)  # 1=forged, 0=authentic
        
        return np.array(features), np.array(labels)
    
    def train_models(self, X, y):
        """Train SVM and Random Forest models"""
        logger.info("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM with hyperparameter tuning
        logger.info("Training SVM...")
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(probability=True, random_state=42)
        svm_grid = GridSearchCV(svm, svm_param_grid, cv=5, scoring='f1', n_jobs=-1)
        svm_grid.fit(X_train_scaled, y_train)
        
        self.svm_model = svm_grid.best_estimator_
        logger.info(f"Best SVM params: {svm_grid.best_params_}")
        
        # Evaluate SVM
        svm_pred = self.svm_model.predict(X_test_scaled)
        svm_metrics = {
            'accuracy': accuracy_score(y_test, svm_pred),
            'precision': precision_score(y_test, svm_pred),
            'recall': recall_score(y_test, svm_pred),
            'f1_score': f1_score(y_test, svm_pred)
        }
        logger.info(f"SVM Metrics: {svm_metrics}")
        
        # Train Random Forest with hyperparameter tuning
        logger.info("Training Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        
        self.rf_model = rf_grid.best_estimator_
        logger.info(f"Best RF params: {rf_grid.best_params_}")
        
        # Evaluate Random Forest
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1_score': f1_score(y_test, rf_pred)
        }
        logger.info(f"RF Metrics: {rf_metrics}")
        
        return svm_metrics, rf_metrics
    
    def save_models(self, svm_metrics, rf_metrics):
        """Save trained models and components"""
        logger.info("Saving models...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save models
        with open('models/fantasyid_svm_model.pkl', 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        with open('models/fantasyid_rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        # Save scaler
        with open('models/fantasyid_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metrics
        with open('models/fantasyid_svm_metrics.pkl', 'wb') as f:
            pickle.dump(svm_metrics, f)
        
        with open('models/fantasyid_rf_metrics.pkl', 'wb') as f:
            pickle.dump(rf_metrics, f)
        
        logger.info("Models saved successfully!")

def main():
    # Set up Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Using Tesseract at: {tesseract_path}")
        else:
            logger.warning("Tesseract not found at default location")
    
    # Initialize trainer
    dataset_path = r"C:\Users\kadam\OneDrive\Desktop\Document_Forgery\Dataset\FantasyID\FantasyID"
    trainer = FantasyIDTrainer(dataset_path)
    
    # Load and process dataset
    X, y = trainer.load_dataset()
    
    if len(X) == 0:
        logger.error("No valid data loaded!")
        return
    
    logger.info(f"Loaded {len(X)} samples with {len(X[0])} features")
    logger.info(f"Class distribution - Authentic: {np.sum(y == 0)}, Forged: {np.sum(y == 1)}")
    
    # Train models
    svm_metrics, rf_metrics = trainer.train_models(X, y)
    
    # Save models
    trainer.save_models(svm_metrics, rf_metrics)
    
    logger.info("Training completed successfully!")
    logger.info("Models saved to 'models' directory")
    logger.info("Restart backend to use the new models")

if __name__ == "__main__":
    main()
