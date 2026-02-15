import numpy as np
import cv2
from PIL import Image
import io
import pytesseract
import re
from typing import Dict, List, Any
import logging
from ml_model_svm_rf import get_document_forgery_model
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    # Common Tesseract installation paths on Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Found Tesseract at: {path}")
            break
    else:
        # Check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract found in PATH")
        except Exception:
            logger.warning("Tesseract OCR not found. OCR features will be limited.")
            logger.warning("Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.warning("Or set TESSERACT_CMD environment variable to the tesseract.exe path")
    
    # Verify Tesseract is working
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR v{version} is ready")
    except Exception as e:
        logger.error(f"Tesseract OCR verification failed: {e}")

class DocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer with ML model"""
        self.model = get_document_forgery_model()
        logger.info("Document analyzer initialized")
    
    def analyze(self, file_contents: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Analyze document for forgery detection
        
        Args:
            file_contents: Raw file bytes
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Convert to image if needed
            image = self._load_image(file_contents, content_type)
            
            # Extract features
            ocr_text = self._extract_ocr_text(image)
            image_features = self._extract_image_features(image)
            ocr_features = self._extract_ocr_features(ocr_text, image)
            
            # Combine features
            combined_features = np.concatenate([
                image_features,
                ocr_features
            ])
            
            # Predict using ML model (with proper calibration)
            prediction, confidence, model_details = self.model.predict(combined_features)
            
            # Get detection details (only if forged)
            detection_details = []
            if prediction == 1:
                detection_details = self._get_detection_details(
                    image, ocr_text, image_features, ocr_features
                )
            else:
                detection_details = [{
                    "type": "Analysis Complete",
                    "description": "Document appears authentic based on comprehensive analysis",
                    "confidence": confidence
                }]
            
            # Get model metrics
            metrics = self.model.get_model_info()
            
            result = {
                "is_forged": bool(prediction),
                "confidence": float(confidence),
                "metrics": metrics,
                "detection_details": detection_details,
                "ocr_text": ocr_text[:1000] if ocr_text else "",  # Limit text length
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}", exc_info=True)
            raise
    
    def _load_image(self, file_contents: bytes, content_type: str) -> np.ndarray:
        """Load image from bytes"""
        try:
            if content_type == "application/pdf":
                # For PDF, convert first page to image
                try:
                    from pdf2image import convert_from_bytes
                    images = convert_from_bytes(file_contents, first_page=1, last_page=1)
                    if images:
                        return np.array(images[0])
                    else:
                        raise ValueError("Could not extract image from PDF")
                except Exception as pdf_error:
                    # Check if it's a Poppler error
                    error_msg = str(pdf_error).lower()
                    if "poppler" in error_msg or "pdfinfo" in error_msg:
                        logger.warning("Poppler not installed. Attempting alternative PDF processing...")
                        # Try alternative: use PyMuPDF (fitz) if available
                        try:
                            import fitz  # PyMuPDF
                            pdf_document = fitz.open(stream=file_contents, filetype="pdf")
                            if len(pdf_document) > 0:
                                page = pdf_document[0]
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                                img_data = pix.tobytes("ppm")
                                image = Image.open(io.BytesIO(img_data))
                                pdf_document.close()
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                return np.array(image)
                            else:
                                raise ValueError("PDF has no pages")
                        except ImportError:
                            raise ValueError(
                                "PDF processing requires Poppler or PyMuPDF. "
                                "Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/ "
                                "Or install PyMuPDF: pip install PyMuPDF"
                            )
                        except Exception as alt_error:
                            raise ValueError(
                                f"PDF processing failed. Install Poppler: "
                                f"https://github.com/oschwartz10612/poppler-windows/releases/ "
                                f"Error: {str(alt_error)}"
                            )
                    else:
                        raise
            else:
                # For image files
                image = Image.open(io.BytesIO(file_contents))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def _extract_ocr_text(self, image: np.ndarray) -> str:
        """Extract text using OCR"""
        try:
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(image)
            text = pytesseract.image_to_string(pil_image, lang='eng')
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR extraction failed: {str(e)}")
            return ""
    
    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract visual features from image with advanced forgery detection"""
        try:
            # Resize for consistent feature extraction
            img_resized = cv2.resize(image, (224, 224))
            
            # Convert to grayscale for some features
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            
            features = []
            
            # 1. Histogram features (reduced to make room for better features)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features.extend(hist.flatten()[:30])  # Reduced from 50 to 30
            
            # 2. Edge detection features (enhanced)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Edge consistency (forged docs have inconsistent edges)
            edge_variance = np.var(edges.astype(float))
            features.append(min(edge_variance / 10000.0, 1.0))  # Normalize
            
            # 3. Error Level Analysis (ELA) - detects JPEG compression inconsistencies
            ela_score = self._calculate_ela_score(img_resized)
            features.append(ela_score)
            
            # 4. Texture features (enhanced)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_variance = np.mean(local_var)
            features.append(min(texture_variance / 1000.0, 1.0))  # Normalize
            
            # Texture consistency (inconsistent = suspicious)
            texture_std = np.std(local_var)
            features.append(min(texture_std / 500.0, 1.0))
            
            # 5. Color statistics (enhanced)
            features.append(np.mean(image[:, :, 0]) / 255.0)  # R mean normalized
            features.append(np.mean(image[:, :, 1]) / 255.0)  # G mean normalized
            features.append(np.mean(image[:, :, 2]) / 255.0)  # B mean normalized
            features.append(np.std(image[:, :, 0]) / 255.0)   # R std normalized
            features.append(np.std(image[:, :, 1]) / 255.0)   # G std normalized
            features.append(np.std(image[:, :, 2]) / 255.0)   # B std normalized
            
            # Color channel correlation (forged docs may have unusual correlations)
            r_g_corr = np.corrcoef(image[:, :, 0].flatten(), image[:, :, 1].flatten())[0, 1]
            g_b_corr = np.corrcoef(image[:, :, 1].flatten(), image[:, :, 2].flatten())[0, 1]
            features.append((r_g_corr + 1) / 2.0)  # Normalize to [0, 1]
            features.append((g_b_corr + 1) / 2.0)
            
            # 6. Frequency domain features (enhanced)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            center = magnitude_spectrum.shape[0] // 2
            
            # Central region mean
            center_mean = np.mean(magnitude_spectrum[center-10:center+10, center-10:center+10])
            features.append(min(center_mean / 10000.0, 1.0))
            
            # Frequency variance (high variance = possible manipulation)
            freq_variance = np.var(magnitude_spectrum)
            features.append(min(freq_variance / 1e8, 1.0))
            
            # 7. Noise analysis (forged docs often have inconsistent noise)
            noise_level = self._estimate_noise_level(gray)
            features.append(min(noise_level / 50.0, 1.0))
            
            # 8. Gradient analysis (detects sharp transitions from editing)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            gradient_std = np.std(gradient_magnitude)
            features.append(min(gradient_mean / 100.0, 1.0))
            features.append(min(gradient_std / 50.0, 1.0))
            
            # 9. Block-based analysis (detects inconsistencies in blocks)
            block_inconsistency = self._calculate_block_inconsistency(gray)
            features.append(min(block_inconsistency, 1.0))
            
            # Pad or truncate to fixed size
            target_size = 64
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting image features: {str(e)}")
            return np.zeros(64, dtype=np.float32)
    
    def _calculate_ela_score(self, image: np.ndarray, quality: int = 90) -> float:
        """Calculate Error Level Analysis score - detects JPEG compression inconsistencies"""
        try:
            # Save and reload at different quality to detect compression artifacts
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Save at specified quality
            cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Reload
            reloaded = cv2.imread(tmp_path)
            os.unlink(tmp_path)
            
            if reloaded is None:
                return 0.0
            
            reloaded_rgb = cv2.cvtColor(reloaded, cv2.COLOR_BGR2RGB)
            
            # Calculate difference
            diff = np.abs(image.astype(float) - reloaded_rgb.astype(float))
            ela_score = np.mean(diff) / 255.0
            
            return min(ela_score, 1.0)
        except Exception as e:
            logger.warning(f"ELA calculation failed: {str(e)}")
            return 0.0
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Use Laplacian variance as noise estimator
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.var()
            return noise_level
        except Exception as e:
            logger.warning(f"Noise estimation failed: {str(e)}")
            return 0.0
    
    def _calculate_block_inconsistency(self, gray: np.ndarray, block_size: int = 32) -> float:
        """Calculate inconsistency across image blocks (detects tampering)"""
        try:
            h, w = gray.shape
            blocks = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    blocks.append(np.mean(block))
            
            if len(blocks) < 2:
                return 0.0
            
            # Calculate variance of block means (high variance = inconsistent)
            block_variance = np.var(blocks)
            return min(block_variance / 1000.0, 1.0)
        except Exception as e:
            logger.warning(f"Block inconsistency calculation failed: {str(e)}")
            return 0.0
    
    def _extract_ocr_features(self, ocr_text: str, image: np.ndarray) -> np.ndarray:
        """Extract features from OCR text"""
        features = []
        
        try:
            # 1. Text length
            features.append(len(ocr_text))
            
            # 2. Word count
            words = ocr_text.split()
            features.append(len(words))
            
            # 3. Average word length
            if words:
                avg_word_len = np.mean([len(w) for w in words])
            else:
                avg_word_len = 0.0
            features.append(avg_word_len)
            
            # 4. Character diversity (unique chars / total chars)
            if ocr_text:
                char_diversity = len(set(ocr_text.lower())) / len(ocr_text)
            else:
                char_diversity = 0.0
            features.append(char_diversity)
            
            # 5. Digit ratio
            if ocr_text:
                digit_ratio = sum(c.isdigit() for c in ocr_text) / len(ocr_text)
            else:
                digit_ratio = 0.0
            features.append(digit_ratio)
            
            # 6. Uppercase ratio
            if ocr_text:
                upper_ratio = sum(c.isupper() for c in ocr_text) / len(ocr_text)
            else:
                upper_ratio = 0.0
            features.append(upper_ratio)
            
            # 7. Punctuation ratio
            if ocr_text:
                punct_ratio = sum(c in '.,!?;:-()[]{}"\'' for c in ocr_text) / len(ocr_text)
            else:
                punct_ratio = 0.0
            features.append(punct_ratio)
            
            # 8. Line count
            lines = ocr_text.split('\n')
            features.append(len([l for l in lines if l.strip()]))
            
            # 9. Suspicious patterns (common in forgeries)
            suspicious_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # Dates
                r'\$\d+',  # Currency
                r'[A-Z]{2,}',  # Multiple uppercase
            ]
            suspicious_count = sum(len(re.findall(pattern, ocr_text)) for pattern in suspicious_patterns)
            features.append(suspicious_count)
            
            # 10. Text consistency (variance in word lengths)
            if words:
                word_lengths = [len(w) for w in words]
                word_length_variance = np.var(word_lengths)
            else:
                word_length_variance = 0.0
            features.append(word_length_variance)
            
            # Pad to fixed size
            target_size = 16
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting OCR features: {str(e)}")
            return np.zeros(16, dtype=np.float32)
    
    def _get_detection_details(
        self, 
        image: np.ndarray, 
        ocr_text: str, 
        image_features: np.ndarray,
        ocr_features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Get detailed detection information with enhanced analysis"""
        details = []
        
        try:
            # 1. ELA Analysis (Error Level Analysis)
            if len(image_features) > 32:
                ela_score = image_features[32]
                if ela_score > 0.15:
                    details.append({
                        "type": "Compression Analysis",
                        "description": f"Error Level Analysis detected compression inconsistencies (score: {ela_score:.3f}) - possible digital manipulation",
                        "confidence": min(0.5 + ela_score, 0.9)
                    })
            
            # 2. Edge Analysis
            if len(image_features) > 31:
                edge_density = image_features[30]
                edge_variance = image_features[31]
                
                if edge_density < 0.10:  # Made more strict
                    details.append({
                        "type": "Edge Detection",
                        "description": f"Very low edge density ({edge_density:.3f}) - document may be blurred or manipulated",
                        "confidence": 0.75
                    })
                elif edge_variance > 0.7:  # Made more strict
                    details.append({
                        "type": "Edge Consistency",
                        "description": f"Inconsistent edge patterns detected (variance: {edge_variance:.3f}) - possible cut-and-paste operations",
                        "confidence": 0.7
                    })
            
            # 3. Texture Analysis
            if len(image_features) > 34:
                texture_consistency = image_features[34]
                if texture_consistency > 0.8:  # Made more strict
                    details.append({
                        "type": "Texture Analysis",
                        "description": f"Inconsistent texture patterns detected (score: {texture_consistency:.3f}) - possible image editing",
                        "confidence": 0.7
                    })
            
            # 4. Block Inconsistency
            if len(image_features) > 48:
                block_inconsistency = image_features[48]
                if block_inconsistency > 0.3:
                    details.append({
                        "type": "Block Analysis",
                        "description": f"High inconsistency across image blocks (score: {block_inconsistency:.3f}) - possible tampering or manipulation",
                        "confidence": 0.8
                    })
            
            # 5. Frequency Domain Analysis
            if len(image_features) > 44:
                freq_variance = image_features[44]
                if freq_variance > 0.4:
                    details.append({
                        "type": "Frequency Analysis",
                        "description": f"Anomalies detected in frequency domain (variance: {freq_variance:.3f}) - possible resampling or compression artifacts",
                        "confidence": 0.75
                    })
            
            # 6. Noise Analysis
            if len(image_features) > 45:
                noise_level = image_features[45]
                if noise_level < 0.05:
                    details.append({
                        "type": "Noise Analysis",
                        "description": f"Abnormally low noise level ({noise_level:.3f}) - document may be artificially smoothed",
                        "confidence": 0.65
                    })
                elif noise_level > 0.7:
                    details.append({
                        "type": "Noise Analysis",
                        "description": f"Abnormally high noise level ({noise_level:.3f}) - possible multiple compression cycles",
                        "confidence": 0.7
                    })
            
            # 7. Gradient Analysis
            if len(image_features) > 47:
                gradient_std = image_features[47]
                if gradient_std > 0.6:
                    details.append({
                        "type": "Gradient Analysis",
                        "description": f"Sharp transitions detected (gradient std: {gradient_std:.3f}) - possible editing artifacts",
                        "confidence": 0.7
                    })
            
            # 8. OCR Quality Check
            if len(ocr_text) < 10:
                details.append({
                    "type": "OCR Quality",
                    "description": "Very little text extracted - document may be low quality, heavily edited, or forged",
                    "confidence": 0.6
                })
            
            # 9. Text Consistency
            if len(ocr_features) > 73:
                word_variance = ocr_features[73]
                if word_variance > 0.6:
                    details.append({
                        "type": "Text Consistency",
                        "description": f"High variance in word lengths (score: {word_variance:.3f}) - possible text tampering or inconsistent formatting",
                        "confidence": 0.7
                    })
            
            # 10. Suspicious Patterns
            if len(ocr_features) > 72:
                suspicious_count = ocr_features[72]
                if suspicious_count > 0.4:
                    details.append({
                        "type": "Pattern Analysis",
                        "description": f"Suspicious patterns detected in text (score: {suspicious_count:.3f}) - unusual formatting or content",
                        "confidence": 0.65
                    })
            
            # 11. Color Analysis
            if len(image_features) > 42:
                r_g_corr = image_features[41]
                if r_g_corr < 0.5 or r_g_corr > 0.95:
                    details.append({
                        "type": "Color Analysis",
                        "description": f"Unusual color channel correlation ({r_g_corr:.3f}) - possible color manipulation",
                        "confidence": 0.65
                    })
            
            # If no suspicious indicators found
            if not details:
                details.append({
                    "type": "Analysis Complete",
                    "description": "No obvious forgery indicators detected - document appears authentic based on visual and text analysis",
                    "confidence": 0.5
                })
                
        except Exception as e:
            logger.error(f"Error getting detection details: {str(e)}")
            details.append({
                "type": "Analysis Error",
                "description": f"Error during detailed analysis: {str(e)}",
                "confidence": 0.0
            })
        
        return details
