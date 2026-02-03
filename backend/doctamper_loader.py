"""
DocTamper Dataset Loader for Document Forgery Detection
Supports: folder-based structure and LMDB format
Dataset: https://github.com/qcf-568/DocTamper (CVPR 2023)
Kaggle: https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data
"""
import os
import numpy as np
import cv2
from PIL import Image
import io
import logging
from pathlib import Path
from typing import Tuple, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try optional LMDB for DocTamper .mdb files
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    logger.warning("lmdb not installed. pip install lmdb for LMDB support.")


def load_doctamper_from_folders(
    data_root: str,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    max_samples: Optional[int] = None,
    authentic_folder: str = "authentic",
    tampered_folder: str = "tampered",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DocTamper (or similar) dataset from folder structure.
    
    Expected structure:
        data_root/
            train/  (or just data_root)
                authentic/   or  0/
                    img1.jpg, img2.png, ...
                tampered/    or  1/
                    img1.jpg, ...
    
    Or:
        data_root/
            authentic/
            tampered/
    
    Returns:
        images: list of numpy arrays (H, W, 3) RGB
        labels: (N,) int 0/1
    """
    data_root = Path(data_root)
    images = []
    labels = []
    
    # Try different folder naming conventions
    for label, folder_names in enumerate([(authentic_folder, "0", "real"), (tampered_folder, "1", "tampered")]):
        found = False
        for folder_name in folder_names:
            folder = data_root / folder_name
            if not folder.exists():
                folder = data_root / "train" / folder_name
            if not folder.exists():
                folder = data_root / "Train" / folder_name
            if not folder.exists():
                continue
            found = True
            count = 0
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in image_extensions:
                    try:
                        img = cv2.imread(str(path))
                        if img is None:
                            img = np.array(Image.open(path).convert("RGB"))
                        else:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img.size == 0:
                            continue
                        images.append(img)
                        labels.append(label)
                        count += 1
                        if max_samples and len(images) >= max_samples:
                            return images, np.array(labels)
                    except Exception as e:
                        logger.warning(f"Skip {path}: {e}")
            logger.info(f"Loaded {count} images from {folder} (label={label})")
            break
        if not found:
            logger.warning(f"No folder found for label {label} (tried {folder_names})")
    
    if len(images) == 0:
        raise FileNotFoundError(
            f"No images found under {data_root}. "
            "Expected subfolders: 'authentic' and 'tampered', or '0' and '1', or 'train/authentic', 'train/tampered'."
        )
    
    return images, np.array(labels)


def load_doctamper_lmdb(
    lmdb_path: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load DocTamper from LMDB (official format).
    Keys are typically 'index-image', 'index-mask'; or single key with serialized data.
    
    Returns:
        images: list of numpy arrays RGB
        labels: 0 = authentic, 1 = tampered (derived from mask if present)
    """
    if not LMDB_AVAILABLE:
        raise ImportError("pip install lmdb to use LMDB dataset.")
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    images = []
    labels = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        seen_indices = set()
        for key, value in cursor:
            key_str = key.decode() if isinstance(key, bytes) else key
            # Common patterns: '000000-image', '000000-mask', or just index
            if "-image" in key_str or "image" in key_str:
                idx = key_str.split("-")[0].split("_")[0]
                if idx in seen_indices:
                    continue
                try:
                    buf = io.BytesIO(value)
                    img = np.array(Image.open(buf).convert("RGB"))
                    if img.size == 0:
                        continue
                    images.append(img)
                    # Check for mask to set label
                    mask_key = f"{idx}-mask" if "-" in key_str else f"{idx}_mask"
                    mask_val = txn.get(mask_key.encode() if isinstance(mask_key, str) else mask_key)
                    if mask_val is not None:
                        mask_buf = io.BytesIO(mask_val)
                        mask = np.array(Image.open(mask_buf))
                        label = 1 if np.any(mask > 0) else 0
                    else:
                        label = 1  # assume tampered if we only have image keys in tampered set
                    labels.append(label)
                    seen_indices.add(idx)
                except Exception as e:
                    logger.warning(f"Skip key {key_str}: {e}")
                if max_samples and len(images) >= max_samples:
                    break
    
    env.close()
    if len(images) == 0:
        raise ValueError(f"No valid images in LMDB at {lmdb_path}. Keys may use different format.")
    
    return images, np.array(labels)


def extract_features_from_images(
    images,
    labels: np.ndarray,
    analyzer,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use DocumentAnalyzer to extract 80-dim features from each image.
    """
    features_list = []
    labels_list = []
    failed = 0
    
    n = len(images)
    if max_samples:
        n = min(n, max_samples)
    
    for i in range(n):
        try:
            img = images[i]
            if img is None:
                failed += 1
                continue
            # Ensure RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            image_features = analyzer._extract_image_features(img)
            # OCR features: simulate from empty/mock if no OCR run (keep same dim)
            ocr_text = analyzer._extract_ocr_text(img)
            ocr_features = analyzer._extract_ocr_features(ocr_text, img)
            combined = np.concatenate([image_features, ocr_features])
            features_list.append(combined)
            labels_list.append(labels[i])
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Feature extraction failed for image {i}: {e}")
    
    if failed > 0:
        logger.info(f"Feature extraction failed for {failed} images.")
    if len(features_list) == 0:
        raise RuntimeError("No features extracted from any image.")
    
    return np.array(features_list, dtype=np.float32), np.array(labels_list)


def get_doctamper_data(
    data_root: str,
    use_lmdb: bool = False,
    lmdb_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    analyzer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main entry: load DocTamper and return (features, labels) for training.
    
    Args:
        data_root: path to dataset root (folder structure) or parent of LMDB
        use_lmdb: if True, load from LMDB (lmdb_path or data_root must point to .mdb)
        lmdb_path: path to .mdb file (e.g. data_root/DocTamper_train.mdb)
        max_samples: cap number of samples (for quick runs)
        analyzer: DocumentAnalyzer instance for feature extraction
    
    Returns:
        features: (N, 80) float32
        labels: (N,) int 0/1
    """
    if analyzer is None:
        from document_analyzer import DocumentAnalyzer
        analyzer = DocumentAnalyzer()
    
    if use_lmdb or (lmdb_path and os.path.exists(lmdb_path)):
        path = lmdb_path or os.path.join(data_root, "DocTamper_train.mdb")
        if not os.path.exists(path):
            path = data_root
            if not path.endswith(".mdb"):
                for name in os.listdir(data_root or "."):
                    if name.endswith(".mdb"):
                        path = os.path.join(data_root, name)
                        break
        logger.info(f"Loading DocTamper from LMDB: {path}")
        images, labels = load_doctamper_lmdb(path, max_samples=max_samples)
    else:
        logger.info(f"Loading DocTamper from folders: {data_root}")
        images, labels = load_doctamper_from_folders(data_root, max_samples=max_samples)
    
    logger.info(f"Extracting features from {len(images)} images...")
    features, labels = extract_features_from_images(images, labels, analyzer, max_samples=None)
    return features, labels
