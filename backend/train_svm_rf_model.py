"""
Training Script for SVM and Random Forest Models
Objective 3: Train SVM and Random Forest classifiers for document forgery detection
"""
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from document_analyzer import DocumentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_data(n_samples=2000):
    """Generate more realistic training data that matches real document characteristics"""
    logger.info(f"Generating {n_samples} realistic samples...")
    
    # Generate authentic documents (80% of data - more realistic ratio)
    n_authentic = int(0.8 * n_samples)
    n_forged = n_samples - n_authentic
    
    # Authentic documents - based on real document characteristics
    authentic_features = []
    for i in range(n_authentic):
        # Image features (64 features) - realistic for authentic documents
        img_features = np.random.normal(0.3, 0.15, 64)  # Lower mean, moderate variance
        img_features = np.clip(img_features, 0, 1)  # Ensure valid range
        
        # OCR features (16 features) - realistic text patterns from real documents
        ocr_features = np.array([
            np.random.normal(300, 100),     # text_length (real documents vary)
            np.random.normal(50, 15),       # word_count
            np.random.normal(5.5, 1.0),     # avg_word_length
            np.random.normal(12, 4),        # line_count
            np.random.normal(0.6, 0.15),    # char_diversity
            np.random.normal(0.2, 0.08),    # digit_ratio (ID cards have digits)
            np.random.normal(0.4, 0.12),    # uppercase_ratio
            np.random.normal(0.06, 0.03),   # punctuation_ratio
            np.random.normal(0.7, 0.15),    # ocr_confidence
            np.random.normal(0.15, 0.08),   # word_length_variance
            np.random.normal(0.05, 0.03),   # suspicious_patterns (low for real)
            np.random.normal(0.7, 0.15),    # text_consistency
            np.random.normal(0.9, 0.08),    # extraction_success
            np.random.normal(0.8, 0.12),    # formatting_consistency
            np.random.normal(0.05, 0.03),   # anomaly_score (low)
            np.random.normal(0.85, 0.1)     # quality_score
        ])
        
        # Ensure valid ranges
        ocr_features = np.maximum(ocr_features, 0)  # No negative values
        
        features = np.concatenate([img_features, ocr_features])
        authentic_features.append(features)
    
    # Forged documents - more extreme anomalies
    forged_features = []
    for i in range(n_forged):
        # Image features (64 features) - more extreme for forged
        img_features = np.random.normal(0.5, 0.25, 64)  # Higher variance
        img_features = np.clip(img_features, 0, 1)
        
        # OCR features (16 features) - more extreme inconsistencies
        ocr_features = np.array([
            np.random.normal(250, 150),    # text_length (more variable)
            np.random.normal(40, 25),      # word_count (more variable)
            np.random.normal(6.5, 2.0),    # avg_word_length (inconsistent)
            np.random.normal(15, 6),       # line_count (inconsistent)
            np.random.normal(0.4, 0.2),    # char_diversity (lower)
            np.random.normal(0.25, 0.12),  # digit_ratio (higher)
            np.random.normal(0.5, 0.2),    # uppercase_ratio (inconsistent)
            np.random.normal(0.1, 0.06),   # punctuation_ratio (higher)
            np.random.normal(0.5, 0.25),    # ocr_confidence (lower)
            np.random.normal(0.3, 0.15),   # word_length_variance (higher)
            np.random.normal(0.2, 0.1),    # suspicious_patterns (more)
            np.random.normal(0.5, 0.2),    # text_consistency (lower)
            np.random.normal(0.7, 0.2),    # extraction_success (lower)
            np.random.normal(0.6, 0.25),    # formatting_consistency (lower)
            np.random.normal(0.25, 0.15),   # anomaly_score (higher)
            np.random.normal(0.6, 0.2)     # quality_score (lower)
        ])
        
        # Ensure valid ranges
        ocr_features = np.maximum(ocr_features, 0)
        
        features = np.concatenate([img_features, ocr_features])
        forged_features.append(features)
    
    # Combine data
    X = np.array(authentic_features + forged_features)
    y = np.array([0] * n_authentic + [1] * n_forged)
    
    logger.info(f"Generated {n_authentic} authentic and {n_forged} forged samples")
    return X, y

def train_svm_model(X_train, y_train, X_val, y_val):
    """Train and tune SVM model"""
    logger.info("Training SVM model...")
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    logger.info(f"Best SVM parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    svm_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    logger.info(f"SVM Validation Metrics: {svm_metrics}")
    
    return best_svm, svm_metrics

def train_rf_model(X_train, y_train, X_val, y_val):
    """Train and tune Random Forest model"""
    logger.info("Training Random Forest model...")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"Best RF parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    rf_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    logger.info(f"RF Validation Metrics: {rf_metrics}")
    
    return best_rf, rf_metrics

def main():
    """Main training function"""
    logger.info("Starting SVM and Random Forest model training...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Generate realistic data
    X, y = generate_realistic_data(n_samples=3000)  # Increased samples for better training
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    svm_model, svm_metrics = train_svm_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Train Random Forest model
    rf_model, rf_metrics = train_rf_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    
    # SVM test evaluation
    svm_test_pred = svm_model.predict(X_test_scaled)
    svm_test_metrics = {
        "accuracy": accuracy_score(y_test, svm_test_pred),
        "precision": precision_score(y_test, svm_test_pred),
        "recall": recall_score(y_test, svm_test_pred),
        "f1_score": f1_score(y_test, svm_test_pred)
    }
    
    # RF test evaluation
    rf_test_pred = rf_model.predict(X_test_scaled)
    rf_test_metrics = {
        "accuracy": accuracy_score(y_test, rf_test_pred),
        "precision": precision_score(y_test, rf_test_pred),
        "recall": recall_score(y_test, rf_test_pred),
        "f1_score": f1_score(y_test, rf_test_pred)
    }
    
    logger.info(f"SVM Test Metrics: {svm_test_metrics}")
    logger.info(f"RF Test Metrics: {rf_test_metrics}")
    
    # Save models and components
    logger.info("Saving models and components...")
    
    with open(models_dir / "svm_model.pkl", 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open(models_dir / "rf_model.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open(models_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(models_dir / "svm_metrics.pkl", 'wb') as f:
        pickle.dump(svm_test_metrics, f)
    
    with open(models_dir / "rf_metrics.pkl", 'wb') as f:
        pickle.dump(rf_test_metrics, f)
    
    # Save training report
    report = {
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "features": len(X[0]),
        "svm_metrics": svm_test_metrics,
        "rf_metrics": rf_test_metrics,
        "svm_params": svm_model.get_params(),
        "rf_params": rf_model.get_params()
    }
    
    with open(models_dir / "training_report.pkl", 'wb') as f:
        pickle.dump(report, f)
    
    logger.info("Training completed successfully!")
    logger.info(f"Models saved to {models_dir}")
    logger.info("Restart the backend to use the trained models.")

if __name__ == "__main__":
    main()
