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
    """Generate more realistic training data with overlap to prevent overfitting"""
    logger.info(f"Generating {n_samples} realistic samples...")
    
    # Generate authentic documents (70% of data - more balanced)
    n_authentic = int(0.7 * n_samples)
    n_forged = n_samples - n_authentic
    
    # Authentic documents - with more variance and overlap
    authentic_features = []
    for i in range(n_authentic):
        # Image features (64 features) - more variance for authentic
        img_features = np.random.normal(0.4, 0.2, 64)  # Higher variance
        img_features = np.clip(img_features, 0, 1)
        
        # OCR features (16 features) - with overlap between classes
        ocr_features = np.array([
            np.random.normal(350, 120),    # text_length (more overlap)
            np.random.normal(60, 20),      # word_count (more overlap)
            np.random.normal(5.8, 1.2),    # avg_word_length (more overlap)
            np.random.normal(13, 5),       # line_count (more overlap)
            np.random.normal(0.55, 0.18),  # char_diversity (more overlap)
            np.random.normal(0.18, 0.1),   # digit_ratio (more overlap)
            np.random.normal(0.45, 0.15),  # uppercase_ratio (more overlap)
            np.random.normal(0.07, 0.04),  # punctuation_ratio (more overlap)
            np.random.normal(0.65, 0.18),  # ocr_confidence (more overlap)
            np.random.normal(0.18, 0.1),   # word_length_variance (more overlap)
            np.random.normal(0.08, 0.05),  # suspicious_patterns (more overlap)
            np.random.normal(0.65, 0.18),  # text_consistency (more overlap)
            np.random.normal(0.85, 0.12),  # extraction_success (more overlap)
            np.random.normal(0.75, 0.15),  # formatting_consistency (more overlap)
            np.random.normal(0.08, 0.05),  # anomaly_score (more overlap)
            np.random.normal(0.8, 0.12)    # quality_score (more overlap)
        ])
        
        # Ensure valid ranges
        ocr_features = np.maximum(ocr_features, 0)
        
        features = np.concatenate([img_features, ocr_features])
        authentic_features.append(features)
    
    # Forged documents - with less extreme differences
    forged_features = []
    for i in range(n_forged):
        # Image features (64 features) - closer to authentic
        img_features = np.random.normal(0.45, 0.22, 64)  # Closer to authentic
        img_features = np.clip(img_features, 0, 1)
        
        # OCR features (16 features) - less extreme differences
        ocr_features = np.array([
            np.random.normal(320, 130),   # text_length (closer to authentic)
            np.random.normal(55, 22),      # word_count (closer to authentic)
            np.random.normal(6.2, 1.4),    # avg_word_length (closer to authentic)
            np.random.normal(14, 6),       # line_count (closer to authentic)
            np.random.normal(0.5, 0.2),    # char_diversity (closer to authentic)
            np.random.normal(0.22, 0.11),  # digit_ratio (closer to authentic)
            np.random.normal(0.48, 0.18),  # uppercase_ratio (closer to authentic)
            np.random.normal(0.09, 0.05),  # punctuation_ratio (closer to authentic)
            np.random.normal(0.58, 0.2),    # ocr_confidence (closer to authentic)
            np.random.normal(0.22, 0.12),  # word_length_variance (closer to authentic)
            np.random.normal(0.12, 0.08),  # suspicious_patterns (closer to authentic)
            np.random.normal(0.58, 0.2),    # text_consistency (closer to authentic)
            np.random.normal(0.8, 0.18),    # extraction_success (closer to authentic)
            np.random.normal(0.7, 0.2),     # formatting_consistency (closer to authentic)
            np.random.normal(0.12, 0.08),   # anomaly_score (closer to authentic)
            np.random.normal(0.75, 0.15)   # quality_score (closer to authentic)
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
    """Train and tune SVM model with regularization"""
    logger.info("Training SVM model...")
    
    # Hyperparameter tuning with more regularization
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Added smaller C for more regularization
        'gamma': ['scale', 'auto', 0.001, 0.01],  # Added smaller gamma
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)  # Increased CV folds
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
    """Train and tune Random Forest model with regularization"""
    logger.info("Training Random Forest model...")
    
    # Hyperparameter tuning with more regularization
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],  # Added shallower trees
        'min_samples_split': [5, 10, 20],  # Increased to prevent overfitting
        'min_samples_leaf': [2, 4, 8],  # Increased to prevent overfitting
        'max_features': ['sqrt', 'log2', 0.5]  # Added feature limiting
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)  # Increased CV folds
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
