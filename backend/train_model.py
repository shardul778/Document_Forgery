"""
Training script for Document Forgery Detection Model
Builds and trains the model from scratch with proper data handling
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import pickle
from pathlib import Path
from document_analyzer import DocumentAnalyzer
from ml_model import DocumentForgeryModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDataset(Dataset):
    """Dataset for document features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ModelTrainer:
    """Trainer for document forgery detection model"""
    
    def __init__(self, input_size=80, hidden_sizes=[256, 128, 64], learning_rate=0.001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        logger.info(f"Training on device: {self.device}")
    
    def generate_synthetic_data(self, num_samples=5000):
        """
        Generate synthetic training data with HEAVY overlap between classes.
        Uses shared base distribution + small class shift + label noise
        so that best achievable accuracy is ~80-90%.
        """
        logger.info(f"Generating {num_samples} synthetic samples (hard mode: heavy overlap + label noise)...")
        
        features = []
        labels = []
        
        # Generate authentic documents (55% of data)
        num_authentic = int(num_samples * 0.55)
        for _ in range(num_authentic):
            feature = self._generate_authentic_features()
            features.append(feature)
            labels.append(0)
        
        # Generate forged documents (45% of data)
        num_forged = num_samples - num_authentic
        for _ in range(num_forged):
            feature = self._generate_forged_features()
            features.append(feature)
            labels.append(1)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # LABEL NOISE: flip 5% of labels (lower = clearer signal, higher achievable accuracy)
        n_flip = int(num_samples * 0.05)
        flip_idx = np.random.choice(num_samples, size=n_flip, replace=False)
        labels[flip_idx] = 1 - labels[flip_idx]
        logger.info(f"Applied label noise: {n_flip} samples ({100*n_flip/num_samples:.1f}%) randomly relabeled")
        
        return features, labels
    
    def _generate_base_features(self):
        """Shared base distribution - same for both classes, high variance."""
        feature = np.zeros(80)
        # All features from one noisy distribution so classes overlap a lot
        feature[0:30] = np.clip(np.random.normal(0.5, 0.25, 30), 0, 1)
        feature[30] = np.clip(np.random.normal(0.4, 0.25), 0, 1)
        feature[31] = np.clip(np.random.normal(0.4, 0.25), 0, 1)
        feature[32] = np.clip(np.random.normal(0.2, 0.15), 0, 1)
        feature[33] = np.clip(np.random.normal(0.45, 0.25), 0, 1)
        feature[34] = np.clip(np.random.normal(0.45, 0.25), 0, 1)
        feature[35:41] = np.clip(np.random.normal(0.5, 0.2, 6), 0, 1)
        feature[41] = np.clip(np.random.normal(0.75, 0.2), 0, 1)
        feature[42] = np.clip(np.random.normal(0.7, 0.2), 0, 1)
        feature[43:49] = np.clip(np.random.normal(0.5, 0.25, 6), 0, 1)
        feature[49:64] = np.clip(np.random.normal(0.5, 0.2, 15), 0, 1)
        feature[64] = np.clip(np.random.normal(0.5, 0.25), 0, 1)
        feature[65] = np.clip(np.random.normal(0.5, 0.25), 0, 1)
        feature[66] = np.clip(np.random.normal(0.5, 0.2), 0, 1)
        feature[67:74] = np.clip(np.random.normal(0.4, 0.25, 7), 0, 1)
        return feature
    
    def _generate_authentic_features(self):
        """Authentic = base + clearer negative shift on 'forgery' indicators + moderate noise."""
        feature = self._generate_base_features().copy()
        # Stronger shift toward "clean" for better learnable signal
        feature[32] = np.clip(feature[32] - 0.12 + np.random.normal(0, 0.08), 0, 1)   # ELA lower
        feature[48] = np.clip(feature[48] - 0.14 + np.random.normal(0, 0.09), 0, 1)  # Block inconsistency lower
        feature[31] = np.clip(feature[31] - 0.08 + np.random.normal(0, 0.08), 0, 1)  # Edge variance
        feature[34] = np.clip(feature[34] - 0.12 + np.random.normal(0, 0.08), 0, 1)  # Texture consistency
        feature[72] = np.clip(feature[72] - 0.12 + np.random.normal(0, 0.09), 0, 1)  # Suspicious patterns
        feature[73] = np.clip(feature[73] - 0.12 + np.random.normal(0, 0.09), 0, 1)  # Word variance
        feature += np.random.normal(0, 0.08, 80)
        return np.clip(feature, 0, 1)
    
    def _generate_forged_features(self):
        """Forged = base + clearer positive shift on 'forgery' indicators + moderate noise."""
        feature = self._generate_base_features().copy()
        # Stronger shift toward "suspicious" for better learnable signal
        feature[32] = np.clip(feature[32] + 0.14 + np.random.normal(0, 0.09), 0, 1)   # ELA higher
        feature[48] = np.clip(feature[48] + 0.14 + np.random.normal(0, 0.09), 0, 1)   # Block inconsistency higher
        feature[31] = np.clip(feature[31] + 0.1 + np.random.normal(0, 0.08), 0, 1)    # Edge variance
        feature[34] = np.clip(feature[34] + 0.12 + np.random.normal(0, 0.09), 0, 1)  # Texture consistency
        feature[72] = np.clip(feature[72] + 0.14 + np.random.normal(0, 0.09), 0, 1)   # Suspicious patterns
        feature[73] = np.clip(feature[73] + 0.14 + np.random.normal(0, 0.09), 0, 1)   # Word variance
        feature += np.random.normal(0, 0.08, 80)
        return np.clip(feature, 0, 1)
    
    def train(self, epochs=150, batch_size=32, validation_split=0.2, doctamper_root=None, doctamper_max_samples=None):
        """Train the model. Use DocTamper dataset if doctamper_root is set."""
        logger.info("Starting model training...")
        
        # Load DocTamper dataset or generate synthetic data
        if doctamper_root and os.path.isdir(doctamper_root):
            try:
                from doctamper_loader import get_doctamper_data
                from document_analyzer import DocumentAnalyzer
                analyzer = DocumentAnalyzer()
                X, y = get_doctamper_data(
                    data_root=doctamper_root,
                    use_lmdb=False,
                    max_samples=doctamper_max_samples,
                    analyzer=analyzer,
                )
                logger.info(f"Loaded DocTamper: {X.shape[0]} samples, {X.shape[1]} features")
            except Exception as e:
                logger.warning(f"DocTamper load failed ({e}). Falling back to synthetic data.")
                X, y = self.generate_synthetic_data(num_samples=5000)
        else:
            X, y = self.generate_synthetic_data(num_samples=5000)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = DocumentDataset(X_train_scaled, y_train)
        val_dataset = DocumentDataset(X_val_scaled, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model (larger: 256, 128, 64 for higher capacity)
        model = DocumentForgeryModel(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=2
        ).to(self.device)
        
        # Loss and optimizer (moderate weight decay for accuracy + generalization)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=3e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_early = 30  # Allow more epochs to find better minimum
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
            
            # Save best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience_early:
                logger.info(f"Early stopping at epoch {epoch+1} (no val loss improvement for {patience_early} epochs). Best was epoch {best_epoch}.")
                break
        
        # Load best model (by validation loss)
        model.load_state_dict(best_model_state)
        logger.info(f"Using best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
        
        # Calculate final metrics on validation set
        metrics = self._calculate_metrics(model, val_loader)
        
        # Also calculate on training set for comparison
        logger.info("Calculating training set metrics...")
        train_metrics = self._calculate_metrics(model, train_loader)
        
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS:")
        logger.info("="*50)
        logger.info("Validation Set Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info("\nTraining Set Metrics:")
        logger.info(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {train_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {train_metrics['f1_score']:.4f}")
        
        # Check for overfitting
        train_acc = train_metrics['accuracy']
        val_acc = metrics['accuracy']
        if train_acc - val_acc > 0.1:
            logger.warning(f"\nWARNING: Possible overfitting detected!")
            logger.warning(f"  Training accuracy ({train_acc:.4f}) is much higher than validation ({val_acc:.4f})")
            logger.warning(f"  Difference: {train_acc - val_acc:.4f}")
        
        logger.info("="*50 + "\n")
        
        return model, metrics
    
    def _calculate_metrics(self, model, data_loader):
        """Calculate precision, recall, F1 score with detailed reporting"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                outputs = model(features)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability of forged class
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        
        total = len(all_labels)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        
        # Log detailed confusion matrix
        logger.info("\n" + "="*50)
        logger.info("Confusion Matrix:")
        logger.info(f"True Positives (TP):  {tp:4d}  |  False Positives (FP): {fp:4d}")
        logger.info(f"False Negatives (FN): {fn:4d}  |  True Negatives (TN):  {tn:4d}")
        logger.info("="*50)
        logger.info(f"Total samples: {total}")
        logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"F1 Score: {f1_score:.4f} ({f1_score*100:.2f}%)")
        logger.info(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        logger.info("="*50 + "\n")
        
        # Check if metrics are suspiciously high
        if accuracy > 0.98:
            logger.warning("WARNING: Accuracy is very high (>98%). This may indicate:")
            logger.warning("  - Data is too easy to separate")
            logger.warning("  - Model may be overfitting")
            logger.warning("  - Consider adding more noise or overlap to training data")
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "specificity": float(specificity)
        }
    
    def save_model(self, model, scaler, metrics, save_dir="models"):
        """Save trained model and scaler"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "forgery_model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Document Forgery Detection Model")
    parser.add_argument("--doctamper", type=str, default=None,
                        help="Path to DocTamper dataset root (folder with authentic/ and tampered/ or train/0, train/1)")
    parser.add_argument("--doctamper-max", type=int, default=None,
                        help="Max samples to use from DocTamper (default: all)")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    doctamper_root = args.doctamper or os.environ.get("DOCTAMPER_DATA", "")
    if doctamper_root and not os.path.isdir(doctamper_root):
        logger.warning(f"DocTamper path not found: {doctamper_root}. Using synthetic data.")
        doctamper_root = None
    
    trainer = ModelTrainer()
    model, metrics = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        doctamper_root=doctamper_root,
        doctamper_max_samples=args.doctamper_max,
    )
    trainer.save_model(model, trainer.scaler, metrics)
    print(f"\nTraining complete! Model saved with metrics: {metrics}")
