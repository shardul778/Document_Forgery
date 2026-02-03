# Model Training Guide

## Overview

The document forgery detection model has been rebuilt from scratch with proper training infrastructure. The model uses a deep neural network that learns from synthetic data to distinguish between authentic and forged documents.

## Architecture

### Model Structure
- **Input Layer**: 80 features (64 image features + 16 OCR features)
- **Hidden Layers**: 
  - 128 neurons → BatchNorm → ReLU → Dropout(0.3)
  - 64 neurons → BatchNorm → ReLU → Dropout(0.3)
  - 32 neurons → BatchNorm → ReLU → Dropout(0.3)
- **Output Layer**: 2 classes (Authentic=0, Forged=1)

### Features
- **Image Features (64)**: Histogram, edges, texture, color, frequency domain, noise, gradients
- **OCR Features (16)**: Text length, word count, consistency, patterns

## Training the Model

### Quick Start

1. **Train the model:**
   ```bash
   # Windows
   train_and_deploy.bat
   
   # Or manually
   cd backend
   venv\Scripts\activate
   python train_model.py
   ```

2. **Training Process:**
   - Generates 2000 synthetic samples (70% authentic, 30% forged)
   - Splits into train/validation (80/20)
   - Trains for 100 epochs with early stopping
   - Saves model, scaler, and metrics to `backend/models/`

3. **Restart backend:**
   ```bash
   start_backend.bat
   ```

### Training Parameters

The model uses:
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 32
- **Epochs**: 100
- **Validation Split**: 20%
- **Learning Rate Scheduler**: Reduces LR on plateau

### Calibration

The model uses a **forgery threshold of 0.65** to reduce false positives:
- Only flags documents as forged if confidence ≥ 65%
- Defaults to "authentic" for lower confidence scores
- This prevents legitimate documents from being flagged

## Model Files

After training, these files are created in `backend/models/`:

- `forgery_model.pth` - Trained PyTorch model weights
- `scaler.pkl` - Feature scaler (StandardScaler)
- `metrics.pkl` - Model performance metrics

## Expected Performance

After training, you should see metrics like:
- **Accuracy**: ~85-90%
- **Precision**: ~80-85% (fewer false positives)
- **Recall**: ~75-85% (catches most forgeries)
- **F1 Score**: ~80-85%

## Adjusting Sensitivity

To reduce false positives further, increase the threshold:

```python
# In ml_model.py, modify:
self.forgery_threshold = 0.75  # Higher = fewer false positives
```

To catch more forgeries (but may increase false positives):

```python
self.forgery_threshold = 0.55  # Lower = more sensitive
```

## Training with Real Data

To train with your own dataset:

1. **Prepare data:**
   - Collect authentic and forged document samples
   - Extract features using `DocumentAnalyzer`
   - Save features and labels

2. **Modify `train_model.py`:**
   - Replace `generate_synthetic_data()` with your data loader
   - Adjust class distribution if needed

3. **Train:**
   ```bash
   python train_model.py
   ```

## Troubleshooting

### Model not loading
- Check that `backend/models/` directory exists
- Verify model files were created after training
- Check file permissions

### Poor performance
- Train for more epochs (increase `epochs` parameter)
- Adjust learning rate
- Add more training samples
- Check feature extraction quality

### Too many false positives
- Increase `forgery_threshold` (e.g., 0.70 or 0.75)
- Retrain with more authentic samples
- Check feature normalization

### Too many false negatives
- Decrease `forgery_threshold` (e.g., 0.55 or 0.60)
- Retrain with more forged samples
- Improve feature extraction

## Next Steps

1. Train the model: `python train_model.py`
2. Test with sample documents
3. Adjust threshold based on results
4. Retrain if needed with adjusted parameters
