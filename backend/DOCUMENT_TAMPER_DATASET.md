# Using the DocTamper Dataset

This project supports training on the **DocTamper** dataset (CVPR 2023) for document forgery/tampering detection.

## Dataset

- **Paper**: Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution (CVPR 2023)
- **GitHub**: https://github.com/qcf-568/DocTamper
- **Kaggle**: https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data
- **Content**: ~170k document images (authentic vs tampered), three tampering types: copy-move, splicing, generation

## Setup

### Option 1: Download from Kaggle (folder structure)

1. Install Kaggle CLI (if needed):
   ```bash
   pip install kaggle
   ```
   Configure your API key: https://www.kaggle.com/settings (Create New Token), then place `kaggle.json` in `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows).

2. Download the dataset:
   ```bash
   cd backend
   mkdir -p data
   kaggle datasets download -d dinmkeljiame/doctamper -p data
   unzip data/doctamper.zip -d data/doctamper
   ```

3. Organize folders (if the zip structure differs). The loader expects one of:
   - `data/doctamper/authentic/` and `data/doctamper/tampered/`
   - `data/doctamper/0/` and `data/doctamper/1/`
   - `data/doctamper/train/authentic/` and `data/doctamper/train/tampered/`

   Place images (`.jpg`, `.png`, etc.) in the appropriate folder.

### Option 2: Official LMDB format

If you have the official DocTamper LMDB files (e.g. from the authors):

1. Install LMDB: `pip install lmdb`
2. Place the `.mdb` folder (e.g. `DocTamper_train`) in `backend/data/`.
3. Use `use_lmdb=True` when calling the loader (see `doctamper_loader.get_doctamper_data`).

## Training with DocTamper

### Command line

```bash
cd backend

# Use DocTamper from a folder
python train_model.py --doctamper ./data/doctamper --epochs 100 --batch-size 32

# Limit number of samples (for quick runs)
python train_model.py --doctamper ./data/doctamper --doctamper-max 5000 --epochs 50
```

### Environment variable

```bash
set DOCTAMPER_DATA=C:\path\to\doctamper
python train_model.py
```

### From code

```python
from train_model import ModelTrainer

trainer = ModelTrainer()
model, metrics = trainer.train(
    epochs=100,
    batch_size=32,
    doctamper_root="./data/doctamper",
    doctamper_max_samples=10000,  # optional cap
)
trainer.save_model(model, trainer.scaler, metrics)
```

## Folder structure (expected)

```
data/doctamper/
  authentic/     (or 0/ or train/authentic/)
    img001.jpg
    img002.png
    ...
  tampered/      (or 1/ or train/tampered/)
    img001.jpg
    ...
```

If your dataset uses different names (e.g. `real` and `fake`), you can call the loader directly:

```python
from doctamper_loader import load_doctamper_from_folders, get_doctamper_data

# Custom folder names
images, labels = load_doctamper_from_folders(
    "path/to/data",
    authentic_folder="real",
    tampered_folder="fake",
)
# Then use extract_features_from_images with your DocumentAnalyzer
```

## Notes

- Feature extraction uses the same pipeline as the app (image + OCR features → 80-dim vector). Training time depends on dataset size and CPU/GPU.
- DocTamper is for **non-commercial use**; check the dataset license and author requirements (e.g. educational email for password) if you use the official release.
- If `--doctamper` path is missing or invalid, training falls back to **synthetic data** automatically.
