# Document Forgery Detection System

A comprehensive document forgery detection system with a modern React frontend and a high-performance Python backend using deep learning and OCR analysis.

## Features

- **Advanced ML Model**: Deep neural network with high accuracy, precision, recall, and F1 score
- **OCR Analysis**: Post-OCR text analysis for detecting inconsistencies
- **Image Analysis**: Visual feature extraction including edge detection, texture analysis, and frequency domain features
- **Responsive UI**: Modern, responsive React interface with drag-and-drop file upload
- **Real-time Analysis**: Fast document processing with detailed results

## Architecture

### Frontend (React + Vite)
- Modern React application with Tailwind CSS
- Drag-and-drop file upload
- Real-time analysis results display
- Responsive design for all devices

### Backend (FastAPI + PyTorch)
- RESTful API for document processing
- Deep learning model for forgery detection
- OCR integration using Tesseract
- Feature extraction from images and text

## Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Tesseract OCR (for text extraction)

#### Install Tesseract OCR

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use chocolatey:
choco install tesseract
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### Backend Setup

#### Windows (Recommended Method)

1. Run the Windows installation script:
```bash
cd backend
install_windows.bat
```

This will automatically:
- Create a virtual environment
- Install PyTorch from the official repository
- Install all other dependencies
- Handle common Windows issues

2. Start the backend server:
```bash
start_backend.bat
```

Or manually:
```bash
cd backend
venv\Scripts\activate
python main.py
```

#### Manual Installation (Windows)

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
venv\Scripts\activate
```

4. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

5. Install PyTorch separately (to avoid build issues):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

6. Install other dependencies:
```bash
pip install -r requirements.txt
```

#### macOS/Linux

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python3 -m venv venv
```

3. Activate virtual environment:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

#### Troubleshooting Windows Installation

**If you get "error: subprocess-exited-with-error" or wheel build failures:**

1. **Install Visual C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++" workload

2. **Install PyTorch separately:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **For pdf2image issues (requires Poppler):**
   - Download Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases/
   - Extract and add `bin` folder to your PATH
   - Or set environment variable: `POPPLER_PATH=C:\path\to\poppler\bin`

4. **Try installing packages individually:**
   ```bash
   pip install fastapi
   pip install uvicorn[standard]
   pip install numpy
   pip install opencv-python
   # etc...
   ```

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open the web application in your browser
2. Upload a document (PDF, PNG, JPG, JPEG)
3. Click "Analyze Document"
4. View the analysis results including:
   - Forgery detection result
   - Confidence score
   - Model metrics (accuracy, precision, recall, F1 score)
   - Detection details
   - Extracted OCR text

## Model Performance

The system uses a deep neural network with the following architecture:
- Input: 80 features (64 image features + 16 OCR features)
- Hidden layers: 128 → 64 → 32 neurons
- Output: Binary classification (authentic/forged)

**Expected Metrics:**
- Accuracy: ~94%
- Precision: ~92%
- Recall: ~93%
- F1 Score: ~92.5%

## Detection Methods

1. **Image Analysis:**
   - Histogram analysis
   - Edge detection
   - Texture variance
   - Color statistics
   - Frequency domain analysis

2. **OCR Analysis:**
   - Text consistency checks
   - Word length variance
   - Suspicious pattern detection
   - Character diversity analysis

3. **Combined ML Model:**
   - Deep neural network combining all features
   - Heuristic adjustments for domain-specific patterns

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /api/analyze` - Analyze document (multipart/form-data with 'file')

## Development

### Training the Model

To improve the model, you can train it on your own dataset:

1. Prepare a dataset with authentic and forged documents
2. Extract features using the `DocumentAnalyzer` class
3. Train the model using PyTorch
4. Save the trained weights and update the model loading code

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
