@echo off
echo ========================================
echo Document Forgery Detection - Windows Setup
echo ========================================
echo.

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
if not exist venv (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing PyTorch (CPU version)...
echo This may take several minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch installation failed. Trying alternative method...
    pip install torch torchvision
)

echo.
echo Step 5: Installing other dependencies...
pip install fastapi uvicorn[standard] python-multipart
pip install numpy opencv-python Pillow pytesseract pydantic
pip install pdf2image

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Note: pdf2image requires Poppler for Windows.
echo If you encounter errors with pdf2image, install Poppler:
echo Download from: https://github.com/oschwartz10612/poppler-windows/releases/
echo Extract and add to PATH, or set POPPLER_PATH environment variable.
echo.
pause
