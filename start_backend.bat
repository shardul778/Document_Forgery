@echo off
echo Starting Document Forgery Detection Backend...
cd backend
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing PyTorch (this may take a few minutes)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo.
    echo Common solutions:
    echo 1. Make sure you have Python 3.9 or higher
    echo 2. Install Visual C++ Build Tools if needed
    echo 3. Try installing packages one by one to identify the issue
    pause
    exit /b 1
)
echo Starting server...
python main.py
pause
