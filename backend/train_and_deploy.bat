@echo off
echo ========================================
echo Training Document Forgery Detection Model
echo ========================================
echo.
echo This will train the model from scratch with synthetic data.
echo Training may take several minutes...
echo.
REM Change to directory where this script lives (backend)
cd /d "%~dp0"
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo Note: venv not found in backend. Using system Python.
)
echo.
echo Starting training...
python train_model.py
echo.
echo ========================================
echo Training complete!
echo ========================================
echo.
echo The trained model has been saved to: backend\models\
echo Restart the backend server to use the trained model.
echo.
pause
