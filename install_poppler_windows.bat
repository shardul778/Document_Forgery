@echo off
echo ========================================
echo Poppler Installation Helper for Windows
echo ========================================
echo.
echo Poppler is required for PDF processing in the Document Forgery Detection system.
echo.
echo Installation Options:
echo.
echo Option 1: Download and Install (Recommended)
echo ----------------------------------------
echo 1. Go to: https://github.com/oschwartz10612/poppler-windows/releases/
echo 2. Download the latest release (e.g., Release-XX.XX.X-X.zip)
echo 3. Extract the ZIP file to a location like: C:\poppler
echo 4. Add to PATH:
echo    - Open System Properties ^> Environment Variables
echo    - Edit PATH variable
echo    - Add: C:\poppler\Library\bin
echo    - Click OK and restart terminal
echo.
echo Option 2: Use Chocolatey (if installed)
echo ----------------------------------------
echo choco install poppler
echo.
echo Option 3: Use winget (Windows 10/11)
echo ----------------------------------------
echo winget install --id Poppler.Poppler
echo.
echo ========================================
echo.
echo After installation, verify by running:
echo pdftoppm -h
echo.
echo If poppler is not found, make sure it's added to PATH.
echo.
pause
