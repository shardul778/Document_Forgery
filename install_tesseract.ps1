# Tesseract OCR Installation Script
Write-Host "Installing Tesseract OCR..." -ForegroundColor Green

# Download URL
$url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.5.0.20241111/tesseract-ocr-w64-setup-5.5.0.20241111.exe"
$output = "tesseract-installer.exe"
$installPath = "C:\Program Files\Tesseract-OCR"

try {
    Write-Host "Downloading Tesseract OCR from GitHub..."
    Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
    Write-Host "Download completed successfully." -ForegroundColor Green
    
    Write-Host "Installing Tesseract OCR..."
    $process = Start-Process -FilePath ".\$output" -ArgumentList "/S", "/D=$installPath" -Wait -PassThru
    
    if ($process.ExitCode -eq 0) {
        Write-Host "Installation completed successfully!" -ForegroundColor Green
        
        # Add to PATH for current session
        $env:PATH += ";$installPath"
        
        # Test installation
        try {
            $version = & "$installPath\tesseract.exe" --version
            Write-Host "Tesseract version: $version" -ForegroundColor Green
            Write-Host "Installation verified!" -ForegroundColor Green
        } catch {
            Write-Host "Warning: Could not verify installation. Please restart your terminal." -ForegroundColor Yellow
        }
        
        # Cleanup
        Remove-Item $output -ErrorAction SilentlyContinue
        
    } else {
        Write-Host "Installation failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    }
    
} catch {
    Write-Host "Error during installation: $_" -ForegroundColor Red
    Write-Host "Please download and install manually from: https://github.com/UB-Mannheim/tesseract/releases" -ForegroundColor Yellow
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
