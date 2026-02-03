"""Check if Tesseract OCR is properly installed and configured"""
import os
import sys

def check_tesseract():
    """Check Tesseract installation"""
    print("Checking Tesseract OCR installation...")
    print("=" * 50)
    
    try:
        import pytesseract
        
        # Try to get version
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract version: {version}")
        except Exception as e:
            print(f"✗ Cannot get Tesseract version: {e}")
            return False
        
        # Check common Windows paths
        if os.name == 'nt':
            print("\nChecking common installation paths:")
            paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            
            found = False
            for path in paths:
                if os.path.exists(path):
                    print(f"✓ Found at: {path}")
                    found = True
                    # Set it
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
            
            if not found:
                print("✗ Not found in common paths")
                print("  Current tesseract_cmd:", pytesseract.pytesseract.tesseract_cmd)
        
        # Test OCR
        print("\nTesting OCR functionality...")
        try:
            from PIL import Image
            import numpy as np
            # Create a simple test image
            test_img = Image.new('RGB', (100, 30), color='white')
            text = pytesseract.image_to_string(test_img)
            print("✓ OCR test successful")
        except Exception as e:
            print(f"✗ OCR test failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("✓ Tesseract OCR is properly configured!")
        return True
        
    except ImportError:
        print("✗ pytesseract is not installed")
        print("  Install with: pip install pytesseract")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = check_tesseract()
    if not success:
        print("\nTo install Tesseract OCR:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install and add to PATH")
        print("3. Or set TESSERACT_CMD environment variable")
        sys.exit(1)
