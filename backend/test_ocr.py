"""Test OCR functionality"""
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr():
    print("Testing OCR functionality...")
    
    # Create a simple test image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use a default font
        text = "This is a test document for OCR"
        draw.text((10, 10), text, fill='black')
        
        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(img)
        print(f"Original text: {text}")
        print(f"Extracted text: {extracted_text.strip()}")
        
        # Get confidence
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print(f"Average confidence: {avg_confidence:.2f}%")
        
        if extracted_text.strip():
            print("SUCCESS: OCR is working correctly!")
            return True
        else:
            print("ERROR: OCR extracted no text")
            return False
            
    except Exception as e:
        print(f"ERROR: OCR test failed: {e}")
        return False

if __name__ == "__main__":
    test_ocr()
