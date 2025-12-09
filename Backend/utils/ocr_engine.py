"""
OCR Engine Module
Integrates with Tesseract OCR
"""

import pytesseract
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR Engine using Tesseract"""
    
    def __init__(self):
        # Tesseract configuration
        self.config = '--oem 3 --psm 6'  # LSTM OCR Engine, Uniform block of text
    
    def extract_text(self, image, lang='vie'):
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image: numpy array (grayscale or RGB)
            lang: language code ('vie', 'eng', etc.)
        
        Returns:
            dict with 'text' and 'confidence'
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    pil_image = Image.fromarray(image)
                else:  # RGB
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, lang=lang, config=self.config)
            
            # Get confidence
            data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': round(avg_confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return {
                'text': '',
                'confidence': 0,
                'error': str(e)
            }
    
    def extract_with_boxes(self, image, lang='vie'):
        """
        Extract text with bounding boxes
        
        Returns:
            dict with 'text', 'boxes', and 'confidences'
        """
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Get detailed data
            data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Extract words with boxes
            words = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    words.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'box': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        }
                    })
            
            full_text = ' '.join([w['text'] for w in words])
            
            return {
                'text': full_text,
                'words': words,
                'count': len(words)
            }
            
        except Exception as e:
            logger.error(f"Error extracting with boxes: {str(e)}")
            return {
                'text': '',
                'words': [],
                'count': 0,
                'error': str(e)
            }
