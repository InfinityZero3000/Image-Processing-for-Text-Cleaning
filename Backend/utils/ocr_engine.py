"""OCR Engine Module

Supports multiple providers:
- local Tesseract (default)
- EasyOCR (Deep learning-based, supports Vietnamese) - requires Python ≤3.12
- VietOCR (Transformer-based, excellent for Vietnamese handwriting) - requires Python ≤3.12
- OCR.space HTTP API (needs env OCRSPACE_API_KEY)
- Google Cloud Vision API (needs env GOOGLE_APPLICATION_CREDENTIALS)
"""

import os
import io
import logging
import base64
from typing import Dict, Any, Optional

import pytesseract
from PIL import Image
import numpy as np
import cv2
import requests

logger = logging.getLogger(__name__)

# Lazy load OCR engines to avoid import errors and save memory
_vietocr_detector = None
_vietocr_config = None
_easyocr_reader = None


def _get_easyocr_reader(langs=['vi', 'en']):
    """Lazy load EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(langs, gpu=False)
            logger.info(f"EasyOCR reader loaded with languages: {langs}")
        except ImportError as e:
            logger.error(f"EasyOCR not installed or Python version incompatible: {e}")
            raise RuntimeError("EasyOCR not available. Requires Python ≤3.12. Use 'tesseract' or 'ocrspace' instead.")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise
    return _easyocr_reader


def _get_vietocr_detector():
    """Lazy load VietOCR detector to save memory and startup time."""
    global _vietocr_detector, _vietocr_config
    if _vietocr_detector is None:
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            
            # Use VGG-Transformer model (best for Vietnamese)
            _vietocr_config = Cfg.load_config_from_name('vgg_transformer')
            _vietocr_config['cnn']['pretrained'] = True
            _vietocr_config['device'] = 'cpu'
            _vietocr_config['predictor']['beamsearch'] = False
            
            _vietocr_detector = Predictor(_vietocr_config)
            logger.info("VietOCR detector loaded successfully")
        except ImportError as e:
            logger.error(f"VietOCR not installed or Python version incompatible: {e}")
            raise RuntimeError("VietOCR not available. Requires Python ≤3.12. Use 'tesseract' or 'ocrspace' instead.")
        except Exception as e:
            logger.error(f"Failed to load VietOCR: {e}")
            raise
    return _vietocr_detector


class OCREngine:
    """OCR Engine supporting multiple providers including cloud APIs"""
    
    def __init__(self):
        # Tesseract configuration
        self.config = '--oem 3 --psm 6'  # LSTM OCR Engine, Uniform block of text
        self.ocrspace_api_key = os.getenv('OCRSPACE_API_KEY')
        self.google_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    def _to_pil(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                return Image.fromarray(image)
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    def _image_bytes(self, image) -> bytes:
        pil = self._to_pil(image)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        return buf.getvalue()

    def _run_tesseract(self, image, lang='vie') -> Dict[str, Any]:
        pil_image = self._to_pil(image)
        text = pytesseract.image_to_string(pil_image, lang=lang, config=self.config)
        data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if str(conf).strip() not in ('', '-1')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        return {
            'provider': 'tesseract',
            'text': text.strip(),
            'confidence': round(avg_confidence, 2)
        }

    def _run_ocrspace(self, image, lang='vie') -> Dict[str, Any]:
        if not self.ocrspace_api_key:
            raise RuntimeError('OCRSPACE_API_KEY is not set')

        # OCR.space language mapping
        # OCR.space uses specific codes: vie -> Vietnamese, eng -> English, etc.
        # But they don't support Vietnamese in free tier, fallback to English
        lang_map = {
            'vie': 'eng',  # Vietnamese not supported, use English
            'vi': 'eng',
            'en': 'eng',
            'eng': 'eng',
        }
        ocrspace_lang = lang_map.get(lang.lower(), 'eng')
        
        try:
            img_bytes = self._image_bytes(image)
            files = {'file': ('image.png', img_bytes, 'image/png')}
            data = {
                'language': ocrspace_lang,
                'isOverlayRequired': False,
                'OCREngine': 2,  # OCR Engine 2 is more accurate
            }
            headers = {'apikey': self.ocrspace_api_key}

            resp = requests.post('https://api.ocr.space/parse/image', files=files, data=data, headers=headers, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to OCR.space API. Please check:\n"
                f"1. Internet connection\n"
                f"2. DNS resolution (api.ocr.space)\n"
                f"3. Firewall/VPN blocking the connection\n"
                f"Original error: {str(e)}"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("OCR.space API timeout after 60 seconds. Try again later.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid OCR.space API key. Check your OCRSPACE_API_KEY in .env")
            raise RuntimeError(f"OCR.space API error: {e.response.status_code} - {e.response.text}")

        if not payload.get('IsErroredOnProcessing') and payload.get('ParsedResults'):
            parsed = payload['ParsedResults'][0]
            text = parsed.get('ParsedText', '').strip()
            # OCR.space does not always give word confidences; use OCRExitCode heuristic
            conf = parsed.get('FileParseExitCode', 1)
            return {
                'provider': 'ocrspace',
                'text': text,
                'confidence': conf
            }

        raise RuntimeError(f"OCR.space error: {payload.get('ErrorMessage') or 'Unknown error'}")

    def _run_google_vision(self, image) -> Dict[str, Any]:
        """Run Google Cloud Vision API for OCR."""
        # Check if credentials are configured
        if not self.google_credentials:
            raise RuntimeError(
                "Google Cloud credentials not found.\n"
                "Please set GOOGLE_APPLICATION_CREDENTIALS environment variable.\n"
                "See: https://cloud.google.com/docs/authentication/getting-started"
            )
        
        try:
            from google.cloud import vision
        except ImportError:
            raise RuntimeError("google-cloud-vision not installed. Run: pip install google-cloud-vision")
        
        try:
            client = vision.ImageAnnotatorClient()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Vision client: {str(e)}\n"
                "Please check your GOOGLE_APPLICATION_CREDENTIALS setting."
            )
        
        # Convert image to bytes
        img_bytes = self._image_bytes(image)
        
        vision_image = vision.Image(content=img_bytes)
        
        # Use document text detection for better results
        response = client.document_text_detection(image=vision_image)
        
        if response.error.message:
            raise RuntimeError(f"Google Vision API error: {response.error.message}")
        
        full_text = response.full_text_annotation.text if response.full_text_annotation else ''
        
        # Calculate average confidence from pages
        confidences = []
        if response.full_text_annotation and response.full_text_annotation.pages:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    confidences.append(block.confidence)
        
        avg_conf = sum(confidences) / len(confidences) * 100 if confidences else 90.0
        
        return {
            'provider': 'google_vision',
            'text': full_text.strip(),
            'confidence': round(avg_conf, 2)
        }

    def _run_easyocr(self, image, langs=['vi', 'en']) -> Dict[str, Any]:
        """Run EasyOCR - Deep learning OCR supporting 80+ languages including Vietnamese."""
        reader = _get_easyocr_reader(langs)
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # EasyOCR expects BGR or grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Check if RGB, convert to BGR
            pass  # EasyOCR handles both
        
        # Run OCR
        results = reader.readtext(img_np, detail=1, paragraph=True)
        
        # Extract text and confidence
        texts = []
        confidences = []
        for result in results:
            if len(result) >= 2:
                bbox, text = result[0], result[1]
                conf = result[2] if len(result) > 2 else 0.9
                texts.append(text)
                confidences.append(conf)
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'provider': 'easyocr',
            'text': '\n'.join(texts),
            'confidence': round(avg_conf * 100, 2),
            'detections': len(results)
        }

    def _run_vietocr(self, image) -> Dict[str, Any]:
        """Run VietOCR - Transformer-based OCR optimized for Vietnamese."""
        detector = _get_vietocr_detector()
        pil_image = self._to_pil(image)
        
        # VietOCR works best with RGB images
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Predict text
        text = detector.predict(pil_image)
        
        return {
            'provider': 'vietocr',
            'text': text.strip() if text else '',
            'confidence': 95.0  # VietOCR doesn't provide confidence, but it's generally very accurate
        }

    def _run_vietocr_lines(self, image) -> Dict[str, Any]:
        """Run VietOCR with line detection for multi-line documents."""
        try:
            detector = _get_vietocr_detector()
            pil_image = self._to_pil(image)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy for line detection
            img_np = np.array(pil_image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
            
            # Detect text lines using projection profile
            lines = self._detect_text_lines(gray)
            
            if not lines:
                # Fallback to single prediction
                text = detector.predict(pil_image)
                return {
                    'provider': 'vietocr',
                    'text': text.strip() if text else '',
                    'confidence': 95.0,
                    'lines': 1
                }
            
            # OCR each line
            all_texts = []
            for y1, y2 in lines:
                # Add padding
                pad = 5
                y1_pad = max(0, y1 - pad)
                y2_pad = min(img_np.shape[0], y2 + pad)
                
                line_img = pil_image.crop((0, y1_pad, pil_image.width, y2_pad))
                line_text = detector.predict(line_img)
                if line_text and line_text.strip():
                    all_texts.append(line_text.strip())
            
            return {
                'provider': 'vietocr',
                'text': '\n'.join(all_texts),
                'confidence': 95.0,
                'lines': len(all_texts)
            }
        except Exception as e:
            logger.error(f"VietOCR lines error: {e}")
            # Fallback to simple prediction
            return self._run_vietocr(image)

    def _detect_text_lines(self, gray: np.ndarray, min_height: int = 15) -> list:
        """Detect text lines using horizontal projection profile."""
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        h_proj = np.sum(binary, axis=1)
        
        # Find line boundaries
        threshold = h_proj.max() * 0.1
        in_line = False
        lines = []
        start = 0
        
        for i, val in enumerate(h_proj):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                if i - start >= min_height:
                    lines.append((start, i))
                in_line = False
        
        # Handle last line
        if in_line and len(h_proj) - start >= min_height:
            lines.append((start, len(h_proj)))
        
        return lines

    def extract_text(self, image, lang='vie', provider='tesseract'):
        """Extract text using selected provider.
        
        Available providers:
        - tesseract: Local Tesseract OCR (default)
        - easyocr: Deep learning OCR (requires Python ≤3.12)
        - vietocr: Vietnamese specialized OCR (requires Python ≤3.12)
        - vietocr_lines: VietOCR with line detection
        - ocrspace: OCR.space cloud API (needs OCRSPACE_API_KEY)
        - google_vision: Google Cloud Vision API (needs GOOGLE_APPLICATION_CREDENTIALS)
        """
        try:
            provider = (provider or 'tesseract').lower()
            if provider == 'easyocr':
                langs = ['vi', 'en'] if lang == 'vie' else [lang[:2], 'en']
                return self._run_easyocr(image, langs=langs)
            elif provider == 'vietocr':
                return self._run_vietocr(image)
            elif provider == 'vietocr_lines':
                return self._run_vietocr_lines(image)
            elif provider == 'ocrspace':
                return self._run_ocrspace(image, lang=lang)
            elif provider in ('google_vision', 'google', 'gcv'):
                return self._run_google_vision(image)
            return self._run_tesseract(image, lang=lang)
        except Exception as e:
            logger.error(f"Error extracting text with provider={provider}: {str(e)}")
            return {
                'provider': provider,
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
