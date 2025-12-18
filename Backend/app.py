"""
Backend API cho DocCleaner AI
Xử lý ảnh tài liệu với Pipeline V2 - Morphological Operations
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import utils
from utils.image_processing import ImageProcessor
from utils.ocr_engine import OCREngine
from utils.config import Config
from utils.experimental_evaluator import ExperimentalEvaluator
from utils.ai_processor import AIDocumentProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors
image_processor = ImageProcessor()
ai_processor = AIDocumentProcessor()
ocr_engine = OCREngine()
evaluator = ExperimentalEvaluator()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== HELPER FUNCTIONS =====

def base64_to_image(base64_string):
    """Convert base64 string to numpy array image"""
    try:
        # Remove data URL prefix if exists
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        logger.error(f"Error decoding base64: {str(e)}")
        return None


def image_to_base64(image):
    """Convert numpy array image to base64 string"""
    try:
        pil_img = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding to base64: {str(e)}")
        return None


# ===== API ENDPOINTS =====

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'service': 'DocCleaner AI Backend',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Main image processing endpoint
    Supports both JSON and FormData:
    JSON: { image: base64, settings: {...} }
    FormData: image file + settings JSON string
    """
    try:
        # Handle both JSON and FormData
        if request.content_type and 'multipart/form-data' in request.content_type:
            # FormData từ Frontend
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            img_data = file.read()
            img = Image.open(io.BytesIO(img_data))
            image = np.array(img)
            
            # Parse settings từ FormData
            settings_str = request.form.get('settings', '{}')
            import json
            settings = json.loads(settings_str)
        else:
            # JSON format
            data = request.json
            
            if not data or 'image' not in data:
                return jsonify({'error': 'No image provided'}), 400
            
            image = base64_to_image(data['image'])
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
            
            settings = data.get('settings', {})

        # Choose pipeline (default: simple for better results)
        pipeline = (settings.get('pipeline') or 'simple').lower()

        # Process image
        start_time = datetime.now()
        if pipeline in ('ai', 'ai_local', 'ai-local'):
            # AI Pipeline - Local Advanced (không cần internet)
            result = ai_processor.process_with_local_ai(image, settings)
        elif pipeline in ('ai_cloud', 'ai-cloud', 'ai_hf', 'huggingface'):
            # AI Pipeline - Cloud (Hugging Face)
            result = ai_processor.process_with_documents_restoration(image, settings)
        elif pipeline in ('simple', 'basic', 'easy'):
            result = image_processor.process_pipeline_simple(image, settings)
        elif pipeline in ('premium', 'pro', 'v4', 'advanced'):
            result = image_processor.process_pipeline_premium(image, settings)
        elif pipeline in ('handwriting', 'hw', 'ink', 'v5'):
            result = image_processor.process_pipeline_handwriting(image, settings)
        elif pipeline in ('robust', 'strong'):
            result = image_processor.process_pipeline_robust(image, settings)
        elif pipeline in ('v3', 'gaussdiff'):
            result = image_processor.process_pipeline_v3(image, settings)
        else:
            result = image_processor.process_pipeline_simple(image, settings)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert intermediate steps to base64 (without data: prefix for Frontend)
        intermediate_steps = {}
        if result.get('intermediate_steps'):
            for key, img in result['intermediate_steps'].items():
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    intermediate_steps[key] = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert final image to base64 (without data: prefix)
        final_img = result['final_image']
        if isinstance(final_img, np.ndarray):
            pil_img = Image.fromarray(final_img)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            processed_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            processed_base64 = ''
        
        return jsonify({
            'success': True,
            'processed_image': processed_base64,
            'processedImage': f"data:image/png;base64,{processed_base64}",  # Legacy support
            'intermediate_steps': intermediate_steps,
            'intermediateSteps': {k: f"data:image/png;base64,{v}" for k, v in intermediate_steps.items()},
            'width': final_img.shape[1] if isinstance(final_img, np.ndarray) else 0,
            'height': final_img.shape[0] if isinstance(final_img, np.ndarray) else 0,
            'metrics': result.get('metrics', {}),
            'stats': {
                'time': round(processing_time, 2),
                'width': final_img.shape[1] if isinstance(final_img, np.ndarray) else 0,
                'height': final_img.shape[0] if isinstance(final_img, np.ndarray) else 0,
                'steps': len(intermediate_steps) + 1,
                'pipeline': settings.get('pipeline', 'premium')
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/ocr', methods=['POST'])
def extract_text():
    """
    OCR endpoint - supports JSON and FormData
    JSON: { image: base64, lang: 'vie', provider: 'tesseract' }
    FormData: image file + provider + language
    """
    try:
        # Handle both JSON and FormData
        if request.content_type and 'multipart/form-data' in request.content_type:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            img_data = file.read()
            img = Image.open(io.BytesIO(img_data))
            image = np.array(img)
            
            lang = request.form.get('language', 'vie')
            provider = request.form.get('provider', 'tesseract')
        else:
            data = request.json
            
            if not data or 'image' not in data:
                return jsonify({'error': 'No image provided'}), 400
            
            image = base64_to_image(data['image'])
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
            
            lang = data.get('lang', 'vie')
            provider = data.get('provider', 'tesseract')
        
        # Perform OCR
        start_time = datetime.now()
        result = ocr_engine.extract_text(image, lang=lang, provider=provider)
        ocr_time = (datetime.now() - start_time).total_seconds() * 1000
        
        status = 200 if not result.get('error') else 500
        return jsonify({
            'success': result.get('error') is None,
            'provider': result.get('provider', provider),
            'text': result.get('text', ''),
            'confidence': result.get('confidence', 0),
            'time': round(ocr_time, 2),
            'error': result.get('error')
        }), status
        
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_quality():
    """
    Evaluate image quality metrics
    Expects JSON with:
    - original: base64 encoded original image
    - processed: base64 encoded processed image
    """
    try:
        data = request.json
        
        if not data or 'original' not in data or 'processed' not in data:
            return jsonify({'error': 'Both original and processed images required'}), 400
        
        # Decode images
        original = base64_to_image(data['original'])
        processed = base64_to_image(data['processed'])
        
        if original is None or processed is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Calculate metrics
        metrics = image_processor.calculate_metrics(original, processed)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error evaluating quality: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get default configuration"""
    return jsonify({
        'success': True,
        'config': Config.get_default_config()
    })


@app.route('/api/config/presets', methods=['GET'])
def get_presets():
    """Get predefined configuration presets"""
    return jsonify({
        'success': True,
        'presets': Config.get_presets()
    })


@app.route('/api/ocr/providers', methods=['GET'])
def get_ocr_providers():
    """Get available OCR providers and their status"""
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    pytorch_compatible = sys.version_info < (3, 13)
    
    providers = {
        'tesseract': {
            'name': 'Tesseract OCR',
            'available': True,
            'description': 'Local OCR engine, supports Vietnamese',
            'requires': 'tesseract-ocr installed on system'
        },
        'ocrspace': {
            'name': 'OCR.space API',
            'available': bool(os.getenv('OCRSPACE_API_KEY')),
            'description': 'Cloud OCR service, high accuracy',
            'requires': 'OCRSPACE_API_KEY environment variable'
        },
        'google_vision': {
            'name': 'Google Cloud Vision',
            'available': bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')),
            'description': 'Google Cloud Vision API, excellent accuracy',
            'requires': 'GOOGLE_APPLICATION_CREDENTIALS and google-cloud-vision package'
        },
        'easyocr': {
            'name': 'EasyOCR',
            'available': pytorch_compatible,
            'description': 'Deep learning OCR, supports 80+ languages',
            'requires': f'Python ≤3.12 (current: {python_version})'
        },
        'vietocr': {
            'name': 'VietOCR',
            'available': pytorch_compatible,
            'description': 'Transformer OCR optimized for Vietnamese handwriting',
            'requires': f'Python ≤3.12 (current: {python_version})'
        }
    }
    
    return jsonify({
        'success': True,
        'python_version': python_version,
        'providers': providers
    })


@app.route('/api/evaluate/report', methods=['GET'])
def get_evaluation_report():
    """
    Get comprehensive evaluation report
    Task requirement: Viết chương trình thực nghiệm đánh giá kết quả
    """
    try:
        report = evaluator.generate_report()
        
        if report is None:
            return jsonify({
                'success': False,
                'message': 'No evaluation data available'
            }), 404
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate/comparison', methods=['GET'])
def get_comparison_table():
    """
    Get comparison table for all evaluated images
    """
    try:
        df = evaluator.generate_comparison_table()
        
        if df is None:
            return jsonify({
                'success': False,
                'message': 'No evaluation data available'
            }), 404
        
        return jsonify({
            'success': True,
            'table': df.to_dict('records'),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        logger.error(f"Error generating comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch/process', methods=['POST'])
def batch_process():
    """
    Batch processing endpoint
    Xử lý nhiều ảnh cùng lúc và trả về kết quả tổng hợp
    
    Expects JSON with:
    - images: array of base64 encoded images
    - settings: processing configuration
    """
    try:
        data = request.json
        
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        images_data = data['images']
        if not isinstance(images_data, list) or len(images_data) == 0:
            return jsonify({'error': 'Images must be a non-empty array'}), 400
        
        settings = data.get('settings', {})
        
        logger.info(f"Processing batch of {len(images_data)} images")
        
        results = []
        failed = 0
        
        for idx, img_data in enumerate(images_data):
            try:
                # Decode image
                image = base64_to_image(img_data)
                if image is None:
                    failed += 1
                    continue
                
                # Process image with V3
                result = image_processor.process_pipeline_v3(image, settings)
                
                # Convert to base64
                final_image = image_to_base64(result['final_image'])
                
                results.append({
                    'index': idx,
                    'success': True,
                    'processedImage': final_image,
                    'metrics': result['metrics'],
                    'size': {
                        'width': result['final_image'].shape[1],
                        'height': result['final_image'].shape[0]
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing image {idx}: {str(e)}")
                failed += 1
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate statistics
        successful_results = [r for r in results if r.get('success')]
        if successful_results:
            psnr_values = [r['metrics']['psnr'] for r in successful_results]
            ssim_values = [r['metrics']['ssim'] for r in successful_results]
            
            statistics = {
                'psnr': {
                    'mean': float(np.mean(psnr_values)),
                    'std': float(np.std(psnr_values)),
                    'min': float(np.min(psnr_values)),
                    'max': float(np.max(psnr_values))
                },
                'ssim': {
                    'mean': float(np.mean(ssim_values)),
                    'std': float(np.std(ssim_values)),
                    'min': float(np.min(ssim_values)),
                    'max': float(np.max(ssim_values))
                }
            }
        else:
            statistics = {}
        
        return jsonify({
            'success': True,
            'total': len(images_data),
            'successful': len(successful_results),
            'failed': failed,
            'results': results,
            'statistics': statistics
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===== ERROR HANDLERS =====

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ===== MAIN =====

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    
    logger.info(f"Starting DocCleaner AI Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
