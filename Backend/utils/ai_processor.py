"""
AI Document Processing Module
Tích hợp các model AI cho làm sạch văn bản/tài liệu

Models:
1. Documents Restoration (Hugging Face) - Làm sạch và phục hồi tài liệu
2. NAFNet - Image denoising/restoration
"""

import cv2
import numpy as np
import base64
import tempfile
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import gradio_client
try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("gradio_client not installed. AI processing unavailable.")


class AIDocumentProcessor:
    """
    Processor sử dụng AI models từ Hugging Face để làm sạch tài liệu
    """
    
    def __init__(self):
        self.intermediate_steps = {}
        self._clients = {}
    
    def _get_client(self, space_id: str) -> "Client":
        """Lazy load Gradio client with timeout"""
        if not GRADIO_AVAILABLE:
            raise RuntimeError("gradio_client not installed")
        
        if space_id not in self._clients:
            logger.info(f"Connecting to Hugging Face Space: {space_id}")
            # Increase timeout to 300 seconds (5 minutes)
            self._clients[space_id] = Client(space_id, download_files=True, verbose=False)
        return self._clients[space_id]
    
    def process_with_documents_restoration(self, image: np.ndarray, settings: dict) -> dict:
        """
        Xử lý ảnh với Documents Restoration model từ Hugging Face
        
        Space: qubvel-hf/documents-restoration
        Chức năng: Làm sạch và phục hồi ảnh tài liệu
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== AI PIPELINE: Documents Restoration ===")
            
            # Lưu ảnh gốc
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            self.intermediate_steps['0_original'] = gray.copy()
            
            # Resize ảnh nếu quá lớn (giảm thời gian upload)
            h, w = image.shape[:2]
            max_size = 2048
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image: {w}x{h} → {new_w}x{new_h}")
            
            # Lưu ảnh tạm để upload
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                # Compress để giảm kích thước file
                cv2.imwrite(tmp_path, 
                           image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 6])  # Compression level 6 (0-9)
            
            try:
                # Gọi Hugging Face Space
                client = self._get_client("qubvel-hf/documents-restoration")
                
                # API endpoint: /run_tasks
                # Available tasks: dewarping, deshadowing, appearance, deblurring, binarization
                tasks = settings.get('tasks', ['appearance', 'binarization'])
                
                logger.info(f"Running AI tasks: {tasks}")
                logger.info(f"Uploading image: {tmp_path} (size: {os.path.getsize(tmp_path)/1024:.1f} KB)")
                
                result = client.predict(
                    image=handle_file(tmp_path),
                    tasks=tasks,
                    api_name="/run_tasks"
                )
                
                logger.info(f"AI processing complete. Result: {type(result)}")
                
                # Đọc kết quả
                if isinstance(result, str) and os.path.exists(result):
                    # Kết quả là đường dẫn file
                    processed = cv2.imread(result, cv2.IMREAD_GRAYSCALE)
                    if processed is None:
                        processed = cv2.imread(result)
                        if processed is not None and len(processed.shape) == 3:
                            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Unexpected result type: {type(result)}")
                
                if processed is None:
                    raise ValueError("Failed to read processed image")
                
                self.intermediate_steps['1_ai_restored'] = processed.copy()
                
                # Post-processing để đảm bảo kết quả tốt
                # Áp dụng các bước morphological theo yêu cầu
                processed = self._post_process(processed, settings)
                
                self.intermediate_steps['2_final'] = processed.copy()
                
                # Calculate metrics
                metrics = self._calculate_metrics(gray, processed)
                
                logger.info(f"Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}")
                logger.info("=== AI PIPELINE COMPLETE ===")
                
                return {
                    'final_image': processed,
                    'intermediate_steps': self.intermediate_steps,
                    'metrics': metrics,
                    'original_gray': gray,
                    'processing_summary': {
                        'pipeline_version': 'AI_DOCUMENTS_RESTORATION',
                        'tasks': tasks,
                        'steps_executed': len(self.intermediate_steps),
                    }
                }
                
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except TimeoutError as e:
            logger.error(f"Timeout in AI pipeline: {str(e)}")
            raise RuntimeError(
                "Hugging Face Space timeout. Reasons:\n"
                "1. Image too large (resize to < 2048px)\n"
                "2. Space is busy or cold starting\n"
                "3. Network connection slow\n"
                "Try: Use 'AI Local' instead (faster, offline)"
            )
        except Exception as e:
            logger.error(f"Error in AI pipeline: {str(e)}")
            # Provide helpful error message
            if "timeout" in str(e).lower():
                raise RuntimeError(
                    f"Hugging Face Space timeout: {str(e)}\n"
                    "Try: Use 'AI Local' pipeline instead (faster, no internet needed)"
                )
            raise
    
    def process_with_local_ai(self, image: np.ndarray, settings: dict) -> dict:
        """
        Pipeline AI kết hợp - sử dụng các kỹ thuật tiên tiến local
        Không cần internet, nhanh hơn
        
        Pipeline:
        1. Denoise với Non-local Means
        2. Document Binarization thông minh
        3. Morphological cleanup
        4. Top-hat/Black-hat để loại nền
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== AI PIPELINE: Local Advanced ===")
            
            # Step 0: Grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            original_gray = gray.copy()
            self.intermediate_steps['0_original'] = gray.copy()
            h, w = gray.shape
            logger.info(f"Input: {w}x{h}")
            
            # Step 1: Non-local Means Denoising (AI-inspired denoising)
            denoise_strength = int(settings.get('denoiseStrength', 10))
            denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)
            self.intermediate_steps['1_denoised'] = denoised.copy()
            logger.info("Step 1: Non-local Means Denoising")
            
            # Step 2: Background removal với Top-hat/Black-hat
            bg_mode = settings.get('bgMode', 'auto')
            kernel_size = int(settings.get('bgKernel', 25))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            if bg_mode == 'auto':
                # Auto detect: nếu mean > 127 thì nền sáng, dùng black-hat
                bg_mode = 'blackhat' if np.mean(denoised) > 127 else 'tophat'
            
            if bg_mode == 'blackhat':
                # Nền sáng, vết đen → Black-hat loại vết đen
                bg_feature = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
                enhanced = cv2.subtract(denoised, bg_feature)
            elif bg_mode == 'tophat':
                # Nền tối → Top-hat làm sáng
                bg_feature = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
                enhanced = cv2.add(denoised, bg_feature)
            else:
                enhanced = denoised.copy()
                bg_feature = np.zeros_like(denoised)
            
            # Normalize
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            self.intermediate_steps['2_bg_removed'] = enhanced.copy()
            self.intermediate_steps['2b_bg_feature'] = bg_feature.copy()
            logger.info(f"Step 2: Background removal ({bg_mode})")
            
            # Step 3: Contrast enhancement với CLAHE
            clahe_clip = float(settings.get('claheClip', 2.0))
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            contrast = clahe.apply(enhanced.astype(np.uint8))
            self.intermediate_steps['3_contrast'] = contrast.copy()
            logger.info("Step 3: CLAHE contrast enhancement")
            
            # Step 4: Binarization
            threshold_method = settings.get('thresholdMethod', 'otsu')
            if threshold_method == 'otsu':
                _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_method == 'adaptive':
                block = int(settings.get('adaptiveBlock', 31))
                block = max(3, block if block % 2 == 1 else block + 1)
                c = int(settings.get('adaptiveC', 10))
                binary = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, block, c)
            else:
                _, binary = cv2.threshold(contrast, 127, 255, cv2.THRESH_BINARY)
            
            self.intermediate_steps['4_binary'] = binary.copy()
            logger.info(f"Step 4: Binarization ({threshold_method})")
            
            # Step 5: Opening (loại nhiễu nhỏ)
            k_open = int(settings.get('openingKernel', 2))
            if k_open >= 2:
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            else:
                opened = binary.copy()
            self.intermediate_steps['5_opening'] = opened.copy()
            logger.info(f"Step 5: Opening (k={k_open})")
            
            # Step 6: Closing (nối nét chữ)
            k_close = int(settings.get('closingKernel', 2))
            if k_close >= 2:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
            else:
                closed = opened.copy()
            self.intermediate_steps['6_closing'] = closed.copy()
            logger.info(f"Step 6: Closing (k={k_close})")
            
            # Step 7: Auto-invert nếu cần (đảm bảo nền trắng, chữ đen)
            white_ratio = np.mean(closed == 255)
            if white_ratio < 0.4:
                final = cv2.bitwise_not(closed)
                logger.info("Step 7: Auto-inverted (white bg, black text)")
            else:
                final = closed.copy()
                logger.info("Step 7: No invert needed")
            
            self.intermediate_steps['7_inverted'] = final.copy()
            
            # Step 8: OCR Enhancement - làm chữ đậm hơn và sắc nét
            # Dilation nhẹ để chữ rõ hơn
            k_dilate = 2
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dilate, k_dilate))
            dilated = cv2.dilate(final, kernel_dilate, iterations=1)
            self.intermediate_steps['8_dilated'] = dilated.copy()
            
            # Đảm bảo độ tương phản tuyệt đối: chỉ có 0 và 255
            _, final_binary = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
            
            # Làm sạch nhiễu cực nhỏ (connected components < 10 pixels)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                255 - final_binary, connectivity=8
            )
            
            # Remove tiny noise components
            min_size = 10
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_size:
                    final_binary[labels == i] = 255
            
            self.intermediate_steps['9_final_ocr'] = final_binary.copy()
            logger.info("Step 8-9: OCR Enhancement (dilation + noise removal)")
            
            final = final_binary
            
            # Calculate metrics
            metrics = self._calculate_metrics(original_gray, final)
            
            logger.info(f"Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}")
            logger.info("=== AI LOCAL PIPELINE COMPLETE ===")
            
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': {
                    'pipeline_version': 'AI_LOCAL_ADVANCED',
                    'denoise_strength': denoise_strength,
                    'bg_mode': bg_mode,
                    'threshold_method': threshold_method,
                    'steps_executed': len(self.intermediate_steps),
                }
            }
            
        except Exception as e:
            logger.error(f"Error in local AI pipeline: {str(e)}")
            raise
    
    def _post_process(self, image: np.ndarray, settings: dict) -> np.ndarray:
        """Post-processing sau khi AI xử lý"""
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Nếu đã chọn binarization trong tasks thì không cần binarize lại
        tasks = settings.get('tasks', [])
        
        # Optional additional binarization (nếu chưa có trong tasks)
        if settings.get('binarize', False) and 'binarization' not in tasks:
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional morphological cleanup (light)
        if settings.get('morphCleanup', False):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Auto-invert if needed
        white_ratio = np.mean(image == 255)
        if white_ratio < 0.4:
            image = cv2.bitwise_not(image)
        
        return image
    
    def _calculate_metrics(self, original: np.ndarray, processed: np.ndarray) -> dict:
        """Calculate quality metrics"""
        try:
            from skimage.metrics import structural_similarity as ssim_func
            from skimage.metrics import peak_signal_noise_ratio as psnr_func
            
            if original.shape != processed.shape:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
            psnr_val = psnr_func(original, processed, data_range=255)
            ssim_val = ssim_func(original, processed, data_range=255)
            mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
            
            return {
                'psnr': round(float(psnr_val), 2),
                'ssim': round(float(ssim_val), 4),
                'mse': round(float(mse), 2)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'psnr': 0, 'ssim': 0, 'mse': 0}


# Singleton instance
ai_processor = AIDocumentProcessor()
