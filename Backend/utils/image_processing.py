"""
Image Processing Module - Pipeline V2
Morphological operations for document cleaning
"""

import cv2
import numpy as np
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Xử lý ảnh với Pipeline V2 - Fixed Background Removal"""
    
    def __init__(self):
        self.intermediate_steps = {}
    
    def process_pipeline_v2(self, image, settings):
        """
        Complete Pipeline V2 - Xử lý ảnh văn bản theo đúng yêu cầu
        
        Các bước xử lý (theo bảng yêu cầu):
        1. Tiền xử lý:
           - Chuyển ảnh sang thang xám (grayscale)
           - Dùng ngưỡng (Otsu hoặc adaptive threshold) để nhị phân ảnh
        
        2. Làm sạch nhiễu:
           - Sử dụng phép mở (opening) để loại bỏ các điểm trắng nhỏ (nhiễu)
           - Kernel nhỏ, ví dụ 2×2 hoặc 3×3
        
        3. Làm liền nét chữ:
           - Dùng phép đóng (closing) để lấp khoảng trống, nối các đoạn chữ đứt gãy
        
        4. Loại bỏ nền và vết bẩn:
           - Dùng black-hat hoặc top-hat tùy loại nền:
             * Nếu nền tối → dùng top-hat
             * Nếu nền sáng có vết đen → dùng black-hat
        
        5. Tăng cường hiển thị và lưu kết quả
        
        6. Đánh giá chất lượng (PSNR, SSIM, MSE)
        """
        try:
            self.intermediate_steps = {}
            logger.info(f"Bắt đầu xử lý ảnh với settings: {settings}")
            
            # BƯỚC 1: Tiền xử lý - Chuyển sang thang xám
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                logger.info("Chuyển đổi ảnh màu sang grayscale")
            else:
                gray = image.copy()
                logger.info("Ảnh đã ở dạng grayscale")
            
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray
            
            # BƯỚC 2: Tiền xử lý - Nhị phân hóa (Otsu hoặc Adaptive Threshold)
            threshold_method = settings.get('thresholdMethod', 'otsu')
            binary = self.apply_threshold(gray, method=threshold_method)
            self.intermediate_steps['2_threshold'] = binary
            logger.info(f"Áp dụng threshold method: {threshold_method}")
            
            # BƯỚC 3: Làm sạch nhiễu - Opening (loại bỏ điểm trắng nhỏ, nhiễu salt)
            kernel_opening = settings.get('kernelOpening', 2)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_opening, kernel_opening))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            self.intermediate_steps['3_opening'] = opened
            logger.info(f"Áp dụng Opening với kernel {kernel_opening}x{kernel_opening}")
            
            # BƯỚC 4: Làm liền nét chữ - Closing (nối các đoạn chữ bị đứt gãy)
            kernel_closing = settings.get('kernelClosing', 3)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_closing, kernel_closing))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
            self.intermediate_steps['4_closing'] = closed
            logger.info(f"Áp dụng Closing với kernel {kernel_closing}x{kernel_closing}")
            
            # BƯỚC 5: Loại bỏ nền và vết bẩn
            # Áp dụng Black-hat hoặc Top-hat tùy theo loại nền
            bg_method = settings.get('backgroundRemoval', 'auto')
            if bg_method != 'none':
                kernel_bg = settings.get('backgroundKernel', 15)
                kernel_background = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_bg, kernel_bg))
                
                if bg_method == 'blackhat':
                    # Black-hat: Loại vết đen trên nền sáng
                    # Phát hiện và loại bỏ các vết bẩn tối (nhiễu pepper)
                    logger.info("Áp dụng Black-hat để loại nền sáng có vết đen")
                    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_background)
                    # Trừ vết bẩn khỏi ảnh gốc
                    cleaned_gray = cv2.subtract(gray, blackhat)
                    # Threshold lại để đảm bảo chất lượng
                    _, final = cv2.threshold(cleaned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    self.intermediate_steps['5_bg_blackhat'] = blackhat
                    self.intermediate_steps['5_cleaned'] = cleaned_gray
                    self.intermediate_steps['6_final'] = final
                    
                elif bg_method == 'tophat':
                    # Top-hat: Loại nền tối, làm nổi bật text sáng
                    # Trích xuất các vùng sáng (text) từ nền tối
                    logger.info("Áp dụng Top-hat để loại nền tối")
                    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_background)
                    # Threshold để có ảnh nhị phân rõ ràng
                    _, final = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    self.intermediate_steps['5_bg_tophat'] = tophat
                    self.intermediate_steps['6_final'] = final
                    
                else:  # auto - Tự động chọn phương pháp phù hợp
                    # Phân tích độ sáng trung bình để quyết định
                    mean_val = np.mean(gray)
                    logger.info(f"Chế độ auto - Độ sáng trung bình: {mean_val:.2f}")
                    
                    if mean_val > 127:
                        # Nền sáng có vết đen → dùng black-hat
                        logger.info("Phát hiện nền sáng → áp dụng Black-hat")
                        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_background)
                        cleaned_gray = cv2.subtract(gray, blackhat)
                        _, final = cv2.threshold(cleaned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        self.intermediate_steps['5_bg_auto_blackhat'] = blackhat
                        self.intermediate_steps['5_cleaned'] = cleaned_gray
                    else:
                        # Nền tối → dùng top-hat
                        logger.info("Phát hiện nền tối → áp dụng Top-hat")
                        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_background)
                        _, final = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        self.intermediate_steps['5_bg_auto_tophat'] = tophat
                    self.intermediate_steps['6_final'] = final
            else:
                final = closed
                self.intermediate_steps['6_final'] = final
                logger.info("Bỏ qua bước loại nền")
            
            # BƯỚC 6: Tăng cường hiển thị và lưu kết quả
            # (CLAHE - optional, cho mục đích hiển thị tốt hơn)
            contrast_method = settings.get('contrastMethod', 'none')
            if contrast_method != 'none':
                logger.info(f"Áp dụng tăng cường độ tương phản: {contrast_method}")
                enhanced = self.enhance_contrast_adaptive(
                    original_gray,
                    method=contrast_method,
                    clip_limit=settings.get('claheClipLimit', 2.0)
                )
                self.intermediate_steps['7_enhanced'] = enhanced
            
            # Đánh giá kết quả so với ảnh gốc
            metrics = self.calculate_metrics(original_gray, final)
            logger.info(f"Đánh giá chất lượng - PSNR: {metrics['psnr']}, SSIM: {metrics['ssim']}, MSE: {metrics['mse']}")
            
            # Tạo summary về quá trình xử lý
            processing_summary = {
                'threshold_method': threshold_method,
                'opening_kernel': kernel_opening,
                'closing_kernel': kernel_closing,
                'background_method': bg_method,
                'background_kernel': kernel_bg if bg_method != 'none' else None,
                'contrast_enhancement': contrast_method,
                'total_steps': len(self.intermediate_steps)
            }
            
            logger.info("Hoàn thành xử lý ảnh")
            
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline_v2: {str(e)}")
            raise
    
    def remove_background_fixed(self, image, method='auto', kernel_size=15):
        """
        Fixed Background Removal (V2)
        
        Methods:
        - blackhat: Remove dark stains (subtract blackhat)
        - tophat: Brighten background
        - auto: Hybrid approach
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if method == 'blackhat':
            # Loại vết tối trên nền sáng
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            result = cv2.subtract(image, blackhat)
            result = np.clip(result + 10, 0, 255).astype(np.uint8)
            
        elif method == 'tophat':
            # Làm nổi text trên nền tối
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            result = cv2.add(image, tophat)
            
        else:  # auto
            # Kết hợp cả hai
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            result = cv2.subtract(image, blackhat)
            result = cv2.add(result, tophat * 0.5)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def enhance_contrast_adaptive(self, image, method='clahe_masked', clip_limit=2.0):
        """
        Contrast Enhancement
        
        Methods:
        - clahe: Global CLAHE
        - clahe_masked: CLAHE only on text regions
        - histogram_eq: Standard histogram equalization
        """
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        elif method == 'clahe_masked':
            # Detect text regions with edges
            edges = cv2.Canny(image, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(edges, kernel, iterations=2)
            
            # Apply CLAHE only to text regions
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            result = np.where(mask > 0, enhanced, image)
            return result.astype(np.uint8)
        
        elif method == 'histogram_eq':
            return cv2.equalizeHist(image)
        
        return image
    
    def apply_threshold(self, image, method='otsu'):
        """
        Threshold methods
        
        Methods:
        - otsu: Automatic threshold
        - adaptive_mean: Adaptive threshold with mean
        - adaptive_gaussian: Adaptive threshold with Gaussian
        """
        if method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        
        elif method == 'adaptive_mean':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        elif method == 'adaptive_gaussian':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        return image
    
    def calculate_metrics(self, original, processed):
        """Calculate quality metrics"""
        try:
            # Convert to grayscale if needed
            if len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            # Resize if shapes don't match
            if original.shape != processed.shape:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
            # Calculate metrics
            psnr_value = psnr(original, processed, data_range=255)
            ssim_value = ssim(original, processed, data_range=255)
            
            # Mean Squared Error
            mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
            
            return {
                'psnr': round(float(psnr_value), 2),
                'ssim': round(float(ssim_value), 4),
                'mse': round(float(mse), 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'psnr': 0, 'ssim': 0, 'mse': 0}
