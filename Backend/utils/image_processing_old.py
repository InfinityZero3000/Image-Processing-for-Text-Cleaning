"""
Image Processing Module - Pipeline V2.2 ENHANCED
Morphological operations for document cleaning - Tối ưu cho chữ viết tay
"""

import cv2
import numpy as np
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Xử lý ảnh với Pipeline V2.2 - Enhanced cho chữ viết tay đứt gãy"""
    
    def __init__(self):
        self.intermediate_steps = {}
    
    def process_pipeline_v2(self, image, settings):
        """
        Complete Pipeline V2.2 ENHANCED - Xử lý ảnh văn bản tăng cường
        
        TĂNG CƯỜNG CHO CHỮ VIẾT TAY:
        1. Tiền xử lý mạnh: Bilateral filter + Normalize
        2. Threshold thích ứng tốt hơn
        3. Làm liền nét chữ MẠNH với Dilation trước Closing
        4. Làm sạch nhiễu NHẸ
        5. Loại nền với kết hợp morphology
        """
        try:
            self.intermediate_steps = {}
            logger.info(f"=== PIPELINE V2.2 ENHANCED ===")
            logger.info(f"Settings: {settings}")
            
            # ========== BƯỚC 1: TIỀN XỬ LÝ MẠNH ==========
            # Chuyển sang grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray.copy()
            logger.info(f"Step 1: Grayscale - shape: {gray.shape}")
            
            # Normalize để chuẩn hóa độ sáng
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Bilateral filter - làm mịn nền, giữ cạnh chữ
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            self.intermediate_steps['1b_filtered'] = gray.copy()
            logger.info("Step 1b: Bilateral filter applied")
            
            # ========== BƯỚC 2: THRESHOLD THÍCH ỨNG ==========
            threshold_method = settings.get('thresholdMethod', 'adaptive_gaussian')
            
            # Dùng adaptive threshold để xử lý tốt hơn với ánh sáng không đều
            if threshold_method == 'adaptive_gaussian':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 8  # block_size=15, C=8
                )
            elif threshold_method == 'adaptive_mean':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, 15, 8
                )
            else:  # otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            self.intermediate_steps['2_threshold'] = binary.copy()
            logger.info(f"Step 2: Threshold ({threshold_method})")
            
            # ========== BƯỚC 3: LÀM LIỀN NÉT CHỮ MẠNH ==========
            kernel_closing = settings.get('kernelClosing', 3)
            
            # BƯỚC 3a: Dilation nhẹ trước để làm dày nét chữ
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
            self.intermediate_steps['3a_dilated'] = dilated.copy()
            logger.info("Step 3a: Dilation to thicken strokes")
            
            # BƯỚC 3b: Closing với kernel lớn hơn để nối các đoạn chữ
            # Kernel theo chiều ngang để nối chữ trong một dòng
            kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_closing + 1, 2))
            closed_h = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close_h)
            
            # Kernel theo chiều dọc để nối các nét dọc
            kernel_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, kernel_closing + 1))
            closed = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_close_v)
            
            # Closing toàn diện với kernel vuông
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing, kernel_closing))
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close)
            
            self.intermediate_steps['3_closing'] = closed.copy()
            logger.info(f"Step 3: Closing with kernel {kernel_closing}")
            
            # ========== BƯỚC 4: LÀM SẠCH NHIỄU NHẸ ==========
            kernel_opening = settings.get('kernelOpening', 2)
            
            # Opening với kernel nhỏ, chỉ loại nhiễu salt nhỏ
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_opening, kernel_opening))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            self.intermediate_steps['4_opening'] = opened.copy()
            logger.info(f"Step 4: Opening with kernel {kernel_opening}")
            
            # ========== BƯỚC 5: LOẠI BỎ NỀN VÀ VẾT BẨN ==========
            bg_method = settings.get('backgroundRemoval', 'auto')
            kernel_bg = settings.get('backgroundKernel', 15)
            
            if bg_method != 'none':
                final = self._remove_background_enhanced(
                    original_gray, opened, bg_method, kernel_bg, kernel_close
                )
            else:
                final = opened
            
            self.intermediate_steps['5_bg_removed'] = final.copy()
            logger.info(f"Step 5: Background removal ({bg_method})")
            
            # ========== BƯỚC 6: HẬU XỬ LÝ - ĐẢM BẢO CHỮ RÕ RÀNG ==========
            # Closing cuối cùng để đảm bảo chữ liền mạch
            kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final)
            
            # Loại bỏ các thành phần nhỏ (nhiễu còn sót)
            final = self._remove_small_components(final, min_size=50)
            
            self.intermediate_steps['6_final'] = final.copy()
            logger.info("Step 6: Final cleanup done")
            
            # ========== BƯỚC 7: TĂNG CƯỜNG HIỂN THỊ (Optional) ==========
            contrast_method = settings.get('contrastMethod', 'none')
            if contrast_method != 'none':
                enhanced = self.enhance_contrast_adaptive(
                    original_gray,
                    method=contrast_method,
                    clip_limit=settings.get('claheClipLimit', 2.0)
                )
                self.intermediate_steps['7_enhanced'] = enhanced
            
            # ========== ĐÁNH GIÁ KẾT QUẢ ==========
            metrics = self.calculate_metrics(original_gray, final)
            logger.info(f"Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}")
            
            processing_summary = {
                'threshold_method': threshold_method,
                'opening_kernel': kernel_opening,
                'closing_kernel': kernel_closing,
                'background_method': bg_method,
                'background_kernel': kernel_bg,
                'contrast_enhancement': contrast_method,
                'total_steps': len(self.intermediate_steps)
            }
            
            logger.info("=== PIPELINE COMPLETE ===")
            
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise
    
    def _remove_background_enhanced(self, gray, binary, method, kernel_size, kernel_close):
        """Loại nền tăng cường với kết hợp nhiều phương pháp"""
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        mean_val = np.mean(gray)
        logger.info(f"Mean brightness: {mean_val:.2f}")
        
        if method == 'blackhat' or (method == 'auto' and mean_val > 127):
            # Nền sáng có vết đen → dùng black-hat
            logger.info("Applying Black-hat for light background")
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bg)
            cleaned = cv2.subtract(gray, blackhat)
            _, result = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'tophat' or (method == 'auto' and mean_val <= 127):
            # Nền tối → dùng top-hat
            logger.info("Applying Top-hat for dark background")
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_bg)
            _, result = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            result = binary
        
        # Kết hợp với binary đã xử lý trước đó
        result = cv2.bitwise_and(result, binary)
        
        # Closing lại để nối các đoạn bị mất
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)
        
        return result
    
    def _remove_small_components(self, binary, min_size=50):
        """Loại bỏ các thành phần nhỏ (nhiễu)"""
        # Đảo ngược để tìm các vùng đen (chữ)
        inverted = cv2.bitwise_not(binary)
        
        # Tìm các connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        # Tạo mask cho các component đủ lớn
        mask = np.zeros(binary.shape, dtype=np.uint8)
        for i in range(1, num_labels):  # Bỏ qua background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        # Đảo ngược lại
        result = cv2.bitwise_not(mask)
        
        return result
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing, kernel_closing))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            self.intermediate_steps['3_closing'] = closed
            logger.info(f"Áp dụng Closing với kernel {kernel_closing}x{kernel_closing} để làm liền chữ")
            
            # BƯỚC 4: Làm sạch nhiễu - Opening (loại bỏ điểm trắng nhỏ, nhiễu salt)
            # *** Thực hiện SAU Closing với kernel rất nhỏ để chỉ loại nhiễu, không phá vỡ chữ ***
            kernel_opening = settings.get('kernelOpening', 2)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_opening, kernel_opening))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            self.intermediate_steps['4_opening'] = opened
            logger.info(f"Áp dụng Opening với kernel {kernel_opening}x{kernel_opening} để làm sạch nhiễu")
            
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
                    # Kết hợp với kết quả từ opening để giữ lại chữ rõ ràng
                    _, final = cv2.threshold(cleaned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # Làm liền chữ một lần nữa sau khi loại nền
                    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_close)
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
                    # Làm liền chữ một lần nữa sau khi loại nền
                    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_close)
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
                        # Làm liền chữ một lần nữa
                        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_close)
                        self.intermediate_steps['5_bg_auto_blackhat'] = blackhat
                        self.intermediate_steps['5_cleaned'] = cleaned_gray
                    else:
                        # Nền tối → dùng top-hat
                        logger.info("Phát hiện nền tối → áp dụng Top-hat")
                        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_background)
                        _, final = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        # Làm liền chữ một lần nữa
                        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_close)
                        self.intermediate_steps['5_bg_auto_tophat'] = tophat
                    self.intermediate_steps['6_final'] = final
            else:
                final = opened
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
