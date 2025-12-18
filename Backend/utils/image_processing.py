"""
Image Processing Module - Pipeline V4.0 PREMIUM
Tích hợp các kỹ thuật tiên tiến từ các công cụ xử lý ảnh hàng đầu:

1. Auto Border/Artifact Removal - Loại bỏ viền đen, vệt bẩn
2. Document Deskewing - Căn chỉnh tài liệu nghiêng
3. Shadow Removal - Loại bỏ bóng
4. Adaptive Background Estimation - Ước lượng nền thông minh
5. Ink Bleeding Removal - Loại hiệu ứng ink bleed
6. Smart Text Preservation - Bảo toàn chữ tốt hơn
7. Multi-scale Processing - Xử lý đa tỷ lệ
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.filters import threshold_sauvola, threshold_niblack
from scipy import ndimage
from scipy.ndimage import median_filter, uniform_filter
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Xử lý ảnh với Pipeline V4.0 PREMIUM - Tích hợp công nghệ tiên tiến"""
    
    def __init__(self):
        self.intermediate_steps = {}

    # ==================== PIPELINE PREMIUM (MỚI) ====================
    def process_pipeline_premium(self, image, settings):
        """
        Pipeline PREMIUM - Theo yêu cầu task
        
        Các bước xử lý:
        1. Tiền xử lý: Chuyển sang thang xám
        2. Nhị phân hóa: Dùng ngưỡng (Otsu hoặc Adaptive) để nhị phân ảnh
        3. Làm sạch nhiễu: Sử dụng phép mở (Opening) để loại bỏ các điểm trắng nhỏ
        4. Làm liền nét chữ: Dùng phép đóng (Closing) để lấp khoảng trống
        5. Loại bỏ nền và vết bẩn: Dùng Black-hat hoặc Top-hat
        6. Tăng cường hiển thị: CLAHE (optional)
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== PIPELINE PREMIUM (Task Requirements) ===")
            logger.info(f"Settings: {settings}")

            # ===== BƯỚC 1: TIỀN XỬ LÝ - CHUYỂN SANG THANG XÁM =====
            gray = self._ensure_gray_uint8(image)
            original_gray = gray.copy()
            h, w = gray.shape
            self.intermediate_steps['1_grayscale'] = gray.copy()
            logger.info(f"Input: {w}x{h}")
            logger.info("Step 1: Grayscale conversion")

            # ===== BƯỚC 2: NHỊ PHÂN HÓA =====
            # Dùng ngưỡng (Otsu hoặc Adaptive threshold) để nhị phân ảnh
            threshold_method = settings.get('thresholdMethod', 'otsu')
            
            if threshold_method == 'otsu':
                # Gaussian blur để giảm nhiễu trước khi threshold
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                logger.info("Step 2: Otsu thresholding")
            elif threshold_method in ('adaptive', 'adaptive_mean'):
                block_size = settings.get('adaptiveBlock', 31)
                if block_size % 2 == 0:
                    block_size += 1
                c_value = settings.get('adaptiveC', 10)
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, block_size, c_value
                )
                logger.info(f"Step 2: Adaptive thresholding (block={block_size}, C={c_value})")
            elif threshold_method == 'adaptive_gaussian':
                block_size = settings.get('adaptiveBlock', 31)
                if block_size % 2 == 0:
                    block_size += 1
                c_value = settings.get('adaptiveC', 10)
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, block_size, c_value
                )
                logger.info(f"Step 2: Adaptive Gaussian thresholding (block={block_size}, C={c_value})")
            else:
                # Default to Otsu
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                logger.info("Step 2: Otsu thresholding (default)")
            
            # Auto-invert nếu nền đen chữ trắng
            white_ratio = (binary > 0).mean()
            if white_ratio > 0.85:
                binary = cv2.bitwise_not(binary)
                logger.info("Step 2b: Auto-inverted (background was black)")
            
            self.intermediate_steps['2_binary'] = binary.copy()

            # ===== BƯỚC 3: LÀM SẠCH NHIỄU - OPENING =====
            # Sử dụng phép mở (opening) để loại bỏ các điểm trắng nhỏ (nhiễu)
            opening_k = int(settings.get('openingKernel', 2))
            if opening_k > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_k, opening_k))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                logger.info(f"Step 3: Opening (kernel={opening_k}x{opening_k})")
            else:
                logger.info("Step 3: Opening skipped (kernel=1)")
            
            self.intermediate_steps['3_opening'] = binary.copy()

            # ===== BƯỚC 4: LÀM LIỀN NÉT CHỮ - CLOSING =====
            # Dùng phép đóng (closing) để lấp khoảng trống, nối các đoạn chữ đứt gãy
            closing_k = int(settings.get('closingKernel', 2))
            if closing_k > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_k, closing_k))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                logger.info(f"Step 4: Closing (kernel={closing_k}x{closing_k})")
            else:
                logger.info("Step 4: Closing skipped (kernel=1)")
            
            self.intermediate_steps['4_closing'] = binary.copy()

            # ===== BƯỚC 5: LOẠI BỎ NỀN VÀ VẾT BẨN =====
            # Dùng Black-hat hoặc Top-hat tùy loại nền
            bg_mode = settings.get('bgMode', 'auto')
            bg_kernel = int(settings.get('bgKernel', 25))
            
            if bg_mode == 'auto':
                # Auto detect: nền sáng (mean > 127) → blackhat, nền tối → tophat
                bg_mode = 'blackhat' if float(gray.mean()) > 127.0 else 'tophat'
                logger.info(f"Step 5: Auto-detected background mode: {bg_mode}")
            
            if bg_mode == 'blackhat':
                # Loại bỏ các vùng tối hơn nền (dấu bẩn, bóng)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bg_kernel, bg_kernel))
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
                gray_enhanced = cv2.subtract(gray, blackhat)
                gray_enhanced = cv2.normalize(gray_enhanced, None, 0, 255, cv2.NORM_MINMAX)
                self.intermediate_steps['5_blackhat'] = blackhat.copy()
                self.intermediate_steps['5b_enhanced'] = gray_enhanced.copy()
                logger.info(f"Step 5: Black-hat applied (kernel={bg_kernel}x{bg_kernel})")
            elif bg_mode == 'tophat':
                # Tăng cường các vùng sáng hơn nền (chữ sáng)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bg_kernel, bg_kernel))
                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
                gray_enhanced = cv2.add(gray, tophat)
                gray_enhanced = cv2.normalize(gray_enhanced, None, 0, 255, cv2.NORM_MINMAX)
                self.intermediate_steps['5_tophat'] = tophat.copy()
                self.intermediate_steps['5b_enhanced'] = gray_enhanced.copy()
                logger.info(f"Step 5: Top-hat applied (kernel={bg_kernel}x{bg_kernel})")
            else:
                # No background removal
                gray_enhanced = gray.copy()
                logger.info("Step 5: Background removal skipped")

            # ===== BƯỚC 6: TĂNG CƯỜNG HIỂN THỊ (OPTIONAL) =====
            # Áp dụng CLAHE để tăng độ tương phản
            contrast_method = settings.get('contrastMethod', 'clahe')
            
            if contrast_method == 'clahe':
                clahe_clip = float(settings.get('claheClip', 2.0))
                clahe_grid = int(settings.get('claheTileGrid', 8))
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
                gray_enhanced = clahe.apply(gray_enhanced)
                logger.info(f"Step 6: CLAHE applied (clip={clahe_clip}, grid={clahe_grid}x{clahe_grid})")
            elif contrast_method == 'histogram':
                gray_enhanced = cv2.equalizeHist(gray_enhanced)
                logger.info("Step 6: Histogram equalization applied")
            else:
                logger.info("Step 6: Contrast enhancement skipped")
            
            self.intermediate_steps['6_enhanced'] = gray_enhanced.copy()

            # ===== FINAL: Kết hợp binary với enhanced grayscale =====
            # Sử dụng binary đã xử lý làm kết quả cuối
            final = binary.copy()
            self.intermediate_steps['7_final'] = final.copy()
            logger.info("Final: Pipeline complete")

            # ===== METRICS =====
            metrics = self.calculate_metrics(original_gray, final)
            logger.info(f"Metrics: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
            logger.info("=== PIPELINE PREMIUM COMPLETE ===")

            processing_summary = {
                'pipeline_version': 'PREMIUM_TASK_REQUIREMENTS',
                'steps_executed': 7,
                'threshold_method': threshold_method,
                'opening_kernel': opening_k,
                'closing_kernel': closing_k,
                'background_mode': bg_mode,
                'bg_kernel': bg_kernel,
                'contrast_method': contrast_method,
            }

            logger.info("=== PIPELINE PREMIUM COMPLETE ===")
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary,
            }

        except Exception as e:
            logger.error(f"Error in premium pipeline: {str(e)}")
            raise

    def _remove_border_artifacts(self, gray, settings):
        """
        NÂNG CẤP: Loại bỏ viền đen, vệt bẩn lớn, artifact scan
        Kỹ thuật mới:
        - Phân tích gradient để phát hiện ranh giới
        - Flood fill từ cạnh
        - Phân tích connected components thông minh
        """
        h, w = gray.shape
        border_threshold = int(settings.get('borderThreshold', 50))
        border_margin = float(settings.get('borderMargin', 0.15))  # 15% mỗi cạnh
        aggressive = settings.get('borderAggressive', True)  # Mode aggressive mới
        
        # Tạo mask cho các vùng cạnh
        margin_h = int(h * border_margin)
        margin_w = int(w * border_margin)
        
        # ========= PHASE 1: Edge-based detection =========
        border_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # Phân tích từng cạnh với gradient
        def analyze_edge_strip(strip, axis):
            """Phân tích dải pixel ở cạnh để tìm vùng tối bất thường"""
            # Tính mean và std của strip
            mean_val = np.mean(strip)
            std_val = np.std(strip)
            # Vùng tối bất thường: mean thấp và std thấp (đồng nhất tối)
            is_dark = mean_val < border_threshold * 1.5
            is_uniform = std_val < 30
            return is_dark and is_uniform
        
        # Kiểm tra cạnh trái - scanning từ trái qua
        left_boundary = 0
        for col in range(min(margin_w, w // 3)):  # Không quá 1/3 ảnh
            strip = gray[:, col]
            if analyze_edge_strip(strip, 0):
                left_boundary = col + 1
            else:
                break
        if left_boundary > 5:  # Có vùng tối đáng kể
            border_mask[:, :left_boundary] = 255
        
        # Kiểm tra cạnh phải
        right_boundary = w
        for col in range(w - 1, max(w - margin_w, w * 2 // 3), -1):
            strip = gray[:, col]
            if analyze_edge_strip(strip, 0):
                right_boundary = col
            else:
                break
        if right_boundary < w - 5:
            border_mask[:, right_boundary:] = 255
        
        # Kiểm tra cạnh trên
        top_boundary = 0
        for row in range(min(margin_h, h // 3)):
            strip = gray[row, :]
            if analyze_edge_strip(strip, 1):
                top_boundary = row + 1
            else:
                break
        if top_boundary > 5:
            border_mask[:top_boundary, :] = 255
        
        # Kiểm tra cạnh dưới
        bottom_boundary = h
        for row in range(h - 1, max(h - margin_h, h * 2 // 3), -1):
            strip = gray[row, :]
            if analyze_edge_strip(strip, 1):
                bottom_boundary = row
            else:
                break
        if bottom_boundary < h - 5:
            border_mask[bottom_boundary:, :] = 255
        
        # ========= PHASE 2: Flood fill từ góc =========
        # Các vệt đen thường connect với góc/cạnh
        flood_seed_points = [
            (0, 0), (w-1, 0), (0, h-1), (w-1, h-1),  # 4 góc
            (w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)  # Giữa các cạnh
        ]
        
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        temp_gray = gray.copy()
        
        for seed in flood_seed_points:
            if temp_gray[seed[1], seed[0]] < border_threshold:
                # FloodFill với tolerance
                cv2.floodFill(
                    temp_gray, flood_mask, seed, 255,
                    loDiff=(border_threshold // 2,), 
                    upDiff=(border_threshold // 2,),
                    flags=cv2.FLOODFILL_MASK_ONLY
                )
        
        # Extract flood mask và thêm vào border_mask
        flood_result = flood_mask[1:-1, 1:-1] * 255
        border_mask = cv2.bitwise_or(border_mask, flood_result)
        
        # ========= PHASE 3: Connected Components Analysis =========
        if aggressive:
            # Threshold tìm tất cả vùng tối
            _, binary_dark = cv2.threshold(gray, border_threshold, 255, cv2.THRESH_BINARY_INV)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dark, connectivity=8)
            
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                cx, cy = centroids[i]
                
                # Kiểm tra các điều kiện để xác định artifact
                touches_edge = (x == 0 or x + bw >= w - 1 or y == 0 or y + bh >= h - 1)
                is_large = area > (h * w * 0.005)  # > 0.5% diện tích ảnh
                is_tall = bh > h * 0.25  # Cao hơn 25% ảnh
                is_wide = bw > w * 0.25  # Rộng hơn 25% ảnh
                is_elongated = max(bw, bh) / max(1, min(bw, bh)) > 5  # Dài/hẹp
                
                # Tính density - artifact thường là vùng tối đặc
                component_pixels = gray[labels == i]
                density = np.mean(component_pixels < border_threshold)
                is_dense = density > 0.7  # 70% pixel trong component là tối
                
                # Artifact: chạm cạnh + (lớn hoặc dài) + đặc
                if touches_edge and is_dense:
                    if is_large or is_tall or is_wide or is_elongated:
                        border_mask[labels == i] = 255
        
        # ========= PHASE 4: Morphological cleanup =========
        # Dilate để cover toàn bộ artifact
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        border_mask = cv2.dilate(border_mask, kernel, iterations=3)
        
        # Close để fill gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # ========= PHASE 5: Inpainting thay vì chỉ fill trắng =========
        result = gray.copy()
        
        # Dùng inpainting cho kết quả tự nhiên hơn
        if np.sum(border_mask) > 0:
            # Nếu vùng artifact quá lớn (> 20%), chỉ fill trắng
            artifact_ratio = np.sum(border_mask > 0) / (h * w)
            if artifact_ratio < 0.2:
                # Inpainting cho kết quả mượt
                result = cv2.inpaint(gray, border_mask, 5, cv2.INPAINT_TELEA)
            else:
                # Fill trắng cho vùng lớn
                result[border_mask > 0] = 255
        
        logger.info(f"Border removal: removed {np.sum(border_mask > 0) / (h * w) * 100:.1f}% of image")
        
        return result, border_mask

    def _correct_illumination(self, gray, method='adaptive', settings=None):
        """Cân bằng ánh sáng không đều"""
        if settings is None:
            settings = {}
        
        if method == 'divide':
            # Phương pháp chia cho background blur
            blur_size = int(settings.get('illuminationBlur', 51))
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            background = np.clip(background, 1, 255).astype(np.float32)
            normalized = (gray.astype(np.float32) / background) * 128
            return np.clip(normalized, 0, 255).astype(np.uint8)
        
        elif method == 'subtract':
            # Phương pháp trừ background
            blur_size = int(settings.get('illuminationBlur', 51))
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            diff = cv2.subtract(background, gray)
            return self._minmax_to_uint8(diff)
        
        elif method == 'adaptive' or method == 'multiscale':
            # Xử lý đa tỷ lệ để bắt cả shadow lớn và nhỏ
            scales = [15, 31, 61]
            normalized = np.zeros_like(gray, dtype=np.float32)
            
            for blur_size in scales:
                bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
                bg = np.clip(bg, 1, 255).astype(np.float32)
                norm = (gray.astype(np.float32) / bg) * 128
                normalized += norm
            
            normalized /= len(scales)
            return np.clip(normalized, 0, 255).astype(np.uint8)
        
        elif method == 'morphological':
            # Dùng opening lớn để ước lượng nền
            kernel_size = int(settings.get('morphKernel', 51))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            background = cv2.GaussianBlur(background, (31, 31), 0)
            background = np.clip(background, 1, 255).astype(np.float32)
            normalized = (gray.astype(np.float32) / background) * 128
            return np.clip(normalized, 0, 255).astype(np.uint8)
        
        return gray

    def _estimate_background_multiscale(self, gray, settings):
        """Ước lượng nền đa tỷ lệ - chính xác hơn"""
        # Sử dụng nhiều kernel size để bắt được nền ở các mức khác nhau
        small_kernel = int(settings.get('bgKernelSmall', 15))
        medium_kernel = int(settings.get('bgKernelMedium', 31))
        large_kernel = int(settings.get('bgKernelLarge', 61))
        
        small_kernel = small_kernel if small_kernel % 2 == 1 else small_kernel + 1
        medium_kernel = medium_kernel if medium_kernel % 2 == 1 else medium_kernel + 1
        large_kernel = large_kernel if large_kernel % 2 == 1 else large_kernel + 1
        
        # Ước lượng nền ở từng scale
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_kernel, small_kernel))
        k_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (medium_kernel, medium_kernel))
        k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_kernel, large_kernel))
        
        bg_small = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_small)
        bg_medium = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_medium)
        bg_large = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_large)
        
        # Kết hợp: ưu tiên large để loại chi tiết text, nhưng giữ thông tin cục bộ
        background = ((bg_small.astype(np.float32) * 0.2 + 
                      bg_medium.astype(np.float32) * 0.3 + 
                      bg_large.astype(np.float32) * 0.5)).astype(np.uint8)
        
        background = cv2.GaussianBlur(background, (31, 31), 0)
        
        # FIX: Normalize thay vì subtract để giữ đúng hướng (nền trắng, chữ đen)
        # Chia gray cho background để normalize ánh sáng
        bg_float = np.clip(background.astype(np.float32), 1, 255)
        enhanced = (gray.astype(np.float32) / bg_float) * 255
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced, background

    def _enhance_contrast_premium(self, gray, method='clahe_adaptive', settings=None):
        """Tăng cường độ tương phản cao cấp"""
        if settings is None:
            settings = {}
        
        if method == 'clahe':
            clip = float(settings.get('claheClipLimit', 3.0))
            grid = int(settings.get('claheGrid', 8))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            return clahe.apply(gray)
        
        elif method == 'clahe_adaptive':
            # CLAHE với clip limit thích ứng theo vùng
            clip = float(settings.get('claheClipLimit', 3.0))
            
            # Tính độ tương phản cục bộ
            local_std = ndimage.generic_filter(gray.astype(np.float32), np.std, size=15)
            mean_std = np.mean(local_std)
            
            # Điều chỉnh clip limit dựa trên độ tương phản
            if mean_std < 20:  # Ảnh có độ tương phản thấp
                clip = min(clip * 1.5, 6.0)
            
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            return clahe.apply(gray)
        
        elif method == 'unsharp':
            # Unsharp masking để làm nét text
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        elif method == 'gamma':
            # Gamma correction
            gamma = float(settings.get('gamma', 1.2))
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(256)]).astype(np.uint8)
            return cv2.LUT(gray, table)
        
        return gray

    def _detect_text_regions(self, gray, method='gradient', settings=None):
        """Phát hiện vùng chứa text để xử lý riêng"""
        if settings is None:
            settings = {}
        
        if method == 'gradient':
            # Sử dụng gradient để phát hiện vùng có cạnh (text)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            gradient = self._minmax_to_uint8(gradient)
            
            # Threshold và dilate để tạo vùng text
            _, text_mask = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
            text_mask = cv2.dilate(text_mask, kernel, iterations=2)
            
            return text_mask.astype(np.uint8)
        
        elif method == 'mser':
            # MSER (Maximally Stable Extremal Regions) cho text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            mask = np.zeros_like(gray)
            for region in regions:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                cv2.fillPoly(mask, [hull], 255)
            
            return mask
        
        elif method == 'edge':
            # Canny edge + dilation
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            text_mask = cv2.dilate(edges, kernel, iterations=3)
            return text_mask
        
        return np.ones_like(gray) * 255

    def _binarize_premium(self, gray, method='sauvola_adaptive', text_mask=None, settings=None):
        """Nhị phân hóa cao cấp với nhiều tùy chọn"""
        if settings is None:
            settings = {}
        
        if method == 'sauvola':
            window = int(settings.get('sauvolaWindow', 31))
            window = max(3, window)
            window = window if window % 2 == 1 else window + 1
            k = float(settings.get('sauvolaK', 0.2))
            thresh = threshold_sauvola(gray, window_size=window, k=k)
            binary = (gray > thresh).astype(np.uint8) * 255
            
        elif method == 'sauvola_adaptive':
            # Sauvola với window size thích ứng theo độ phức tạp cục bộ
            h, w = gray.shape
            
            # Chia ảnh thành blocks và tính window size tối ưu cho từng block
            block_size = 128
            binary = np.zeros_like(gray)
            
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    block = gray[y:y_end, x:x_end]
                    
                    # Tính độ phức tạp của block
                    block_std = np.std(block)
                    
                    # Window size nhỏ hơn cho vùng có nhiều chi tiết
                    if block_std > 40:
                        window = 15
                    elif block_std > 25:
                        window = 25
                    else:
                        window = 41
                    
                    k = float(settings.get('sauvolaK', 0.2))
                    thresh = threshold_sauvola(block, window_size=window, k=k)
                    binary[y:y_end, x:x_end] = (block > thresh).astype(np.uint8) * 255
            
        elif method == 'niblack':
            window = int(settings.get('niblackWindow', 31))
            window = max(3, window)
            window = window if window % 2 == 1 else window + 1
            k = float(settings.get('niblackK', -0.2))
            thresh = threshold_niblack(gray, window_size=window, k=k)
            binary = (gray > thresh).astype(np.uint8) * 255
            
        elif method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'adaptive_gaussian':
            block = int(settings.get('adaptiveBlock', 31))
            block = max(3, block)
            block = block if block % 2 == 1 else block + 1
            c = int(settings.get('adaptiveC', 10))
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block, c)
        
        elif method == 'hybrid':
            # Kết hợp Otsu global với Sauvola local
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            sauvola_thresh = threshold_sauvola(gray, window_size=31, k=0.2)
            sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255
            
            # Kết hợp: dùng Sauvola cho vùng text, Otsu cho vùng khác
            if text_mask is not None:
                binary = np.where(text_mask > 0, sauvola, otsu)
            else:
                binary = cv2.bitwise_and(otsu, sauvola)
        
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return binary

    def _remove_noise_intelligent(self, binary, method='adaptive', text_mask=None, settings=None):
        """Loại nhiễu thông minh - bảo toàn nét chữ"""
        if settings is None:
            settings = {}
        
        if method == 'adaptive':
            # Opening nhỏ để loại salt noise
            k_open = int(settings.get('openingKernel', 2))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            
            # Trong vùng text: opening rất nhẹ để không phá nét
            # Ngoài vùng text: opening mạnh hơn
            if text_mask is not None:
                text_region = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
                
                non_text_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                non_text_region = cv2.morphologyEx(binary, cv2.MORPH_OPEN, non_text_kernel, iterations=2)
                
                result = np.where(text_mask > 0, text_region, non_text_region)
            else:
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            
            return result.astype(np.uint8)
        
        elif method == 'median':
            # Median filter - tốt cho salt-pepper noise
            kernel_size = int(settings.get('medianKernel', 3))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            return cv2.medianBlur(binary, kernel_size)
        
        elif method == 'morphological':
            # Opening rồi closing
            k_size = int(settings.get('morphKernel', 2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        
        return binary

    def _enhance_strokes(self, binary, method='adaptive', settings=None):
        """Tăng cường và làm liền nét chữ"""
        if settings is None:
            settings = {}
        
        if method == 'none':
            return binary
        
        elif method == 'light':
            # Closing nhẹ
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        elif method == 'adaptive':
            # Closing thích ứng: mạnh hơn theo chiều ngang (cho chữ viết)
            k_h = int(settings.get('strokeKernelH', 3))
            k_v = int(settings.get('strokeKernelV', 2))
            
            # Closing theo chiều ngang trước
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_h, 1))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
            
            # Closing theo chiều dọc
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_v))
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_v)
            
            # Closing ellipse nhỏ để làm mịn
            kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_e)
            
            return closed
        
        elif method == 'strong':
            # Closing mạnh cho chữ viết tay đứt nhiều
            k_close = int(settings.get('closingKernel', 4))
            
            # Dilation nhẹ trước
            kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dilated = cv2.dilate(binary, kernel_d, iterations=1)
            
            # Closing mạnh
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            # Erosion nhẹ để về lại kích thước gần ban đầu
            eroded = cv2.erode(closed, kernel_d, iterations=1)
            
            return eroded
        
        return binary

    def _final_polish(self, binary, settings):
        """Hoàn thiện cuối cùng - đảm bảo nền TRẮNG, chữ ĐEN"""
        # Đảm bảo binary hoàn toàn
        final = (binary > 127).astype(np.uint8) * 255
        
        # Closing nhẹ cuối cùng
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)
        
        # AUTO-INVERT: Đảm bảo nền trắng, chữ đen
        # Tài liệu thường có nền sáng (>50%) và chữ tối
        white_ratio = np.mean(final == 255)
        if white_ratio < 0.3:  # Nếu ít hơn 30% là trắng → bị invert
            final = cv2.bitwise_not(final)
            logger.info("Final polish: Auto-inverted to ensure white background")
        
        return final
    
    def process_pipeline_v3(self, image, settings):
        """
        Pipeline V3.0 - Kết hợp GaussianDiff + DivideConquer
        Tối ưu cho nhận diện chữ viết tay cổ
        
        Các bước:
        1. Tiền xử lý: Denoise
        2. Gaussian Difference: Loại nền
        3. Divide & Conquer: Local thresholding
        4. Morphological cleanup
        5. Loại nhiễu nhỏ
        """
        try:
            self.intermediate_steps = {}
            logger.info(f"=== PIPELINE V3.0 - GaussianDiff + DivideConquer ===")
            logger.info(f"Settings: {settings}")
            
            # ========== BƯỚC 1: TIỀN XỬ LÝ ==========
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray.copy()
            logger.info(f"Step 1: Grayscale - shape: {gray.shape}")
            
            # Median filter để loại nhiễu salt-pepper
            denoised = cv2.medianBlur(gray, 3)
            # Bilateral filter - giữ cạnh
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            self.intermediate_steps['1b_denoised'] = denoised.copy()
            logger.info("Step 1b: Denoised (Median + Bilateral)")
            
            # ========== BƯỚC 2: GAUSSIAN DIFFERENCE (Loại nền) ==========
            logger.info("Step 2: Gaussian Difference - Background removal")
            
            # Gaussian blur với sigma nhỏ (giữ chi tiết)
            small_blur = cv2.GaussianBlur(denoised, (3, 3), 1)
            
            # Gaussian blur với sigma lớn (ước lượng background)
            large_blur = cv2.GaussianBlur(denoised, (51, 51), 20)
            
            # Trừ để lấy chi tiết (text)
            diff = cv2.subtract(large_blur, small_blur)
            diff = self._minmax_to_uint8(diff)
            
            self.intermediate_steps['2_gaussian_diff'] = diff.copy()
            logger.info("Step 2: Gaussian Difference applied")
            
            # ========== BƯỚC 3: CLAHE ENHANCEMENT ==========
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(diff.astype(np.uint8))
            self.intermediate_steps['3_clahe'] = enhanced.copy()
            logger.info("Step 3: CLAHE enhancement")
            
            # ========== BƯỚC 4: DIVIDE & CONQUER THRESHOLDING ==========
            block_size = settings.get('blockSize', 64)
            binary = self._divide_conquer_threshold(enhanced, block_size)
            self.intermediate_steps['4_binary'] = binary.copy()
            logger.info(f"Step 4: Divide & Conquer threshold (block={block_size})")
            
            # ========== BƯỚC 5: MORPHOLOGICAL CLEANUP ==========
            kernel_closing = settings.get('kernelClosing', 3)
            kernel_opening = settings.get('kernelOpening', 2)
            
            # Closing để nối nét đứt
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing, kernel_closing))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            
            # Opening nhẹ để loại nhiễu
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_opening, kernel_opening))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            self.intermediate_steps['5_morphology'] = opened.copy()
            logger.info(f"Step 5: Morphology (close={kernel_closing}, open={kernel_opening})")
            
            # ========== BƯỚC 6: LOẠI NHIỄU NHỎ ==========
            min_noise_size = settings.get('minNoiseSize', 40)
            final = self._remove_small_components(opened, min_size=min_noise_size)
            
            # Final closing
            kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final)
            
            self.intermediate_steps['6_final'] = final.copy()
            logger.info(f"Step 6: Noise removal (min_size={min_noise_size})")
            
            # ========== ĐÁNH GIÁ ==========
            metrics = self.calculate_metrics(original_gray, final)
            logger.info(f"Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}")
            
            processing_summary = {
                'pipeline_version': 'V3.0',
                'method': 'GaussianDiff + DivideConquer',
                'block_size': block_size,
                'closing_kernel': kernel_closing,
                'opening_kernel': kernel_opening,
                'min_noise_size': min_noise_size,
                'total_steps': len(self.intermediate_steps)
            }
            
            logger.info("=== PIPELINE V3.0 COMPLETE ===")
            
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary
            }
            
        except Exception as e:
            logger.error(f"Error in pipeline V3: {str(e)}")
            raise

    def process_pipeline_robust(self, image, settings):
        """Robust document text cleaning pipeline.

        Mục tiêu: làm sạch văn bản (đặc biệt tài liệu scan/chụp), khử nền,
        giảm nhiễu, và (tuỳ chọn) loại bỏ đường kẻ bảng.

        Không ép buộc bước nối nét (closing) — chỉ chạy khi bật `enableStrokeRepair`.
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== PIPELINE ROBUST - Document Text Cleaning ===")
            logger.info(f"Settings: {settings}")

            gray = self._ensure_gray_uint8(image)
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray.copy()

            # Denoise nhẹ, giữ biên chữ
            denoise_strength = int(settings.get('denoiseStrength', 10))
            denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)
            denoised = cv2.bilateralFilter(denoised, 7, 60, 60)
            self.intermediate_steps['1b_denoised'] = denoised.copy()

            # Normalize illumination / background
            bg_blur = int(settings.get('backgroundBlur', 35))
            bg_blur = bg_blur if bg_blur % 2 == 1 else bg_blur + 1
            normalized = self._normalize_illumination_divide(denoised, blur_ksize=bg_blur)
            self.intermediate_steps['2_normalized'] = normalized.copy()

            # Morphological background cleanup (auto top-hat / black-hat)
            bg_kernel = int(settings.get('backgroundKernel', 25))
            bg_kernel = max(3, bg_kernel)
            bg_kernel = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
            bg_method = settings.get('backgroundRemoval', 'auto')
            bg_removed = self._apply_hat_background_removal(normalized, method=bg_method, kernel_size=bg_kernel)
            self.intermediate_steps['3_bg_removed'] = bg_removed.copy()

            # Contrast enhancement (CLAHE)
            clahe_clip = float(settings.get('claheClipLimit', 2.5))
            clahe_grid = int(settings.get('claheGrid', 8))
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
            enhanced = clahe.apply(bg_removed)
            self.intermediate_steps['4_clahe'] = enhanced.copy()

            # Binarization
            threshold_method = settings.get('thresholdMethod', 'sauvola')
            binary = self._binarize(enhanced, method=threshold_method, settings=settings)
            self.intermediate_steps['5_binary'] = binary.copy()

            # Optional: remove table/grid lines
            remove_lines = bool(settings.get('removeLines', True))
            if remove_lines:
                line_scale = float(settings.get('lineScale', 40.0))
                binary_wo_lines, line_mask = self._remove_table_lines(binary, scale=line_scale)
                self.intermediate_steps['6_line_mask'] = line_mask.copy()
                self.intermediate_steps['6_no_lines'] = binary_wo_lines.copy()
                working = binary_wo_lines
            else:
                working = binary

            # Cleanup small specks
            kernel_open = int(settings.get('kernelOpening', 2))
            kernel_open = max(1, kernel_open)
            kernel_open = kernel_open if kernel_open % 2 == 1 else kernel_open + 1
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
            opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, open_kernel)
            self.intermediate_steps['7_opening'] = opened.copy()

            min_area = int(settings.get('minComponentArea', 40))
            cleaned = self._remove_small_components(opened, min_size=min_area)
            self.intermediate_steps['8_components'] = cleaned.copy()

            # Optional stroke repair (NOT forced)
            enable_stroke_repair = bool(settings.get('enableStrokeRepair', False))
            if enable_stroke_repair:
                kernel_close = int(settings.get('kernelClosing', 3))
                kernel_close = max(1, kernel_close)
                kernel_close = kernel_close if kernel_close % 2 == 1 else kernel_close + 1
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
                repaired = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
                self.intermediate_steps['9_stroke_repair'] = repaired.copy()
                final = repaired
            else:
                final = cleaned

            # Final polish: ensure strong binary
            final = (final > 127).astype(np.uint8) * 255
            self.intermediate_steps['10_final'] = final.copy()

            metrics = self.calculate_metrics(original_gray, final)

            processing_summary = {
                'pipeline_version': 'ROBUST',
                'threshold_method': threshold_method,
                'background_method': bg_method,
                'remove_lines': remove_lines,
                'enable_stroke_repair': enable_stroke_repair,
                'total_steps': len(self.intermediate_steps)
            }

            logger.info("=== PIPELINE ROBUST COMPLETE ===")
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary
            }

        except Exception as e:
            logger.error(f"Error in robust pipeline: {str(e)}")
            raise

    def process_pipeline_handwriting(self, image, settings):
        """Handwriting-focused pipeline (historical/old ink).

        Target: like the sample you provided (paper texture removed, ink strokes preserved).
        Approach: strong background estimation + subtraction, then Sauvola on enhanced ink.

        Key knob: `strokeRepairLevel`: 'none' | 'light' | 'strong' (default: 'strong')
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== PIPELINE HANDWRITING - Ink/Paper Separation ===")
            logger.info(f"Settings: {settings}")

            gray = self._ensure_gray_uint8(image)
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray.copy()

            # STEP 1: Denoise (median + bilateral)
            denoised = cv2.medianBlur(gray, 3)
            d = int(settings.get('bilateralDiameter', 11))
            sc = float(settings.get('bilateralSigmaColor', 85))
            ss = float(settings.get('bilateralSigmaSpace', 85))
            denoised = cv2.bilateralFilter(denoised, d, sc, ss)
            self.intermediate_steps['1b_denoised'] = denoised.copy()

            # STEP 2: Strong background estimation (large opening + blur)
            bg_open = int(settings.get('bgOpenKernel', 61))
            bg_open = max(15, bg_open)
            bg_open = bg_open if bg_open % 2 == 1 else bg_open + 1
            kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_open, bg_open))
            background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)

            bg_blur = int(settings.get('bgBlurKernel', 51))
            bg_blur = max(15, bg_blur)
            bg_blur = bg_blur if bg_blur % 2 == 1 else bg_blur + 1
            background = cv2.GaussianBlur(background, (bg_blur, bg_blur), 0)
            self.intermediate_steps['2_background'] = background.copy()

            # STEP 3: Background subtraction (paper -> flat, ink -> strong)
            # Use int16 to avoid underflow and keep ink signal.
            diff16 = background.astype(np.int16) - denoised.astype(np.int16)
            diff = np.clip(diff16, 0, 255).astype(np.uint8)
            diff = self._minmax_to_uint8(diff)
            self.intermediate_steps['3_bg_subtracted'] = diff.copy()

            # STEP 4: Contrast enhancement
            clip = float(settings.get('claheClipLimit', 3.5))
            grid = int(settings.get('claheGrid', 8))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            enhanced = clahe.apply(diff)
            self.intermediate_steps['4_clahe'] = enhanced.copy()

            # STEP 5: Invert to make ink dark / paper bright, then Sauvola
            # Background subtraction makes ink bright; inversion restores conventional polarity.
            ink_dark = cv2.bitwise_not(enhanced)
            self.intermediate_steps['5_ink_dark'] = ink_dark.copy()

            # Sauvola (large window reduces paper noise sensitivity)
            window_size = int(settings.get('sauvolaWindowSize', 51))
            window_size = max(15, window_size)
            window_size = window_size if window_size % 2 == 1 else window_size + 1
            k = float(settings.get('sauvolaK', 0.25))
            thresh = threshold_sauvola(ink_dark, window_size=window_size, k=k)
            binary = (ink_dark > thresh).astype(np.uint8) * 255
            self.intermediate_steps['6_binary'] = binary.copy()

            # STEP 6: Morphology (configurable)
            stroke_level = (settings.get('strokeRepairLevel') or 'strong').lower()
            working = binary
            if stroke_level in ('light', 'strong'):
                close_ks = 3 if stroke_level == 'light' else 4
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
                working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel_close)

                if stroke_level == 'strong':
                    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
                    working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel_h)

            open_ks = int(settings.get('kernelOpening', 2))
            open_ks = max(1, open_ks)
            open_ks = open_ks if open_ks % 2 == 1 else open_ks + 1
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
            working = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel_open)
            self.intermediate_steps['7_morphology'] = working.copy()

            # STEP 7: Remove small components (paper specks)
            min_area = int(settings.get('minComponentArea', 50))
            cleaned = self._remove_small_components(working, min_size=min_area)
            self.intermediate_steps['8_components'] = cleaned.copy()

            # STEP 8: Final closing (optional)
            if stroke_level == 'strong':
                kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_final)
                self.intermediate_steps['9_final_close'] = cleaned.copy()

            final = (cleaned > 127).astype(np.uint8) * 255
            self.intermediate_steps['10_final'] = final.copy()

            metrics = self.calculate_metrics(original_gray, final)
            processing_summary = {
                'pipeline_version': 'HANDWRITING',
                'stroke_repair_level': stroke_level,
                'bg_open_kernel': bg_open,
                'bg_blur_kernel': bg_blur,
                'sauvola_window_size': window_size,
                'sauvola_k': k,
                'total_steps': len(self.intermediate_steps),
            }

            logger.info("=== PIPELINE HANDWRITING COMPLETE ===")
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': processing_summary,
            }

        except Exception as e:
            logger.error(f"Error in handwriting pipeline: {str(e)}")
            raise

    def _divide_conquer_threshold(self, gray, block_size=64):
        """
        Divide & Conquer local thresholding
        Chia ảnh thành blocks, threshold từng block riêng
        """
        h, w = gray.shape
        
        # Padding để chia đều
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        ph, pw = padded.shape
        result_padded = np.zeros((ph, pw), dtype=np.uint8)
        
        for y in range(0, ph, block_size):
            for x in range(0, pw, block_size):
                block = padded[y:y+block_size, x:x+block_size]
                
                # Local statistics
                block_mean = np.mean(block)
                block_std = np.std(block)
                
                # Sauvola-like threshold
                k = 0.2
                R = 128
                threshold = block_mean * (1 + k * (block_std / R - 1))
                
                # Apply threshold
                binary_block = (block > threshold).astype(np.uint8) * 255
                result_padded[y:y+block_size, x:x+block_size] = binary_block
        
        # Crop về kích thước gốc
        return result_padded[:h, :w]

    def _ensure_gray_uint8(self, image):
        """Ensure input is grayscale uint8."""
        if image is None:
            raise ValueError("image is None")

        if len(image.shape) == 3:
            # Heuristic: most callers use RGB; cv2.imread gives BGR.
            # Use COLOR_RGB2GRAY because upstream API passes numpy from PIL (RGB).
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            gray_f = gray.astype(np.float32)
            min_v = float(np.min(gray_f))
            max_v = float(np.max(gray_f))
            if max_v > min_v:
                gray_f = (gray_f - min_v) * (255.0 / (max_v - min_v))
            else:
                gray_f = np.zeros_like(gray_f)
            gray = np.clip(gray_f, 0, 255).astype(np.uint8)

        return gray

    def _normalize_illumination_divide(self, gray, blur_ksize=35):
        """Normalize uneven illumination using division by a blurred background."""
        blur_ksize = max(3, int(blur_ksize))
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        background = np.clip(background, 1, 255).astype(np.uint8)
        normalized = cv2.divide(gray, background, scale=255)
        return normalized.astype(np.uint8)

    def _minmax_to_uint8(self, image):
        """Fast min-max normalize to uint8 [0,255]."""
        img = image.astype(np.float32)
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        if max_v <= min_v:
            return np.zeros_like(image, dtype=np.uint8)
        img = (img - min_v) * (255.0 / (max_v - min_v))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _difference_of_gaussians(self, gray, sigma_small=1.0, sigma_large=8.0):
        """Difference of Gaussians tuned to make dark ink strokes pop on paper.

        Returns float32 image with higher values around strokes.
        """
        sigma_small = max(0.1, float(sigma_small))
        sigma_large = max(sigma_small + 0.1, float(sigma_large))

        # Use sigma-based blurs; kernel=(0,0) lets OpenCV pick size from sigma
        small = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small)
        large = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_large, sigmaY=sigma_large)

        # large - small yields positive response on dark strokes
        dog = large.astype(np.float32) - small.astype(np.float32)
        return dog

    def _normalize_minmax_uint8(self, img):
        """Min-max normalize any array to uint8 [0..255]."""
        img_f = img.astype(np.float32)
        min_v = float(np.min(img_f))
        max_v = float(np.max(img_f))
        if max_v <= min_v:
            return np.zeros(img_f.shape, dtype=np.uint8)
        out = (img_f - min_v) * (255.0 / (max_v - min_v))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_hat_background_removal(self, gray, method='auto', kernel_size=25):
        """Apply top-hat/black-hat to suppress background artifacts."""
        kernel_size = max(3, int(kernel_size))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        mean_val = float(np.mean(gray))
        chosen = method
        if method == 'auto':
            chosen = 'blackhat' if mean_val > 127 else 'tophat'

        if chosen == 'blackhat':
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            cleaned = cv2.subtract(gray, blackhat)
            return cleaned.astype(np.uint8)

        if chosen == 'tophat':
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            cleaned = cv2.add(gray, tophat)
            return np.clip(cleaned, 0, 255).astype(np.uint8)

        return gray

    def _binarize(self, gray, method='sauvola', settings=None):
        """Binarize image into white background (255) and black text (0)."""
        if settings is None:
            settings = {}

        method = (method or 'sauvola').lower()
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        if method in ('adaptive_gaussian', 'adaptive_mean'):
            block_size = int(settings.get('adaptiveBlockSize', 25))
            block_size = max(3, block_size)
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            c = int(settings.get('adaptiveC', 8))
            adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 'adaptive_gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C
            return cv2.adaptiveThreshold(gray, 255, adaptive_type, cv2.THRESH_BINARY, block_size, c)

        # Default: Sauvola (good for uneven illumination)
        window_size = int(settings.get('sauvolaWindowSize', 31))
        window_size = max(3, window_size)
        window_size = window_size if window_size % 2 == 1 else window_size + 1
        k = float(settings.get('sauvolaK', 0.2))
        thresh = threshold_sauvola(gray, window_size=window_size, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary

    def _remove_table_lines(self, binary, scale=40.0):
        """Remove long horizontal/vertical lines from a binary document.

        Input/Output: binary with white background (255), black text (0).
        Returns: (binary_without_lines, line_mask)
        """
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        h, w = binary.shape
        inv = cv2.bitwise_not(binary)  # text/lines as white

        # Kernel length proportional to image size
        scale = float(scale) if scale else 40.0
        horiz_len = max(15, int(w / scale))
        vert_len = max(15, int(h / scale))

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

        horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel)
        vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel)

        line_mask = cv2.bitwise_or(horiz, vert)
        # Keep mask thin to avoid eating text
        thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_mask = cv2.erode(line_mask, thin_kernel, iterations=1)

        inv_wo_lines = cv2.bitwise_and(inv, cv2.bitwise_not(line_mask))
        out = cv2.bitwise_not(inv_wo_lines)
        out = (out > 127).astype(np.uint8) * 255
        return out, line_mask
    
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
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            original_gray = gray.copy()
            self.intermediate_steps['1_grayscale'] = gray.copy()
            logger.info(f"Step 1: Grayscale - shape: {gray.shape}")
            
            # Normalize để chuẩn hóa độ sáng
            gray = self._minmax_to_uint8(gray)
            
            # Bilateral filter - làm mịn nền, giữ cạnh chữ
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            self.intermediate_steps['1b_filtered'] = gray.copy()
            logger.info("Step 1b: Bilateral filter applied")
            
            # ========== BƯỚC 2: THRESHOLD THÍCH ỨNG ==========
            threshold_method = settings.get('thresholdMethod', 'adaptive_gaussian')
            
            if threshold_method == 'adaptive_gaussian':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 15, 8
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
            kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_closing + 1, 2))
            closed_h = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close_h)
            
            kernel_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, kernel_closing + 1))
            closed = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_close_v)
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing, kernel_closing))
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close)
            
            self.intermediate_steps['3_closing'] = closed.copy()
            logger.info(f"Step 3: Closing with kernel {kernel_closing}")
            
            # ========== BƯỚC 4: LÀM SẠCH NHIỄU NHẸ ==========
            kernel_opening = settings.get('kernelOpening', 2)
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
            
            # ========== BƯỚC 6: HẬU XỬ LÝ ==========
            kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final)
            final = self._remove_small_components(final, min_size=30)
            
            self.intermediate_steps['6_final'] = final.copy()
            logger.info("Step 6: Final cleanup done")
            
            # ========== BƯỚC 7: TĂNG CƯỜNG HIỂN THỊ ==========
            contrast_method = settings.get('contrastMethod', 'none')
            if contrast_method != 'none':
                enhanced = self.enhance_contrast_adaptive(
                    original_gray,
                    method=contrast_method,
                    clip_limit=settings.get('claheClipLimit', 2.0)
                )
                self.intermediate_steps['7_enhanced'] = enhanced
            
            # ========== ĐÁNH GIÁ ==========
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
        """Loại nền tăng cường"""
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        mean_val = np.mean(gray)
        logger.info(f"Mean brightness: {mean_val:.2f}")
        
        if method == 'blackhat' or (method == 'auto' and mean_val > 127):
            logger.info("Applying Black-hat for light background")
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bg)
            cleaned = cv2.subtract(gray, blackhat)
            _, result = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'tophat' or (method == 'auto' and mean_val <= 127):
            logger.info("Applying Top-hat for dark background")
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_bg)
            _, result = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            result = binary
        
        result = cv2.bitwise_and(result, binary)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)
        
        return result
    
    def _remove_small_components(self, binary, min_size=30):
        """Loại bỏ các thành phần nhỏ (nhiễu)"""
        inverted = cv2.bitwise_not(binary)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        
        mask = np.zeros(binary.shape, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        return cv2.bitwise_not(mask)
    
    def remove_background_fixed(self, image, method='auto', kernel_size=15):
        """Background Removal (V2)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if method == 'blackhat':
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            result = cv2.subtract(image, blackhat)
            result = np.clip(result + 10, 0, 255).astype(np.uint8)
        elif method == 'tophat':
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            result = cv2.add(image, tophat)
        else:
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            result = cv2.subtract(image, blackhat)
            tophat_scaled = (tophat * 0.5).astype(np.uint8)
            result = cv2.add(result, tophat_scaled)
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def enhance_contrast_adaptive(self, image, method='clahe_masked', clip_limit=2.0):
        """Contrast Enhancement"""
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
        
        elif method == 'clahe_masked':
            edges = cv2.Canny(image, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(edges, kernel, iterations=2)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            result = np.where(mask > 0, enhanced, image)
            return result.astype(np.uint8)
        
        elif method == 'histogram_eq':
            return cv2.equalizeHist(image)
        
        return image
    
    def apply_threshold(self, image, method='otsu'):
        """Threshold methods"""
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
            if len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            if original.shape != processed.shape:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
            psnr_value = psnr(original, processed, data_range=255)
            ssim_raw = ssim(original, processed, data_range=255)
            ssim_value = ssim_raw[0] if isinstance(ssim_raw, tuple) else ssim_raw
            mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
            
            return {
                'psnr': round(float(psnr_value), 2),
                'ssim': round(float(ssim_value), 4),
                'mse': round(float(mse), 2)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'psnr': 0, 'ssim': 0, 'mse': 0}

    # ==================== PIPELINE SIMPLE (MỚI - DỄ DÙNG) ====================
    def process_pipeline_simple(self, image, settings):
        """
        Pipeline SIMPLE - Xử lý nhẹ nhàng, dễ hiểu
        Phù hợp cho tài liệu scan thông thường
        
        Các bước:
        1. Grayscale
        2. Gaussian blur nhẹ (denoise)
        3. Otsu threshold
        4. Opening (loại nhiễu nhỏ)
        5. Closing (nối nét chữ)
        6. Auto-invert nếu cần
        """
        try:
            self.intermediate_steps = {}
            logger.info("=== PIPELINE SIMPLE ===")
            
            # Step 0: Input
            gray = self._ensure_gray_uint8(image)
            original_gray = gray.copy()
            h, w = gray.shape
            self.intermediate_steps['0_original'] = gray.copy()
            logger.info(f"Input: {w}x{h}")
            
            # Step 1: Denoise với Gaussian blur nhẹ
            blur_size = int(settings.get('blurSize', 3))
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            if blur_size >= 3:
                blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            else:
                blurred = gray.copy()
            self.intermediate_steps['1_blurred'] = blurred.copy()
            logger.info("Step 1: Gaussian blur")
            
            # Step 2: Threshold
            threshold_method = settings.get('thresholdMethod', 'otsu')
            if threshold_method == 'otsu':
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_method == 'adaptive':
                block = int(settings.get('adaptiveBlock', 31))
                block = max(3, block if block % 2 == 1 else block + 1)
                c = int(settings.get('adaptiveC', 10))
                binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, block, c)
            else:
                _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            self.intermediate_steps['2_threshold'] = binary.copy()
            logger.info(f"Step 2: Threshold ({threshold_method})")
            
            # Step 3: Opening (loại nhiễu nhỏ)
            k_open = int(settings.get('openingKernel', 2))
            if k_open >= 2:
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            else:
                opened = binary.copy()
            self.intermediate_steps['3_opening'] = opened.copy()
            logger.info(f"Step 3: Opening (k={k_open})")
            
            # Step 4: Closing (nối nét chữ)
            k_close = int(settings.get('closingKernel', 2))
            if k_close >= 2:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
            else:
                closed = opened.copy()
            self.intermediate_steps['4_closing'] = closed.copy()
            logger.info(f"Step 4: Closing (k={k_close})")
            
            # Step 5: Auto-invert nếu cần (đảm bảo nền trắng, chữ đen)
            white_ratio = np.mean(closed == 255)
            if white_ratio < 0.4:  # Nếu ít hơn 40% là trắng → bị invert
                final = cv2.bitwise_not(closed)
                logger.info("Step 5: Auto-inverted (white bg, black text)")
            else:
                final = closed.copy()
                logger.info("Step 5: No invert needed")
            
            self.intermediate_steps['5_final'] = final.copy()
            
            # Metrics
            metrics = self.calculate_metrics(original_gray, final)
            logger.info(f"Metrics: PSNR={metrics['psnr']}, SSIM={metrics['ssim']}")
            logger.info("=== PIPELINE SIMPLE COMPLETE ===")
            
            return {
                'final_image': final,
                'intermediate_steps': self.intermediate_steps,
                'metrics': metrics,
                'original_gray': original_gray,
                'processing_summary': {
                    'pipeline_version': 'SIMPLE_V1.0',
                    'steps_executed': 5,
                    'threshold_method': threshold_method,
                    'opening_kernel': k_open,
                    'closing_kernel': k_close,
                }
            }
            
        except Exception as e:
            logger.error(f"Error in simple pipeline: {str(e)}")
            raise
