"""
IMPROVED Document Binarization for Handwriting Recognition
Cải tiến xử lý ảnh cho nhận diện chữ viết tay tốt hơn

Vấn đề đã phát hiện từ kết quả trước:
1. Quá nhiều nhiễu nền (background noise)
2. Nét chữ bị đứt gãy
3. Sauvola window size quá nhỏ → bắt cả nhiễu

Giải pháp:
1. Background subtraction mạnh hơn trước khi binarization
2. Window size lớn hơn cho Sauvola
3. Median filter để loại nhiễu salt-pepper
4. Closing mạnh hơn để nối nét
5. Contrast enhancement tốt hơn
"""

import cv2
import numpy as np
import os
import sys
from skimage.filters import threshold_sauvola, threshold_niblack

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def improved_handwriting_binarization(image_path, output_dir="test_output"):
    """
    Pipeline cải tiến cho nhận diện chữ viết tay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"IMPROVED HANDWRITING BINARIZATION")
    print(f"Image: {image_path}")
    print(f"{'='*70}")
    
    # Đọc ảnh
    original = cv2.imread(image_path)
    if original is None:
        print(f"ERROR: Cannot read image {image_path}")
        return
    
    print(f"Image shape: {original.shape}")
    
    # Chuyển grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/improved_00_gray.png", gray)
    
    results = {}
    
    # ========== PIPELINE 1: Conservative (Bảo toàn chi tiết) ==========
    print("\n" + "="*50)
    print("[PIPELINE 1] CONSERVATIVE - Bảo toàn chi tiết")
    print("="*50)
    result1 = pipeline_conservative(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_01_conservative.png", result1)
    results['P1_Conservative'] = result1
    print("✓ Saved: improved_01_conservative.png")
    
    # ========== PIPELINE 2: Aggressive Background Removal ==========
    print("\n" + "="*50)
    print("[PIPELINE 2] AGGRESSIVE BG REMOVAL - Loại nền mạnh")
    print("="*50)
    result2 = pipeline_aggressive_bg(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_02_aggressive_bg.png", result2)
    results['P2_AggressiveBG'] = result2
    print("✓ Saved: improved_02_aggressive_bg.png")
    
    # ========== PIPELINE 3: Large Window Sauvola ==========
    print("\n" + "="*50)
    print("[PIPELINE 3] LARGE WINDOW SAUVOLA - Sauvola cửa sổ lớn")
    print("="*50)
    result3 = pipeline_large_window_sauvola(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_03_large_sauvola.png", result3)
    results['P3_LargeSauvola'] = result3
    print("✓ Saved: improved_03_large_sauvola.png")
    
    # ========== PIPELINE 4: Divide and Conquer ==========
    print("\n" + "="*50)
    print("[PIPELINE 4] DIVIDE & CONQUER - Chia và xử lý")
    print("="*50)
    result4 = pipeline_divide_conquer(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_04_divide_conquer.png", result4)
    results['P4_DivideConquer'] = result4
    print("✓ Saved: improved_04_divide_conquer.png")
    
    # ========== PIPELINE 5: Strong Closing (Nối nét mạnh) ==========
    print("\n" + "="*50)
    print("[PIPELINE 5] STRONG CLOSING - Nối nét mạnh")
    print("="*50)
    result5 = pipeline_strong_closing(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_05_strong_closing.png", result5)
    results['P5_StrongClosing'] = result5
    print("✓ Saved: improved_05_strong_closing.png")
    
    # ========== PIPELINE 6: Gaussian Difference ==========
    print("\n" + "="*50)
    print("[PIPELINE 6] GAUSSIAN DIFFERENCE - Trừ Gaussian")
    print("="*50)
    result6 = pipeline_gaussian_difference(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_06_gaussian_diff.png", result6)
    results['P6_GaussianDiff'] = result6
    print("✓ Saved: improved_06_gaussian_diff.png")
    
    # ========== PIPELINE 7: Multi-stage Enhancement ==========
    print("\n" + "="*50)
    print("[PIPELINE 7] MULTI-STAGE ENHANCEMENT")
    print("="*50)
    result7 = pipeline_multistage(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_07_multistage.png", result7)
    results['P7_MultiStage'] = result7
    print("✓ Saved: improved_07_multistage.png")
    
    # ========== PIPELINE 8: BEST - Kết hợp tối ưu ==========
    print("\n" + "="*50)
    print("[PIPELINE 8] BEST - Kết hợp tối ưu cho chữ viết tay")
    print("="*50)
    result8 = pipeline_best_for_handwriting(gray.copy())
    cv2.imwrite(f"{output_dir}/improved_08_BEST.png", result8)
    results['P8_BEST'] = result8
    print("✓ Saved: improved_08_BEST.png")
    
    # Tạo comparison
    create_comparison_grid(gray, results, f"{output_dir}/improved_comparison.png")
    
    # Before/After với BEST
    create_before_after(gray, result8, f"{output_dir}/improved_before_after.png")
    
    print(f"\n{'='*70}")
    print(f"HOÀN THÀNH! Xem kết quả trong: {output_dir}/")
    print(f"{'='*70}")
    print("\n📌 KHUYẾN NGHỊ:")
    print("   - P1_Conservative: Giữ chi tiết nhưng có thể còn nhiễu")
    print("   - P2_AggressiveBG: Loại nền tốt nhưng có thể mất chi tiết mảnh")
    print("   - P3_LargeSauvola: Ít nhiễu hơn, tốt cho text đều")
    print("   - P5_StrongClosing: Nét liền hơn")
    print("   - P8_BEST: Cân bằng giữa loại nhiễu và giữ chi tiết")


# ============== CÁC PIPELINE CẢI TIẾN ==============

def pipeline_conservative(gray):
    """
    Pipeline bảo toàn chi tiết - ưu tiên giữ nét chữ
    """
    print("  [1] Bilateral filter (giữ cạnh)")
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    print("  [2] CLAHE nhẹ")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    print("  [3] Sauvola (window=51, k=0.2)")
    thresh = threshold_sauvola(enhanced, window_size=51, k=0.2)
    binary = (enhanced > thresh).astype(np.uint8) * 255
    
    print("  [4] Closing nhẹ (3x3)")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    print("  [5] Median filter (loại nhiễu salt-pepper)")
    binary = cv2.medianBlur(binary, 3)
    
    return binary


def pipeline_aggressive_bg(gray):
    """
    Pipeline loại nền mạnh
    """
    print("  [1] Ước lượng background bằng morphological opening lớn")
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    
    print("  [2] Gaussian blur background")
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    print("  [3] Trừ background")
    # Tránh underflow
    diff = cv2.subtract(background, gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [4] CLAHE")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    print("  [5] Otsu threshold (sau khi đã loại nền)")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("  [6] Closing (3x3)")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    print("  [7] Loại nhiễu nhỏ")
    binary = remove_small_noise(binary, min_size=50)
    
    return binary


def pipeline_large_window_sauvola(gray):
    """
    Sauvola với window size lớn - giảm nhiễu
    """
    print("  [1] Bilateral filter")
    filtered = cv2.bilateralFilter(gray, 11, 85, 85)
    
    print("  [2] Normalize")
    normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [3] Sauvola (window=81, k=0.3)")
    # Window lớn = ít nhạy với nhiễu local
    thresh = threshold_sauvola(normalized, window_size=81, k=0.3)
    binary = (normalized > thresh).astype(np.uint8) * 255
    
    print("  [4] Closing 2 lần (2x2)")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    print("  [5] Median filter")
    binary = cv2.medianBlur(binary, 3)
    
    return binary


def pipeline_divide_conquer(gray):
    """
    Chia ảnh thành blocks, xử lý từng block
    """
    print("  [1] Chia ảnh thành các blocks và tính background local")
    
    h, w = gray.shape
    block_size = 64
    result = np.zeros_like(gray)
    
    # Padding để chia đều
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    ph, pw = padded.shape
    result_padded = np.zeros((ph, pw), dtype=np.uint8)
    
    for y in range(0, ph, block_size):
        for x in range(0, pw, block_size):
            block = padded[y:y+block_size, x:x+block_size]
            
            # Local thresholding cho mỗi block
            block_mean = np.mean(block)
            block_std = np.std(block)
            
            # Sauvola-like threshold cho block
            k = 0.2
            R = 128
            threshold = block_mean * (1 + k * (block_std / R - 1))
            
            binary_block = (block > threshold).astype(np.uint8) * 255
            result_padded[y:y+block_size, x:x+block_size] = binary_block
    
    # Crop về kích thước gốc
    result = result_padded[:h, :w]
    
    print("  [2] Closing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    print("  [3] Median filter")
    result = cv2.medianBlur(result, 3)
    
    return result


def pipeline_strong_closing(gray):
    """
    Pipeline với closing mạnh để nối nét đứt
    """
    print("  [1] Bilateral + Median filter")
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    filtered = cv2.medianBlur(filtered, 3)
    
    print("  [2] CLAHE")
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    print("  [3] Sauvola (window=41, k=0.2)")
    thresh = threshold_sauvola(enhanced, window_size=41, k=0.2)
    binary = (enhanced > thresh).astype(np.uint8) * 255
    
    print("  [4] Closing mạnh (4x4)")
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    print("  [5] Thêm Closing theo hướng (ngang/dọc)")
    # Closing ngang - nối các nét ngang
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    # Closing dọc - nối các nét dọc
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
    
    print("  [6] Opening nhẹ để loại nhiễu dày")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    return binary


def pipeline_gaussian_difference(gray):
    """
    Difference of Gaussians - loại nền hiệu quả
    """
    print("  [1] Gaussian blur với sigma nhỏ (chi tiết)")
    small_blur = cv2.GaussianBlur(gray, (3, 3), 1)
    
    print("  [2] Gaussian blur với sigma lớn (background)")
    large_blur = cv2.GaussianBlur(gray, (51, 51), 20)
    
    print("  [3] Trừ để lấy chi tiết")
    diff = cv2.subtract(large_blur, small_blur)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [4] CLAHE")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    print("  [5] Otsu threshold")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("  [6] Closing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def pipeline_multistage(gray):
    """
    Xử lý đa giai đoạn
    """
    print("  [STAGE 1] Tiền xử lý nặng")
    # Median để loại nhiễu salt-pepper
    denoised = cv2.medianBlur(gray, 5)
    # Bilateral để giữ cạnh
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    print("  [STAGE 2] Background estimation và subtraction")
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)
    background = cv2.GaussianBlur(background, (41, 41), 0)
    
    # Normalize background
    diff = cv2.absdiff(denoised, background)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [STAGE 3] Contrast enhancement")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized.astype(np.uint8))
    
    print("  [STAGE 4] Multi-scale thresholding")
    # Kết hợp nhiều window sizes
    results = []
    for ws in [31, 51, 71]:
        thresh = threshold_sauvola(enhanced, window_size=ws, k=0.25)
        binary = (enhanced > thresh).astype(np.float32)
        results.append(binary)
    
    # Voting
    combined = np.mean(results, axis=0)
    binary = (combined > 0.5).astype(np.uint8) * 255
    
    print("  [STAGE 5] Morphological cleanup")
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    print("  [STAGE 6] Loại nhiễu nhỏ")
    binary = remove_small_noise(binary, min_size=40)
    
    return binary


def pipeline_best_for_handwriting(gray):
    """
    BEST Pipeline cho nhận diện chữ viết tay cổ
    Kết hợp các kỹ thuật tốt nhất
    """
    print("  [STEP 1] Denoise: Median + Bilateral")
    # Median trước để loại nhiễu xung
    denoised = cv2.medianBlur(gray, 3)
    # Bilateral để giữ cạnh, mịn vùng đồng nhất
    denoised = cv2.bilateralFilter(denoised, 11, 85, 85)
    
    print("  [STEP 2] Background subtraction mạnh")
    # Background estimation với kernel lớn
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)
    # Blur thêm để mịn
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # Subtract và normalize
    diff = cv2.subtract(background.astype(np.int16), denoised.astype(np.int16))
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [STEP 3] CLAHE enhancement")
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff)
    
    print("  [STEP 4] Sauvola với window size lớn")
    # Window size lớn = ít nhạy với nhiễu cục bộ
    thresh = threshold_sauvola(enhanced, window_size=51, k=0.25)
    binary = (enhanced > thresh).astype(np.uint8) * 255
    
    print("  [STEP 5] Morphological: Closing mạnh để nối nét")
    # Closing với kernel tròn
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Thêm closing theo hướng để nối nét ngang
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    print("  [STEP 6] Opening nhẹ để loại nhiễu dày còn lại")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    print("  [STEP 7] Loại bỏ thành phần nhiễu nhỏ")
    binary = remove_small_noise(binary, min_size=50)
    
    print("  [STEP 8] Final closing để đảm bảo nét liền")
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)
    
    return binary


def remove_small_noise(binary, min_size=30):
    """Loại bỏ các thành phần nhiễu nhỏ"""
    # Đảo ngược để xử lý (text đen -> trắng)
    inverted = cv2.bitwise_not(binary)
    
    # Tìm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # Loại các thành phần nhỏ
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_size:
            inverted[labels == i] = 0
    
    # Đảo ngược lại
    return cv2.bitwise_not(inverted)


def create_comparison_grid(original, results, output_path):
    """Tạo ảnh so sánh dạng lưới"""
    n = len(results) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    h, w = original.shape[:2]
    
    # Resize nếu ảnh quá lớn
    max_size = 280
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    padding = 12
    text_height = 28
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_height
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    # Vẽ original
    all_images = [('Original', original)] + list(results.items())
    
    for i, (label, img) in enumerate(all_images):
        row = i // cols
        col = i % cols
        
        x = col * cell_w + padding
        y = row * cell_h + padding + text_height
        
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label[:18], (x, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Comparison: {output_path}")


def create_before_after(before, after, output_path):
    """Tạo ảnh so sánh trước/sau"""
    h, w = before.shape[:2]
    
    # Resize nếu cần
    max_size = 450
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        before = cv2.resize(before, (new_w, new_h))
        after = cv2.resize(after, (new_w, new_h))
        h, w = new_h, new_w
    
    padding = 25
    text_height = 45
    
    canvas_w = w * 2 + padding * 3
    canvas_h = h + padding * 2 + text_height
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 240
    
    # BEFORE
    x1 = padding
    y1 = padding + text_height
    canvas[y1:y1+h, x1:x1+w] = before
    cv2.putText(canvas, "BEFORE", (x1 + w//3, text_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    
    # AFTER
    x2 = w + padding * 2
    canvas[y1:y1+h, x2:x2+w] = after
    cv2.putText(canvas, "AFTER", (x2 + w//3, text_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    
    # Mũi tên
    arrow_x = x1 + w + padding//2
    arrow_y = y1 + h//2
    cv2.arrowedLine(canvas, (arrow_x - 10, arrow_y), (arrow_x + 10, arrow_y), 
                    0, 2, tipLength=0.5)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Before/After: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved handwriting binarization')
    parser.add_argument('--image', '-i', type=str,
                        help='Path to input image',
                        default='../Frontend/public/test/image-1765276809510.png')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory',
                        default='test_output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        alternatives = [
            'Frontend/public/test/image-1765276809510.png',
            '../Frontend/public/test/image-1765276809510.png',
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                args.image = alt
                break
    
    improved_handwriting_binarization(args.image, args.output)
