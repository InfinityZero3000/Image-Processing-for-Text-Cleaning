"""
ADVANCED Document Image Binarization - Phương pháp tốt nhất từ nghiên cứu
Sử dụng Sauvola, Niblack và các thuật toán chuyên dụng cho tài liệu cổ
"""

import cv2
import numpy as np
import os
import sys
from skimage import filters, exposure, restoration
from skimage.filters import threshold_sauvola, threshold_niblack, threshold_local
from skimage.morphology import disk, square, dilation, erosion, opening, closing
from skimage.util import img_as_ubyte, img_as_float
from scipy import ndimage

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_advanced_binarization(image_path, output_dir="test_output"):
    """
    Test các phương pháp binarization chuyên dụng cho tài liệu cổ/chữ viết tay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ADVANCED DOCUMENT BINARIZATION TEST")
    print(f"Image: {image_path}")
    print(f"{'='*70}")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read image {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Chuyển grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    cv2.imwrite(f"{output_dir}/00_original.png", gray)
    print(f"✓ Saved: 00_original.png")
    
    results = {}
    
    # ========== TEST CÁC PHƯƠNG PHÁP TIÊN TIẾN ==========
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 1: Sauvola (Tốt nhất cho tài liệu)")
    print("="*50)
    result1 = method_sauvola_advanced(gray.copy())
    cv2.imwrite(f"{output_dir}/01_sauvola.png", result1)
    results['Sauvola'] = result1
    print(f"✓ Saved: 01_sauvola.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 2: Niblack (Tốt cho chữ viết tay)")
    print("="*50)
    result2 = method_niblack_advanced(gray.copy())
    cv2.imwrite(f"{output_dir}/02_niblack.png", result2)
    results['Niblack'] = result2
    print(f"✓ Saved: 02_niblack.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 3: Local Mean (Đơn giản, hiệu quả)")
    print("="*50)
    result3 = method_local_mean(gray.copy())
    cv2.imwrite(f"{output_dir}/03_local_mean.png", result3)
    results['LocalMean'] = result3
    print(f"✓ Saved: 03_local_mean.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 4: Wolf (Biến thể của Sauvola)")
    print("="*50)
    result4 = method_wolf(gray.copy())
    cv2.imwrite(f"{output_dir}/04_wolf.png", result4)
    results['Wolf'] = result4
    print(f"✓ Saved: 04_wolf.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 5: Contrast Stretch + Sauvola")
    print("="*50)
    result5 = method_contrast_sauvola(gray.copy())
    cv2.imwrite(f"{output_dir}/05_contrast_sauvola.png", result5)
    results['ContrastSauvola'] = result5
    print(f"✓ Saved: 05_contrast_sauvola.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 6: CLAHE + Niblack")
    print("="*50)
    result6 = method_clahe_niblack(gray.copy())
    cv2.imwrite(f"{output_dir}/06_clahe_niblack.png", result6)
    results['CLAHENiblack'] = result6
    print(f"✓ Saved: 06_clahe_niblack.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 7: Background Estimation + Binarization")
    print("="*50)
    result7 = method_background_estimation(gray.copy())
    cv2.imwrite(f"{output_dir}/07_background_est.png", result7)
    results['BackgroundEst'] = result7
    print(f"✓ Saved: 07_background_est.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 8: Multi-scale Sauvola (DIBCO style)")
    print("="*50)
    result8 = method_multiscale_sauvola(gray.copy())
    cv2.imwrite(f"{output_dir}/08_multiscale_sauvola.png", result8)
    results['MultiSauvola'] = result8
    print(f"✓ Saved: 08_multiscale_sauvola.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 9: Gaussian + Local Otsu")
    print("="*50)
    result9 = method_gaussian_local_otsu(gray.copy())
    cv2.imwrite(f"{output_dir}/09_gaussian_local_otsu.png", result9)
    results['GaussianOtsu'] = result9
    print(f"✓ Saved: 09_gaussian_local_otsu.png")
    
    print("\n" + "="*50)
    print("PHƯƠNG PHÁP 10: BEST - Combination (Tối ưu nhất)")
    print("="*50)
    result10 = method_best_combination(gray.copy())
    cv2.imwrite(f"{output_dir}/10_best_combination.png", result10)
    results['BEST'] = result10
    print(f"✓ Saved: 10_best_combination.png")
    
    # Tạo comparison image
    create_comparison_grid(gray, results, f"{output_dir}/comparison_advanced.png")
    
    print(f"\n{'='*70}")
    print(f"DONE! Results saved in: {output_dir}/")
    print(f"{'='*70}")
    print("\n📌 KHUYẾN NGHỊ:")
    print("   - Sauvola: Tốt nhất cho tài liệu có nền không đều")
    print("   - Niblack: Tốt cho chữ viết tay đậm")
    print("   - BEST Combination: Kết hợp tốt nhất cho hầu hết trường hợp")


# ============== CÁC PHƯƠNG PHÁP BINARIZATION TIÊN TIẾN ==============

def method_sauvola_advanced(gray, window_size=25, k=0.2):
    """
    Sauvola binarization - Tốt nhất cho tài liệu
    Paper: Sauvola, J., & Pietikäinen, M. (2000)
    """
    print("  → Áp dụng Gaussian blur nhẹ")
    # Denoise nhẹ
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    
    print(f"  → Sauvola threshold (window={window_size}, k={k})")
    # Sauvola threshold
    thresh_sauvola = threshold_sauvola(denoised, window_size=window_size, k=k)
    binary = (denoised > thresh_sauvola).astype(np.uint8) * 255
    
    print("  → Post-processing: Closing + Opening")
    # Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary


def method_niblack_advanced(gray, window_size=25, k=0.2):
    """
    Niblack binarization - Tốt cho chữ viết tay
    Paper: Niblack, W. (1986)
    """
    print("  → Normalize + Bilateral filter")
    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # Bilateral filter để giữ cạnh
    filtered = cv2.bilateralFilter(normalized, 7, 50, 50)
    
    print(f"  → Niblack threshold (window={window_size}, k={k})")
    # Niblack threshold (k âm để lấy text tối trên nền sáng)
    thresh_niblack = threshold_niblack(filtered, window_size=window_size, k=k)
    binary = (filtered > thresh_niblack).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_local_mean(gray, block_size=35, offset=10):
    """
    Local mean thresholding - Đơn giản và hiệu quả
    """
    print(f"  → Local mean threshold (block={block_size}, offset={offset})")
    
    # Gaussian blur nhẹ
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Local threshold với mean
    thresh_local = threshold_local(blurred, block_size=block_size, method='mean', offset=offset)
    binary = (blurred > thresh_local).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_wolf(gray, window_size=25, k=0.5):
    """
    Wolf binarization - Biến thể cải tiến của Sauvola
    Paper: Wolf, C., Jolion, J. M., & Chassaing, F. (2002)
    """
    print("  → Tính local mean và standard deviation")
    
    # Tính local statistics
    kernel_size = window_size
    mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray.astype(np.float64)**2, (kernel_size, kernel_size))
    std = np.sqrt(sqr_mean - mean**2)
    
    # Wolf formula
    R = np.max(std)
    M = np.min(gray)
    
    print(f"  → Wolf threshold (k={k}, R={R:.2f}, M={M})")
    threshold = mean - k * (1 - std/R) * (mean - M)
    
    binary = (gray > threshold).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_contrast_sauvola(gray):
    """
    Contrast enhancement + Sauvola
    """
    print("  → Contrast stretching")
    # Contrast stretching
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = exposure.rescale_intensity(gray, in_range=(p2, p98))
    stretched = img_as_ubyte(stretched)
    
    print("  → CLAHE enhancement")
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)
    
    print("  → Sauvola threshold")
    # Sauvola
    thresh_sauvola = threshold_sauvola(enhanced, window_size=25, k=0.15)
    binary = (enhanced > thresh_sauvola).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_clahe_niblack(gray):
    """
    CLAHE + Niblack combination
    """
    print("  → CLAHE enhancement")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    print("  → Gaussian blur")
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    print("  → Niblack threshold")
    thresh_niblack = threshold_niblack(blurred, window_size=21, k=0.1)
    binary = (blurred > thresh_niblack).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_background_estimation(gray):
    """
    Background estimation + Binarization
    Ước lượng background rồi trừ đi
    """
    print("  → Ước lượng background bằng morphological opening")
    
    # Ước lượng background bằng morphological opening với kernel lớn
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    
    print("  → Trừ background khỏi ảnh gốc")
    # Trừ background
    diff = cv2.absdiff(gray, background)
    
    # Normalize
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  → Otsu threshold")
    # Otsu threshold
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert nếu cần (text đen trên nền trắng)
    binary = cv2.bitwise_not(binary)
    
    return binary


def method_multiscale_sauvola(gray):
    """
    Multi-scale Sauvola - DIBCO style
    Kết hợp nhiều window size
    """
    print("  → Gaussian blur preprocessing")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    results = []
    window_sizes = [11, 21, 31, 51]
    
    print(f"  → Multi-scale Sauvola với windows: {window_sizes}")
    for ws in window_sizes:
        thresh = threshold_sauvola(blurred, window_size=ws, k=0.2)
        binary = (blurred > thresh).astype(np.float32)
        results.append(binary)
    
    print("  → Kết hợp bằng voting (majority voting)")
    # Kết hợp bằng voting
    combined = np.mean(results, axis=0)
    binary = (combined > 0.5).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_gaussian_local_otsu(gray):
    """
    Gaussian filter + Local Otsu
    """
    print("  → Gaussian filter")
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    print("  → Local Otsu threshold")
    # Local thresholding với Gaussian method
    thresh_local = threshold_local(blurred, block_size=35, method='gaussian', offset=5)
    binary = (blurred > thresh_local).astype(np.uint8) * 255
    
    print("  → Post-processing")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_best_combination(gray):
    """
    BEST - Kết hợp các phương pháp tốt nhất
    Pipeline tối ưu cho tài liệu cổ/chữ viết tay
    """
    print("  [STEP 1] Normalize + Denoise")
    # 1. Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Bilateral filter - giữ cạnh, mịn vùng đồng nhất
    denoised = cv2.bilateralFilter(normalized, 9, 75, 75)
    
    print("  [STEP 2] Background estimation")
    # 3. Background estimation
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)
    
    # 4. Subtract background
    diff = cv2.absdiff(denoised, background)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    print("  [STEP 3] Contrast enhancement (CLAHE)")
    # 5. CLAHE để tăng contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    print("  [STEP 4] Sauvola binarization")
    # 6. Sauvola threshold - tốt nhất cho tài liệu
    thresh_sauvola = threshold_sauvola(enhanced, window_size=25, k=0.2)
    binary = (enhanced > thresh_sauvola).astype(np.uint8) * 255
    
    print("  [STEP 5] Morphological cleanup")
    # 7. Morphological cleanup
    # Closing để nối các nét đứt
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening nhẹ để loại nhiễu nhỏ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    print("  [STEP 6] Remove small noise components")
    # 8. Loại bỏ thành phần nhỏ (noise)
    binary = remove_small_noise(binary, min_size=30)
    
    print("  [STEP 7] Final closing để đảm bảo chữ liền")
    # 9. Final closing
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)
    
    return binary


def remove_small_noise(binary, min_size=30):
    """Loại bỏ các thành phần nhiễu nhỏ"""
    # Tìm contours
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo mask cho các thành phần nhỏ
    mask = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # Loại bỏ các thành phần nhỏ
    result = cv2.bitwise_or(binary, mask)
    
    return result


def create_comparison_grid(original, results, output_path):
    """Tạo ảnh so sánh dạng lưới"""
    n = len(results) + 1
    cols = 4
    rows = (n + cols - 1) // cols
    
    h, w = original.shape[:2]
    
    # Resize nếu ảnh quá lớn
    max_size = 250
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    # Tạo canvas
    padding = 10
    text_height = 25
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
        
        # Vẽ ảnh
        canvas[y:y+h, x:x+w] = img
        
        # Vẽ label
        cv2.putText(canvas, label[:15], (x, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Comparison grid saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced document binarization test')
    parser.add_argument('--image', '-i', type=str,
                        help='Path to input image',
                        default='../Frontend/public/test/image-1765276809510.png')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory',
                        default='test_output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        # Thử các đường dẫn khác
        alternatives = [
            'Frontend/public/test/image-1765276809510.png',
            '../Frontend/public/test/image-1765276809510.png',
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                args.image = alt
                break
    
    test_advanced_binarization(args.image, args.output)
