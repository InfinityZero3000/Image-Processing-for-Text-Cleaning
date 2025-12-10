"""
GaussianDiff Enhanced - Phiên bản tối ưu nhất
Cải tiến từ GaussianDiff cơ bản với:
1. Multi-scale Gaussian Difference
2. Adaptive CLAHE
3. Strong morphological cleanup
4. Better noise removal
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def gaussian_diff_v1_basic(gray):
    """V1: GaussianDiff cơ bản"""
    small_blur = cv2.GaussianBlur(gray, (3, 3), 1)
    large_blur = cv2.GaussianBlur(gray, (51, 51), 20)
    diff = cv2.subtract(large_blur, small_blur)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def gaussian_diff_v2_multiscale(gray):
    """V2: Multi-scale Gaussian Difference"""
    # Denoise trước
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Multi-scale Gaussian - kết hợp nhiều mức
    results = []
    scales = [(3, 31), (3, 51), (5, 71)]  # (small_kernel, large_kernel)
    
    for small_k, large_k in scales:
        small_blur = cv2.GaussianBlur(denoised, (small_k, small_k), small_k//2)
        large_blur = cv2.GaussianBlur(denoised, (large_k, large_k), large_k//3)
        diff = cv2.subtract(large_blur, small_blur)
        results.append(diff.astype(np.float32))
    
    # Combine bằng max (giữ chi tiết rõ nhất)
    combined = np.maximum.reduce(results)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    # Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def gaussian_diff_v3_enhanced(gray):
    """V3: Enhanced với adaptive parameters"""
    # Step 1: Strong denoise
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.bilateralFilter(denoised, 11, 85, 85)
    
    # Step 2: Multi-scale Gaussian Difference
    small_blur1 = cv2.GaussianBlur(denoised, (3, 3), 1)
    large_blur1 = cv2.GaussianBlur(denoised, (51, 51), 15)
    diff1 = cv2.subtract(large_blur1, small_blur1)
    
    small_blur2 = cv2.GaussianBlur(denoised, (5, 5), 2)
    large_blur2 = cv2.GaussianBlur(denoised, (71, 71), 25)
    diff2 = cv2.subtract(large_blur2, small_blur2)
    
    # Combine
    combined = cv2.addWeighted(diff1, 0.6, diff2, 0.4, 0)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 3: Strong CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    # Step 4: Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 5: Strong closing để nối nét
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Closing theo hướng ngang (nối chữ)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    # Step 6: Opening nhẹ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    return binary


def gaussian_diff_v4_best(gray):
    """V4: BEST - Tối ưu nhất cho chữ viết tay"""
    print("  [1] Strong denoise: Median + Bilateral")
    denoised = cv2.medianBlur(gray, 3)
    denoised = cv2.bilateralFilter(denoised, 11, 90, 90)
    
    print("  [2] Multi-scale Gaussian Difference (3 scales)")
    # Scale 1: Fine details
    diff1 = cv2.subtract(
        cv2.GaussianBlur(denoised, (41, 41), 12),
        cv2.GaussianBlur(denoised, (3, 3), 1)
    )
    
    # Scale 2: Medium details  
    diff2 = cv2.subtract(
        cv2.GaussianBlur(denoised, (61, 61), 18),
        cv2.GaussianBlur(denoised, (5, 5), 1.5)
    )
    
    # Scale 3: Coarse details
    diff3 = cv2.subtract(
        cv2.GaussianBlur(denoised, (81, 81), 25),
        cv2.GaussianBlur(denoised, (7, 7), 2)
    )
    
    # Combine với weighted average
    combined = (diff1.astype(np.float32) * 0.4 + 
                diff2.astype(np.float32) * 0.35 + 
                diff3.astype(np.float32) * 0.25)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    print("  [3] Adaptive CLAHE")
    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    print("  [4] Contrast stretching")
    p2, p98 = np.percentile(enhanced, (2, 98))
    enhanced = np.clip((enhanced - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    
    print("  [5] Otsu threshold")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("  [6] Strong morphological closing")
    # Ellipse closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Horizontal closing (nối nét ngang)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    # Vertical closing (nối nét dọc)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
    
    print("  [7] Light opening để loại nhiễu")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    print("  [8] Remove small noise components")
    binary = remove_small_noise(binary, min_size=50)
    
    print("  [9] Final closing")
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)
    
    return binary


def gaussian_diff_v5_ultra(gray):
    """V5: ULTRA - Phiên bản mạnh nhất"""
    print("  [1] Heavy denoise")
    # Median trước để loại salt-pepper
    denoised = cv2.medianBlur(gray, 5)
    # Non-local means denoising (mạnh hơn bilateral)
    denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
    # Bilateral để giữ edge
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    print("  [2] Background estimation")
    # Ước lượng background bằng morphological opening lớn
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    print("  [3] Multi-scale Gaussian Difference")
    # Subtract background
    bg_removed = cv2.subtract(background, denoised)
    
    # Thêm Gaussian Difference
    diff1 = cv2.subtract(
        cv2.GaussianBlur(denoised, (51, 51), 15),
        cv2.GaussianBlur(denoised, (3, 3), 1)
    )
    diff2 = cv2.subtract(
        cv2.GaussianBlur(denoised, (71, 71), 22),
        cv2.GaussianBlur(denoised, (5, 5), 1.5)
    )
    
    # Combine tất cả
    combined = (bg_removed.astype(np.float32) * 0.3 +
                diff1.astype(np.float32) * 0.4 +
                diff2.astype(np.float32) * 0.3)
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    print("  [4] Strong CLAHE")
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(combined)
    
    print("  [5] Contrast stretching")
    p1, p99 = np.percentile(enhanced, (1, 99))
    enhanced = np.clip((enhanced - p1) * 255.0 / (p99 - p1 + 1), 0, 255).astype(np.uint8)
    
    print("  [6] Otsu threshold")
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print("  [7] Aggressive closing")
    # Multi-pass closing
    kernel_c1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_c1)
    
    kernel_c2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_c2)
    
    # Directional closing
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
    
    print("  [8] Opening")
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    print("  [9] Remove noise")
    binary = remove_small_noise(binary, min_size=60)
    
    print("  [10] Final cleanup")
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_final)
    
    return binary


def remove_small_noise(binary, min_size=30):
    """Loại bỏ nhiễu nhỏ"""
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            inverted[labels == i] = 0
    
    return cv2.bitwise_not(inverted)


def test_all_versions():
    """Test tất cả các phiên bản"""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image_path = "../Frontend/public/test/image-1765276809510.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Cannot read {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image loaded: {gray.shape}")
    print("="*60)
    
    versions = [
        ("V1_Basic", gaussian_diff_v1_basic),
        ("V2_MultiScale", gaussian_diff_v2_multiscale),
        ("V3_Enhanced", gaussian_diff_v3_enhanced),
        ("V4_BEST", gaussian_diff_v4_best),
        ("V5_ULTRA", gaussian_diff_v5_ultra),
    ]
    
    results = {}
    
    for name, func in versions:
        print(f"\n{'='*60}")
        print(f"Testing {name}...")
        print("="*60)
        result = func(gray.copy())
        results[name] = result
        cv2.imwrite(f"{output_dir}/gaussdiff_{name.lower()}.png", result)
        print(f"✓ Saved: gaussdiff_{name.lower()}.png")
    
    # Create comparison grid
    print("\n" + "="*60)
    print("Creating comparison grid...")
    print("="*60)
    
    create_comparison_grid(gray, results, f"{output_dir}/gaussdiff_comparison.png")
    
    # Create before/after with BEST
    create_before_after(gray, results["V4_BEST"], f"{output_dir}/gaussdiff_before_after_v4.png", "V4 BEST")
    create_before_after(gray, results["V5_ULTRA"], f"{output_dir}/gaussdiff_before_after_v5.png", "V5 ULTRA")
    
    print("\n" + "="*60)
    print("DONE! All files saved in test_output/")
    print("="*60)


def create_comparison_grid(original, results, output_path):
    """Tạo grid so sánh"""
    n = len(results) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    h, w = original.shape[:2]
    
    max_size = 280
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    padding = 12
    text_h = 28
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_h
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    all_images = [("Original", original)] + list(results.items())
    
    for i, (label, img) in enumerate(all_images):
        row, col = i // cols, i % cols
        x = col * cell_w + padding
        y = row * cell_h + padding + text_h
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Comparison: {output_path}")


def create_before_after(before, after, output_path, label="AFTER"):
    """Tạo ảnh trước/sau"""
    h, w = before.shape[:2]
    
    max_size = 400
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        before = cv2.resize(before, (new_w, new_h))
        after = cv2.resize(after, (new_w, new_h))
        h, w = new_h, new_w
    
    padding = 20
    text_h = 40
    
    canvas_w = w * 2 + padding * 3
    canvas_h = h + padding * 2 + text_h
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 240
    
    x1 = padding
    y1 = padding + text_h
    canvas[y1:y1+h, x1:x1+w] = before
    cv2.putText(canvas, "BEFORE", (x1 + w//4, text_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    
    x2 = w + padding * 2
    canvas[y1:y1+h, x2:x2+w] = after
    cv2.putText(canvas, label, (x2 + w//5, text_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Before/After: {output_path}")


if __name__ == "__main__":
    test_all_versions()
