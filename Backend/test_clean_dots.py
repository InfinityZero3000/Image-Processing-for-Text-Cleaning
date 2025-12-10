"""
FINAL CLEAN - Loại bỏ các chấm đen nhiễu nhỏ
Giữ nguyên phương pháp tốt, chỉ thêm bước lọc noise
"""

import cv2
import numpy as np
import os


def remove_small_dots(binary, min_area=15, max_area=None):
    """
    Loại bỏ các chấm đen nhỏ (noise) dựa trên diện tích
    - min_area: diện tích tối thiểu để giữ lại (pixel)
    - Các thành phần nhỏ hơn min_area sẽ bị xóa
    """
    # Đảo màu: chữ đen -> trắng để tìm connected components
    inverted = cv2.bitwise_not(binary)
    
    # Tìm tất cả connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    
    # Tạo mask mới, chỉ giữ các thành phần đủ lớn
    clean_mask = np.zeros_like(binary)
    
    removed_count = 0
    kept_count = 0
    
    for i in range(1, num_labels):  # Bỏ qua background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Chỉ giữ nếu đủ lớn
        if area >= min_area:
            if max_area is None or area <= max_area:
                clean_mask[labels == i] = 255
                kept_count += 1
            else:
                removed_count += 1
        else:
            removed_count += 1
    
    # Đảo lại: chữ trắng -> đen
    result = cv2.bitwise_not(clean_mask)
    
    print(f"  Removed {removed_count} small components, kept {kept_count}")
    
    return result


def method_gaussdiff_clean(gray, min_dot_size=15):
    """
    GaussianDiff + Loại chấm nhỏ
    """
    # 1. Bilateral nhẹ
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 2. Gaussian Difference
    bg = cv2.GaussianBlur(filtered, (51, 51), 0)
    diff = cv2.subtract(bg, filtered)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. CLAHE nhẹ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff)
    
    # 4. Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Closing nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 6. LOẠI BỎ CHẤM NHỎ
    print(f"  Removing dots smaller than {min_dot_size} pixels...")
    clean = remove_small_dots(binary, min_area=min_dot_size)
    
    return clean


def method_sauvola_clean(gray, min_dot_size=15):
    """
    Sauvola + Loại chấm nhỏ
    """
    from skimage.filters import threshold_sauvola
    
    # 1. Bilateral nhẹ
    filtered = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # 2. Sauvola threshold
    thresh = threshold_sauvola(filtered, window_size=51, k=0.2)
    binary = (filtered > thresh).astype(np.uint8) * 255
    
    # 3. Closing nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. LOẠI BỎ CHẤM NHỎ
    print(f"  Removing dots smaller than {min_dot_size} pixels...")
    clean = remove_small_dots(binary, min_area=min_dot_size)
    
    return clean


def test_different_sizes():
    """Test với các ngưỡng min_area khác nhau"""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load
    image = cv2.imread("../Frontend/public/test/image-1765276809510.png")
    if image is None:
        print("ERROR: Cannot load image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image: {gray.shape}")
    
    # Test GaussianDiff với các min_area khác nhau
    print("\n=== Testing GaussianDiff with different min_area ===")
    
    min_areas = [10, 15, 20, 25, 30, 40, 50]
    results = {"Original": gray}
    
    for min_area in min_areas:
        print(f"\nMin area = {min_area}:")
        result = method_gaussdiff_clean(gray.copy(), min_dot_size=min_area)
        results[f"min{min_area}"] = result
        cv2.imwrite(f"{output_dir}/clean_min{min_area}.png", result)
    
    # Comparison grid
    create_grid(results, f"{output_dir}/clean_comparison.png")
    
    print(f"\n{'='*50}")
    print("DONE! Check test_output/clean_comparison.png")
    print("="*50)


def create_grid(results, output_path):
    """Tạo grid so sánh"""
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    
    first_img = list(results.values())[0]
    h, w = first_img.shape[:2]
    
    # Resize
    max_size = 220
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    padding = 8
    text_h = 22
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_h
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    for i, (label, img) in enumerate(results.items()):
        row, col = i // cols, i % cols
        x = col * cell_w + padding
        y = row * cell_h + padding + text_h
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Grid: {output_path}")


if __name__ == "__main__":
    test_different_sizes()
