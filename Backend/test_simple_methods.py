"""
SIMPLE & EFFECTIVE - Phương pháp đơn giản nhưng hiệu quả
Nguyên tắc: ÍT xử lý hơn = KẾT QUẢ tốt hơn

Vấn đề trước đó:
- Quá nhiều blur → mờ chi tiết
- Quá nhiều closing → méo chữ  
- Quá nhiều bước → tích lũy lỗi

Giải pháp: Chỉ làm những gì CẦN THIẾT
"""

import cv2
import numpy as np
import os


def method_simple_gaussdiff(gray):
    """
    Phương pháp 1: GaussianDiff đơn giản nhất
    Chỉ 3 bước: Gaussian Diff → Threshold → Nhẹ cleanup
    """
    # Chỉ 1 lần Gaussian Difference
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.subtract(background, gray)
    
    # Normalize
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Otsu threshold trực tiếp
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Chỉ 1 lần closing nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_simple_adaptive(gray):
    """
    Phương pháp 2: Adaptive threshold đơn giản
    Không cần Gaussian Diff, dùng adaptive trực tiếp
    """
    # Chỉ 1 lần bilateral nhẹ
    filtered = cv2.bilateralFilter(gray, 5, 50, 50)
    
    # Adaptive threshold - block size lớn hơn
    binary = cv2.adaptiveThreshold(
        filtered, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,  # block size lớn
        10   # C constant
    )
    
    # Closing nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_simple_contrast(gray):
    """
    Phương pháp 3: Tăng contrast rồi threshold
    """
    # Contrast stretching đơn giản
    p5, p95 = np.percentile(gray, (5, 95))
    stretched = np.clip((gray - p5) * 255.0 / (p95 - p5), 0, 255).astype(np.uint8)
    
    # CLAHE nhẹ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)
    
    # Otsu
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def method_minimal_gaussdiff(gray):
    """
    Phương pháp 4: MINIMAL - Ít bước nhất có thể
    """
    # Background estimation
    bg = cv2.GaussianBlur(gray, (71, 71), 0)
    
    # Subtract
    diff = cv2.absdiff(bg, gray)
    
    # Threshold
    _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    return binary


def method_sauvola_only(gray):
    """
    Phương pháp 5: Chỉ dùng Sauvola - không xử lý thêm
    """
    from skimage.filters import threshold_sauvola
    
    # Sauvola với window lớn
    thresh = threshold_sauvola(gray, window_size=51, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    
    return binary


def method_niblack_only(gray):
    """
    Phương pháp 6: Chỉ dùng Niblack
    """
    from skimage.filters import threshold_niblack
    
    thresh = threshold_niblack(gray, window_size=51, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    
    return binary


def method_best_simple(gray):
    """
    Phương pháp 7: BEST SIMPLE - Kết hợp tốt nhất nhưng đơn giản
    """
    # 1. Chỉ 1 lần bilateral nhẹ (giữ edge, giảm noise)
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 2. Gaussian Difference đơn giản
    bg = cv2.GaussianBlur(filtered, (51, 51), 0)
    diff = cv2.subtract(bg, filtered)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 3. CLAHE nhẹ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff)
    
    # 4. Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Chỉ 1 closing nhẹ với kernel nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def method_no_processing(gray):
    """
    Phương pháp 8: Chỉ Otsu - không xử lý gì cả
    Để so sánh baseline
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def test_simple_methods():
    """Test các phương pháp đơn giản"""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load
    image = cv2.imread("../Frontend/public/test/image-1765276809510.png")
    if image is None:
        print("ERROR: Cannot load image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Image: {gray.shape}")
    
    methods = [
        ("1_SimpleGaussDiff", method_simple_gaussdiff),
        ("2_SimpleAdaptive", method_simple_adaptive),
        ("3_SimpleContrast", method_simple_contrast),
        ("4_MinimalGaussDiff", method_minimal_gaussdiff),
        ("5_SauvolaOnly", method_sauvola_only),
        ("6_NiblackOnly", method_niblack_only),
        ("7_BestSimple", method_best_simple),
        ("8_OtsuOnly", method_no_processing),
    ]
    
    results = {}
    
    for name, func in methods:
        print(f"Testing {name}...")
        try:
            result = func(gray.copy())
            results[name] = result
            cv2.imwrite(f"{output_dir}/simple_{name.lower()}.png", result)
            print(f"  ✓ Saved")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Comparison grid
    create_grid(gray, results, f"{output_dir}/simple_comparison.png")
    
    print(f"\n{'='*50}")
    print("DONE! Check test_output/simple_comparison.png")
    print("="*50)


def create_grid(original, results, output_path):
    """Tạo grid so sánh"""
    n = len(results) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    h, w = original.shape
    
    # Resize
    max_size = 250
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    padding = 10
    text_h = 25
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_h
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    all_imgs = [("Original", original)] + list(results.items())
    
    for i, (label, img) in enumerate(all_imgs):
        row, col = i // cols, i % cols
        x = col * cell_w + padding
        y = row * cell_h + padding + text_h
        canvas[y:y+h, x:x+w] = img
        # Rút gọn label
        short_label = label.split("_")[-1] if "_" in label else label
        cv2.putText(canvas, short_label[:15], (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Grid: {output_path}")


if __name__ == "__main__":
    test_simple_methods()
