"""
Test Script - Kiểm tra xử lý ảnh chữ viết tay
Chạy trực tiếp để test trên ảnh local
"""

import cv2
import numpy as np
import os
import sys

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_handwriting_processing(image_path, output_dir="test_output"):
    """
    Test xử lý ảnh chữ viết tay với nhiều phương pháp khác nhau
    """
    
    # Tạo output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc ảnh
    print(f"\n{'='*60}")
    print(f"TESTING IMAGE: {image_path}")
    print(f"{'='*60}")
    
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
    
    cv2.imwrite(f"{output_dir}/0_original.png", gray)
    print(f"✓ Saved: 0_original.png")
    
    # ========== TEST CÁC PHƯƠNG PHÁP ==========
    
    print("\n--- PHƯƠNG PHÁP 1: Basic Pipeline ---")
    result1 = method_basic(gray.copy())
    cv2.imwrite(f"{output_dir}/1_basic.png", result1)
    print(f"✓ Saved: 1_basic.png")
    
    print("\n--- PHƯƠNG PHÁP 2: Adaptive Threshold + Closing mạnh ---")
    result2 = method_adaptive_strong_closing(gray.copy())
    cv2.imwrite(f"{output_dir}/2_adaptive_strong_closing.png", result2)
    print(f"✓ Saved: 2_adaptive_strong_closing.png")
    
    print("\n--- PHƯƠNG PHÁP 3: Bilateral + Morphology ---")
    result3 = method_bilateral_morphology(gray.copy())
    cv2.imwrite(f"{output_dir}/3_bilateral_morphology.png", result3)
    print(f"✓ Saved: 3_bilateral_morphology.png")
    
    print("\n--- PHƯƠNG PHÁP 4: Dilation trước (Làm dày nét) ---")
    result4 = method_dilation_first(gray.copy())
    cv2.imwrite(f"{output_dir}/4_dilation_first.png", result4)
    print(f"✓ Saved: 4_dilation_first.png")
    
    print("\n--- PHƯƠNG PHÁP 5: CLAHE + Otsu ---")
    result5 = method_clahe_otsu(gray.copy())
    cv2.imwrite(f"{output_dir}/5_clahe_otsu.png", result5)
    print(f"✓ Saved: 5_clahe_otsu.png")
    
    print("\n--- PHƯƠNG PHÁP 6: Skeleton + Dilation (Làm liền nét tốt nhất) ---")
    result6 = method_skeleton_dilate(gray.copy())
    cv2.imwrite(f"{output_dir}/6_skeleton_dilate.png", result6)
    print(f"✓ Saved: 6_skeleton_dilate.png")
    
    print("\n--- PHƯƠNG PHÁP 7: Blackhat + Closing (Loại vết bẩn) ---")
    result7 = method_blackhat_closing(gray.copy())
    cv2.imwrite(f"{output_dir}/7_blackhat_closing.png", result7)
    print(f"✓ Saved: 7_blackhat_closing.png")
    
    print("\n--- PHƯƠNG PHÁP 8: FULL PIPELINE V3 (Tổng hợp tốt nhất) ---")
    result8 = method_full_pipeline_v3(gray.copy())
    cv2.imwrite(f"{output_dir}/8_full_pipeline_v3.png", result8)
    print(f"✓ Saved: 8_full_pipeline_v3.png")
    
    # Tạo comparison image
    create_comparison(gray, [result1, result2, result3, result4, result5, result6, result7, result8],
                      ["Basic", "Adaptive", "Bilateral", "Dilation", "CLAHE", "Skeleton", "Blackhat", "Pipeline V3"],
                      f"{output_dir}/comparison.png")
    
    print(f"\n{'='*60}")
    print(f"DONE! Check results in: {output_dir}/")
    print(f"{'='*60}")


def method_basic(gray):
    """Phương pháp cơ bản: Otsu + Opening + Closing"""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def method_adaptive_strong_closing(gray):
    """Adaptive threshold + Closing mạnh để nối nét chữ"""
    # Bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    
    # Closing mạnh - theo hướng ngang để nối chữ
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    closed_h = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    
    # Closing theo hướng dọc
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    closed = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_v)
    
    # Opening nhẹ để loại nhiễu
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return result


def method_bilateral_morphology(gray):
    """Bilateral filter mạnh + Morphology"""
    # Bilateral filter mạnh
    filtered = cv2.bilateralFilter(gray, 11, 100, 100)
    
    # Otsu
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing trước
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening nhẹ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return result


def method_dilation_first(gray):
    """Dilation trước để làm dày nét chữ"""
    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Bilateral filter
    filtered = cv2.bilateralFilter(normalized, 9, 75, 75)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 10
    )
    
    # DILATION TRƯỚC - làm dày nét chữ (quan trọng!)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
    
    # Closing để nối
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening nhẹ loại nhiễu
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return result


def method_clahe_otsu(gray):
    """CLAHE tăng contrast + Otsu"""
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Otsu
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return result


def method_skeleton_dilate(gray):
    """Tạo skeleton rồi dilate để có nét đều"""
    # Normalize + filter
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    filtered = cv2.bilateralFilter(normalized, 9, 75, 75)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8  # INV để chữ trắng trên nền đen
    )
    
    # Thinning/Skeleton
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) if hasattr(cv2, 'ximgproc') else binary
    
    # Dilate skeleton để nét dày hơn, đều hơn
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Invert lại
    result = cv2.bitwise_not(dilated)
    
    return result


def method_blackhat_closing(gray):
    """Blackhat để loại vết bẩn + Closing"""
    # Blackhat để phát hiện vết bẩn
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bg)
    
    # Trừ vết bẩn
    cleaned = cv2.subtract(gray, blackhat)
    
    # Normalize
    cleaned = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)
    
    # Bilateral
    filtered = cv2.bilateralFilter(cleaned, 9, 75, 75)
    
    # Threshold
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing mạnh
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening nhẹ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    return result


def method_full_pipeline_v3(gray):
    """
    PIPELINE V3 - Tổng hợp các phương pháp tốt nhất
    Tối ưu cho chữ viết tay đứt gãy
    """
    print("  → Step 1: Normalize + CLAHE")
    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # CLAHE tăng contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    print("  → Step 2: Bilateral filter (giữ cạnh, mịn nền)")
    # Bilateral filter - quan trọng để giữ cạnh chữ
    filtered = cv2.bilateralFilter(enhanced, 11, 80, 80)
    
    print("  → Step 3: Adaptive Gaussian Threshold")
    # Adaptive threshold - tốt hơn cho lighting không đều
    binary = cv2.adaptiveThreshold(
        filtered, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=17,  # Tăng block size
        C=7
    )
    
    print("  → Step 4: Dilation nhẹ (làm dày nét)")
    # Dilation nhẹ - làm dày nét trước
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
    
    print("  → Step 5: Closing theo hướng (nối nét ngang)")
    # Closing theo hướng ngang - nối các phần chữ gãy ngang
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    closed_h = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_h)
    
    print("  → Step 6: Closing theo hướng (nối nét dọc)")
    # Closing theo hướng dọc
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    closed_v = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_v)
    
    print("  → Step 7: Closing đều (nối tổng thể)")
    # Closing với kernel tròn
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, kernel_close)
    
    print("  → Step 8: Opening nhẹ (loại nhiễu nhỏ)")
    # Opening nhẹ - chỉ loại nhiễu rất nhỏ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    print("  → Step 9: Loại bỏ thành phần nhỏ")
    # Loại bỏ thành phần nhỏ (noise)
    result = remove_small_components(opened, min_size=50)
    
    print("  → Step 10: Final closing")
    # Final closing để đảm bảo nét liền
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_final)
    
    return result


def remove_small_components(binary, min_size=50):
    """Loại bỏ các thành phần nhỏ (noise)"""
    # Tìm contours
    contours, _ = cv2.findContours(
        cv2.bitwise_not(binary), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Tạo mask
    mask = np.ones_like(binary) * 255
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_size:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    
    # Áp dụng mask
    result = cv2.bitwise_or(binary, cv2.bitwise_not(mask))
    
    return result


def create_comparison(original, results, labels, output_path):
    """Tạo ảnh so sánh các phương pháp"""
    n = len(results) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    
    h, w = original.shape[:2]
    
    # Resize nếu ảnh quá lớn
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        results = [cv2.resize(r, (new_w, new_h)) for r in results]
        h, w = new_h, new_w
    
    # Tạo canvas
    padding = 10
    text_height = 30
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_height
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    # Vẽ original
    all_images = [original] + results
    all_labels = ["Original"] + labels
    
    for i, (img, label) in enumerate(zip(all_images, all_labels)):
        row = i // cols
        col = i % cols
        
        x = col * cell_w + padding
        y = row * cell_h + padding + text_height
        
        # Vẽ ảnh
        canvas[y:y+h, x:x+w] = img
        
        # Vẽ label
        cv2.putText(canvas, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"\n✓ Comparison saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test image processing')
    parser.add_argument('--image', '-i', type=str, 
                        help='Path to input image',
                        default='Frontend/public/test/handwriting_sample.png')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory',
                        default='test_output')
    
    args = parser.parse_args()
    
    # Nếu không có ảnh test, tạo ảnh mẫu
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        print("Creating sample test image...")
        
        # Tạo thư mục
        os.makedirs(os.path.dirname(args.image) if os.path.dirname(args.image) else '.', exist_ok=True)
        
        # Tạo ảnh mẫu với chữ đứt gãy
        sample = np.ones((300, 400), dtype=np.uint8) * 255
        
        # Vẽ text mô phỏng chữ viết tay đứt gãy
        cv2.putText(sample, "Test", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 0, 2)
        cv2.putText(sample, "Hand", (50, 180), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 0, 2)
        cv2.putText(sample, "Writing", (50, 260), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 0, 2)
        
        # Thêm nhiễu
        noise = np.random.randint(0, 50, sample.shape, dtype=np.uint8)
        sample = cv2.add(sample, noise)
        
        # Thêm vết bẩn
        cv2.circle(sample, (300, 150), 30, 200, -1)
        
        cv2.imwrite(args.image, sample)
        print(f"Created sample image: {args.image}")
    
    test_handwriting_processing(args.image, args.output)
