"""
Color Image Processing với GaussianDiff
- Giữ nguyên kênh đen (text)
- Giảm độ tương phản các kênh màu khác (nền)
"""

import cv2
import numpy as np
import os


def process_color_gaussdiff(image):
    """
    Xử lý ảnh màu với GaussianDiff
    - Phát hiện chữ đen bằng GaussianDiff
    - Giảm độ tương phản nền màu
    - Giữ nguyên chữ đen
    """
    # Convert sang grayscale để phát hiện text
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GaussianDiff để tách text khỏi nền
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.subtract(bg, gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold để tạo mask chữ
    _, text_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing nhẹ để làm liền chữ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    
    # Invert mask (255 = text, 0 = background)
    text_mask = cv2.bitwise_not(text_mask)
    
    # Giảm độ tương phản nền (làm mờ màu)
    # Chuyển sang float để tính toán
    result = image.astype(np.float32)
    
    # Làm mờ nền màu (tăng độ sáng, giảm saturation)
    for c in range(3):  # BGR channels
        # Tăng độ sáng nền lên gần trắng
        result[:, :, c] = result[:, :, c] * 0.5 + 128
    
    # Chuyển lại uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Apply mask: giữ nguyên vùng text đen từ ảnh gốc
    result = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(text_mask))
    original_text = cv2.bitwise_and(image, image, mask=text_mask)
    result = cv2.add(result, original_text)
    
    return result, text_mask


def process_color_v2(image):
    """
    Version 2: Giữ chữ đen, làm nền trắng hơn
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GaussianDiff
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.subtract(bg, gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold
    _, text_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = cv2.bitwise_not(text_mask)
    
    # Tạo nền trắng
    white_bg = np.ones_like(image) * 255
    
    # Lấy text từ ảnh gốc
    text_only = cv2.bitwise_and(image, image, mask=text_mask)
    bg_only = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(text_mask))
    
    # Kết hợp
    result = cv2.add(text_only, bg_only)
    
    return result, text_mask


def process_color_v3(image):
    """
    Version 3: Giữ text tối, làm nền sáng hơn (giữ một chút màu)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GaussianDiff
    bg = cv2.GaussianBlur(gray, (71, 71), 0)
    diff = cv2.subtract(bg, gray)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE nhẹ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff)
    
    # Threshold
    _, text_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = cv2.bitwise_not(text_mask)
    
    # Làm sáng nền màu (tăng brightness, giảm saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Tăng V (brightness), giảm S (saturation) cho vùng không phải text
    mask_inv = cv2.bitwise_not(text_mask).astype(np.float32) / 255.0
    
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask_inv * 0.7)  # Giảm saturation 70%
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + mask_inv * 100, 0, 255)  # Tăng brightness
    
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result, text_mask


def process_color_v4_best(image):
    """
    Version 4: BEST - Giữ text đen rõ, làm nền gần trắng với chút màu
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter nhẹ để giữ edge
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # GaussianDiff
    bg = cv2.GaussianBlur(filtered, (61, 61), 0)
    diff = cv2.subtract(bg, filtered)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff)
    
    # Threshold với Otsu
    _, text_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing nhẹ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    
    # Invert (255 = text)
    text_mask = cv2.bitwise_not(text_mask)
    
    # Tạo nền sáng với chút màu kem/vàng nhạt
    bg_color = np.array([245, 245, 235], dtype=np.uint8)  # Màu kem nhạt (BGR)
    light_bg = np.full_like(image, bg_color)
    
    # Extract text từ ảnh gốc (làm đậm hơn)
    text_only = cv2.bitwise_and(image, image, mask=text_mask)
    
    # Làm đậm text (giảm brightness của text)
    text_only_darker = (text_only.astype(np.float32) * 0.7).astype(np.uint8)
    
    # Combine
    bg_only = cv2.bitwise_and(light_bg, light_bg, mask=cv2.bitwise_not(text_mask))
    result = cv2.add(text_only_darker, bg_only)
    
    return result, text_mask


def test_color_processing():
    """Test các phương pháp xử lý màu"""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ảnh màu
    image_path = "../Frontend/public/test/image-1765276809510.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Cannot load {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    print(f"Color image: BGR format")
    print("="*60)
    
    # Save original
    cv2.imwrite(f"{output_dir}/color_00_original.png", image)
    
    # Test V1
    print("\n[V1] Basic color processing...")
    result_v1, mask_v1 = process_color_gaussdiff(image)
    cv2.imwrite(f"{output_dir}/color_v1_result.png", result_v1)
    cv2.imwrite(f"{output_dir}/color_v1_mask.png", mask_v1)
    print("✓ Saved: color_v1_result.png")
    
    # Test V2
    print("\n[V2] White background...")
    result_v2, mask_v2 = process_color_v2(image)
    cv2.imwrite(f"{output_dir}/color_v2_result.png", result_v2)
    print("✓ Saved: color_v2_result.png")
    
    # Test V3
    print("\n[V3] HSV adjustment...")
    result_v3, mask_v3 = process_color_v3(image)
    cv2.imwrite(f"{output_dir}/color_v3_result.png", result_v3)
    print("✓ Saved: color_v3_result.png")
    
    # Test V4
    print("\n[V4] BEST - Light background with darker text...")
    result_v4, mask_v4 = process_color_v4_best(image)
    cv2.imwrite(f"{output_dir}/color_v4_best.png", result_v4)
    cv2.imwrite(f"{output_dir}/color_v4_mask.png", mask_v4)
    print("✓ Saved: color_v4_best.png")
    
    # Create comparison
    print("\nCreating comparison...")
    create_comparison(image, result_v1, result_v2, result_v3, result_v4, 
                     f"{output_dir}/color_comparison.png")
    
    print("\n" + "="*60)
    print("DONE! Check test_output/color_*.png")
    print("="*60)


def create_comparison(original, v1, v2, v3, v4, output_path):
    """Tạo ảnh so sánh"""
    images = [
        ("Original", original),
        ("V1_Basic", v1),
        ("V2_WhiteBG", v2),
        ("V3_HSV", v3),
        ("V4_BEST", v4),
    ]
    
    h, w = original.shape[:2]
    
    # Resize if needed
    max_size = 280
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        images = [(name, cv2.resize(img, (new_w, new_h))) for name, img in images]
        h, w = new_h, new_w
    
    # Create grid
    cols = 3
    rows = 2
    padding = 10
    text_h = 25
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_h
    
    canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
    
    for i, (label, img) in enumerate(images):
        row, col = i // cols, i % cols
        x = col * cell_w + padding
        y = row * cell_h + padding + text_h
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"✓ Comparison: {output_path}")


if __name__ == "__main__":
    test_color_processing()
