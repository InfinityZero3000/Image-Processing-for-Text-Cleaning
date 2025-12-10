"""
Test Pipeline V3 - So sánh GaussianDiff, DivideConquer và V3 Combined
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_processing import ImageProcessor


def pipeline_gaussian_difference(gray):
    """Gaussian Difference - loại nền hiệu quả"""
    # Gaussian blur với sigma nhỏ (chi tiết)
    small_blur = cv2.GaussianBlur(gray, (3, 3), 1)
    
    # Gaussian blur với sigma lớn (background)
    large_blur = cv2.GaussianBlur(gray, (51, 51), 20)
    
    # Trừ để lấy chi tiết
    diff = cv2.subtract(large_blur, small_blur)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    # Otsu threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def pipeline_divide_conquer(gray, block_size=64):
    """Divide & Conquer - local thresholding"""
    h, w = gray.shape
    
    # Padding
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    ph, pw = padded.shape
    result_padded = np.zeros((ph, pw), dtype=np.uint8)
    
    for y in range(0, ph, block_size):
        for x in range(0, pw, block_size):
            block = padded[y:y+block_size, x:x+block_size]
            
            block_mean = np.mean(block)
            block_std = np.std(block)
            
            k = 0.2
            R = 128
            threshold = block_mean * (1 + k * (block_std / R - 1))
            
            binary_block = (block > threshold).astype(np.uint8) * 255
            result_padded[y:y+block_size, x:x+block_size] = binary_block
    
    result = result_padded[:h, :w]
    
    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    # Median filter
    result = cv2.medianBlur(result, 3)
    
    return result


def create_comparison():
    """Tạo ảnh so sánh các phương pháp"""
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
    
    # Test GaussianDiff
    print("Testing GaussianDiff...")
    gauss_diff = pipeline_gaussian_difference(gray.copy())
    cv2.imwrite(f"{output_dir}/compare_gaussdiff.png", gauss_diff)
    
    # Test DivideConquer
    print("Testing DivideConquer...")
    div_conquer = pipeline_divide_conquer(gray.copy())
    cv2.imwrite(f"{output_dir}/compare_divconquer.png", div_conquer)
    
    # Test Pipeline V3 (Combined)
    print("Testing Pipeline V3 (Combined)...")
    processor = ImageProcessor()
    settings = {
        'blockSize': 64,
        'kernelClosing': 3,
        'kernelOpening': 2,
        'minNoiseSize': 40
    }
    result = processor.process_pipeline_v3(image, settings)
    v3_result = result['final_image']
    cv2.imwrite(f"{output_dir}/compare_v3_combined.png", v3_result)
    
    # Create comparison grid
    print("Creating comparison grid...")
    h, w = gray.shape
    
    # Resize nếu quá lớn
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h))
        gauss_diff = cv2.resize(gauss_diff, (new_w, new_h))
        div_conquer = cv2.resize(div_conquer, (new_w, new_h))
        v3_result = cv2.resize(v3_result, (new_w, new_h))
        h, w = new_h, new_w
    
    padding = 15
    text_h = 30
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_h
    
    canvas = np.ones((2 * cell_h, 2 * cell_w), dtype=np.uint8) * 255
    
    images = [
        ('Original', gray),
        ('GaussianDiff', gauss_diff),
        ('DivideConquer', div_conquer),
        ('V3 Combined', v3_result)
    ]
    
    for i, (label, img) in enumerate(images):
        row, col = i // 2, i % 2
        x = col * cell_w + padding
        y = row * cell_h + padding + text_h
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    cv2.imwrite(f"{output_dir}/v3_comparison.png", canvas)
    
    # Create before/after
    print("Creating before/after...")
    ba_h = max(h, 300)
    ba_w = w * 2 + padding * 3
    ba_canvas = np.ones((ba_h + text_h + padding * 2, ba_w), dtype=np.uint8) * 240
    
    # Before
    x1 = padding
    y1 = padding + text_h
    ba_canvas[y1:y1+h, x1:x1+w] = gray
    cv2.putText(ba_canvas, "BEFORE", (x1 + w//4, text_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    # After
    x2 = w + padding * 2
    ba_canvas[y1:y1+h, x2:x2+w] = v3_result
    cv2.putText(ba_canvas, "AFTER (V3)", (x2 + w//5, text_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    cv2.imwrite(f"{output_dir}/v3_before_after.png", ba_canvas)
    
    print(f"\n{'='*50}")
    print("DONE! Files saved:")
    print(f"  - {output_dir}/compare_gaussdiff.png")
    print(f"  - {output_dir}/compare_divconquer.png")
    print(f"  - {output_dir}/compare_v3_combined.png")
    print(f"  - {output_dir}/v3_comparison.png")
    print(f"  - {output_dir}/v3_before_after.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    create_comparison()
