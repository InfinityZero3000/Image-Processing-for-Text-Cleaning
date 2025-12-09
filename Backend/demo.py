#!/usr/bin/env python3
"""
Demo Script - Xử lý ảnh văn bản theo yêu cầu task
Chạy nhanh để test pipeline xử lý
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
from utils.image_processing import ImageProcessor

def create_sample_image():
    """Tạo ảnh mẫu có nhiễu để test"""
    # Tạo ảnh trắng
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Thêm text (giả lập chữ đen)
    cv2.putText(img, "Xu ly anh van ban", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    cv2.putText(img, "Document Image Processing", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.putText(img, "Clean & Clear Text", (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3)
    
    # Thêm nhiễu salt (điểm trắng)
    noise_salt = np.random.rand(*img.shape) > 0.98
    img[noise_salt] = 255
    
    # Thêm nhiễu pepper (điểm đen)
    noise_pepper = np.random.rand(*img.shape) > 0.98
    img[noise_pepper] = 0
    
    # Thêm vết bẩn (các vùng xám)
    for _ in range(10):
        x, y = np.random.randint(50, 550), np.random.randint(50, 350)
        cv2.circle(img, (x, y), np.random.randint(5, 15), 
                  np.random.randint(100, 200), -1)
    
    return img


def main():
    """Demo xử lý ảnh theo pipeline"""
    print("=" * 70)
    print("DEMO: XỬ LÝ ẢNH VĂN BẢN")
    print("=" * 70)
    print()
    
    # Tạo thư mục output
    output_dir = 'logs/demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo hoặc đọc ảnh mẫu
    if len(sys.argv) > 1:
        # Đọc ảnh từ file
        image_path = sys.argv[1]
        print(f"📁 Đọc ảnh từ: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return
        # Chuyển sang grayscale nếu cần
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Tạo ảnh mẫu
        print("📝 Tạo ảnh mẫu với nhiễu...")
        image = create_sample_image()
        # Lưu ảnh gốc
        cv2.imwrite(f'{output_dir}/0_original.png', image)
        print(f"   ✓ Lưu ảnh gốc: {output_dir}/0_original.png")
    
    print(f"   Kích thước: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # Khởi tạo processor
    processor = ImageProcessor()
    
    # Cấu hình xử lý theo yêu cầu task
    settings = {
        'thresholdMethod': 'otsu',       # Bước 1: Otsu threshold
        'kernelOpening': 2,               # Bước 2: Opening với kernel 2x2
        'kernelClosing': 3,               # Bước 3: Closing với kernel 3x3
        'backgroundRemoval': 'auto',      # Bước 4: Auto chọn black-hat/top-hat
        'backgroundKernel': 15,           # Kernel lớn cho background
        'contrastMethod': 'none'          # Bước 5: Không cần CLAHE
    }
    
    print("⚙️  CẤU HÌNH XỬ LÝ:")
    print(f"   • Threshold: {settings['thresholdMethod']}")
    print(f"   • Opening kernel: {settings['kernelOpening']}×{settings['kernelOpening']}")
    print(f"   • Closing kernel: {settings['kernelClosing']}×{settings['kernelClosing']}")
    print(f"   • Background removal: {settings['backgroundRemoval']}")
    print(f"   • Background kernel: {settings['backgroundKernel']}×{settings['backgroundKernel']}")
    print()
    
    # Xử lý ảnh
    print("🔄 BẮT ĐẦU XỬ LÝ...")
    print()
    
    start_time = datetime.now()
    result = processor.process_pipeline_v2(image, settings)
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Lưu các bước trung gian
    print("💾 LƯU CÁC BƯỚC XỬ LÝ:")
    for step_name, step_image in result['intermediate_steps'].items():
        output_path = f'{output_dir}/{step_name}.png'
        cv2.imwrite(output_path, step_image)
        print(f"   ✓ {step_name}: {output_path}")
    
    # Lưu ảnh cuối
    cv2.imwrite(f'{output_dir}/final_result.png', result['final_image'])
    print(f"   ✓ final_result: {output_dir}/final_result.png")
    print()
    
    # In summary
    summary = result.get('processing_summary', {})
    print("📊 TỔNG KẾT XỬ LÝ:")
    print(f"   • Thời gian: {processing_time:.2f} ms")
    print(f"   • Số bước: {summary.get('total_steps', len(result['intermediate_steps']))}")
    print(f"   • Phương pháp threshold: {summary.get('threshold_method', 'N/A')}")
    print(f"   • Phương pháp loại nền: {summary.get('background_method', 'N/A')}")
    print()
    
    # In metrics
    metrics = result['metrics']
    print("📈 ĐÁNH GIÁ CHẤT LƯỢNG:")
    print(f"   • PSNR: {metrics['psnr']:.2f} dB")
    print(f"   • SSIM: {metrics['ssim']:.4f}")
    print(f"   • MSE:  {metrics['mse']:.2f}")
    print()
    
    # Đánh giá kết quả
    if metrics['psnr'] > 30:
        quality = "Tốt ✅"
    elif metrics['psnr'] > 25:
        quality = "Khá 👍"
    else:
        quality = "Cần cải thiện 🔧"
    
    print(f"   Đánh giá: {quality}")
    print()
    
    # Tạo ảnh so sánh
    print("🖼️  TẠO ẢNH SO SÁNH...")
    comparison = np.hstack([
        result['original_gray'], 
        result['final_image']
    ])
    
    # Thêm text
    h, w = comparison.shape
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    cv2.putText(comparison_rgb, "ORIGINAL", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison_rgb, "PROCESSED", (w//2 + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    comparison_path = f'{output_dir}/comparison.png'
    cv2.imwrite(comparison_path, comparison_rgb)
    print(f"   ✓ Lưu ảnh so sánh: {comparison_path}")
    print()
    
    print("=" * 70)
    print("✅ HOÀN THÀNH!")
    print(f"📂 Kết quả được lưu tại: {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Đã hủy bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
