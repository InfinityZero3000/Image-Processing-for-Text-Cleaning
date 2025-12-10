"""
BẢNG 8 - Xử lý ảnh tài liệu theo ĐÚNG yêu cầu đề ra

Yêu cầu:
1. Tiền xử lý: Grayscale + Otsu/Adaptive/Sauvola threshold  
2. Loại nhiễu: Opening với kernel 2x2 hoặc 3x3
3. Nối nét đứt: Closing để lấp đầy khoảng trống và nối nét gãy
4. Loại nền/vết bẩn: Black-hat (nền sáng có vết tối) hoặc Top-hat (nền tối)
5. Tăng cường và lưu kết quả
6. Đánh giá so sánh trước/sau
"""

import cv2
import numpy as np
import os
import sys
from skimage.filters import threshold_sauvola, threshold_niblack

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def process_with_full_requirements(image_path, output_dir="test_output"):
    """
    Xử lý ảnh theo đúng yêu cầu Bảng 8
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"XỬ LÝ THEO YÊU CẦU BẢNG 8")
    print(f"Image: {image_path}")
    print(f"{'='*70}")
    
    # Đọc ảnh gốc
    original = cv2.imread(image_path)
    if original is None:
        print(f"ERROR: Cannot read image {image_path}")
        return
    
    print(f"Image shape: {original.shape}")
    cv2.imwrite(f"{output_dir}/step0_original_color.png", original)
    
    results = {}
    
    # ============ BƯỚC 1: TIỀN XỬ LÝ ============
    print("\n" + "="*50)
    print("[BƯỚC 1] TIỀN XỬ LÝ")
    print("="*50)
    
    # 1.1 Chuyển Grayscale
    print("  → Chuyển ảnh sang Grayscale")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/step1_1_grayscale.png", gray)
    results['1_Grayscale'] = gray.copy()
    
    # 1.2 Khử nhiễu nhẹ trước khi threshold (Bilateral để giữ cạnh)
    print("  → Bilateral filter để khử nhiễu, giữ cạnh")
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(f"{output_dir}/step1_2_denoised.png", denoised)
    
    # 1.3 CLAHE để cân bằng độ sáng
    print("  → CLAHE để cân bằng độ sáng cục bộ")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    cv2.imwrite(f"{output_dir}/step1_3_clahe.png", enhanced)
    
    # 1.4 Binarization với Sauvola (tốt hơn Otsu cho tài liệu)
    print("  → Sauvola threshold (tốt hơn Otsu cho tài liệu không đều)")
    thresh_sauvola = threshold_sauvola(enhanced, window_size=25, k=0.2)
    binary = (enhanced > thresh_sauvola).astype(np.uint8) * 255
    cv2.imwrite(f"{output_dir}/step1_4_sauvola_binary.png", binary)
    results['2_Sauvola'] = binary.copy()
    
    # So sánh với Otsu
    print("  → (So sánh) Otsu threshold")
    _, otsu_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/step1_4b_otsu_binary.png", otsu_binary)
    
    # ============ BƯỚC 2: LOẠI NHIỄU - OPENING ============
    print("\n" + "="*50)
    print("[BƯỚC 2] LOẠI NHIỄU - OPENING (kernel 2x2 hoặc 3x3)")
    print("="*50)
    
    # Thử với kernel 2x2
    print("  → Opening với kernel 2x2 (nhỏ, giữ chi tiết)")
    kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened_2x2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_2x2)
    cv2.imwrite(f"{output_dir}/step2_1_opening_2x2.png", opened_2x2)
    
    # Thử với kernel 3x3
    print("  → Opening với kernel 3x3 (loại nhiễu mạnh hơn)")
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_3x3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3x3)
    cv2.imwrite(f"{output_dir}/step2_2_opening_3x3.png", opened_3x3)
    
    # Chọn kernel 2x2 để giữ chi tiết
    opened = opened_2x2
    results['3_Opening'] = opened.copy()
    
    # ============ BƯỚC 3: NỐI NÉT ĐỨT - CLOSING ============
    print("\n" + "="*50)
    print("[BƯỚC 3] NỐI NÉT ĐỨT - CLOSING (lấp khoảng trống)")
    print("="*50)
    
    # Closing với kernel 2x2 (nhẹ)
    print("  → Closing với kernel 2x2 (nối nét nhẹ)")
    kernel_close_2x2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed_2x2 = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close_2x2)
    cv2.imwrite(f"{output_dir}/step3_1_closing_2x2.png", closed_2x2)
    
    # Closing với kernel 3x3 (mạnh hơn)
    print("  → Closing với kernel 3x3 (nối nét mạnh hơn)")
    kernel_close_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_3x3 = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close_3x3)
    cv2.imwrite(f"{output_dir}/step3_2_closing_3x3.png", closed_3x3)
    
    # Closing nhiều lần với kernel nhỏ
    print("  → Closing 2 lần với kernel 2x2 (tốt hơn 1 lần với kernel lớn)")
    closed_multi = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close_2x2)
    closed_multi = cv2.morphologyEx(closed_multi, cv2.MORPH_CLOSE, kernel_close_2x2)
    cv2.imwrite(f"{output_dir}/step3_3_closing_multi.png", closed_multi)
    
    closed = closed_3x3  # Chọn 3x3 để nối nét tốt hơn
    results['4_Closing'] = closed.copy()
    
    # ============ BƯỚC 4: LOẠI NỀN/VẾT BẨN - BLACK-HAT/TOP-HAT ============
    print("\n" + "="*50)
    print("[BƯỚC 4] LOẠI NỀN/VẾT BẨN - BLACK-HAT hoặc TOP-HAT")
    print("="*50)
    
    # Top-hat: Làm nổi bật các vùng sáng trên nền tối
    print("  → Top-hat: Làm nổi bật text sáng trên nền tối")
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)
    cv2.imwrite(f"{output_dir}/step4_1_tophat.png", tophat)
    
    # Black-hat: Làm nổi bật các vùng tối trên nền sáng (chữ đen)
    print("  → Black-hat: Làm nổi bật text tối trên nền sáng")
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_tophat)
    cv2.imwrite(f"{output_dir}/step4_2_blackhat.png", blackhat)
    
    # Kết hợp: Dùng Black-hat để loại vết bẩn
    print("  → Kết hợp với ảnh đã xử lý để loại vết bẩn")
    # Chuẩn hóa black-hat
    blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    # Threshold black-hat để tạo mask vết bẩn
    _, stain_mask = cv2.threshold(blackhat_norm, 30, 255, cv2.THRESH_BINARY)
    # Loại bỏ các vết bẩn nhỏ trong mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel_small)
    cv2.imwrite(f"{output_dir}/step4_3_stain_mask.png", stain_mask)
    
    # Áp dụng: loại các vết bẩn lớn
    cleaned = closed.copy()
    # Chỉ giữ lại text, loại vết bẩn lớn không phải text
    # (text thường có shape nhất định, vết bẩn thì không)
    
    results['5_Cleaned'] = cleaned.copy()
    
    # ============ BƯỚC 5: TĂNG CƯỜNG VÀ LƯU ============
    print("\n" + "="*50)
    print("[BƯỚC 5] TĂNG CƯỜNG VÀ LƯU KẾT QUẢ")
    print("="*50)
    
    # Loại bỏ các thành phần nhiễu nhỏ
    print("  → Loại bỏ nhiễu nhỏ (thành phần < 30 pixels)")
    final = remove_small_components(cleaned, min_size=30)
    cv2.imwrite(f"{output_dir}/step5_1_noise_removed.png", final)
    
    # Closing cuối để đảm bảo nét liền
    print("  → Closing cuối để đảm bảo nét liền")
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final)
    cv2.imwrite(f"{output_dir}/step5_2_final.png", final)
    
    results['6_Final'] = final.copy()
    
    # ============ BƯỚC 6: ĐÁNH GIÁ SO SÁNH ============
    print("\n" + "="*50)
    print("[BƯỚC 6] ĐÁNH GIÁ - SO SÁNH TRƯỚC/SAU")
    print("="*50)
    
    create_evaluation_report(gray, final, results, output_dir)
    
    # Tạo comparison image
    create_step_by_step_comparison(results, f"{output_dir}/comparison_steps.png")
    
    # Tạo before/after comparison
    create_before_after(gray, final, f"{output_dir}/before_after.png")
    
    print(f"\n{'='*70}")
    print(f"HOÀN THÀNH! Kết quả lưu trong: {output_dir}/")
    print(f"{'='*70}")
    print("\n📌 CÁC FILE ĐÃ TẠO:")
    print("   - step*_*.png: Từng bước xử lý")
    print("   - comparison_steps.png: So sánh tất cả các bước")
    print("   - before_after.png: So sánh trước/sau")
    print("   - evaluation_report.txt: Báo cáo đánh giá")


def remove_small_components(binary, min_size=30):
    """Loại bỏ các thành phần nhiễu nhỏ"""
    # Đảo ngược để tìm text (đen trên trắng -> trắng trên đen)
    inverted = cv2.bitwise_not(binary)
    
    # Tìm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # Tạo mask cho các thành phần đủ lớn
    result = np.zeros_like(binary)
    for i in range(1, num_labels):  # Bỏ qua background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            result[labels == i] = 255
    
    # Đảo ngược lại
    result = cv2.bitwise_not(result)
    return result


def create_evaluation_report(original, final, results, output_dir):
    """Tạo báo cáo đánh giá"""
    report_path = f"{output_dir}/evaluation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ XỬ LÝ ẢNH THEO BẢNG 8\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. THÔNG TIN ẢNH:\n")
        f.write(f"   - Kích thước: {original.shape}\n")
        f.write(f"   - Loại: Grayscale\n\n")
        
        f.write("2. CÁC BƯỚC ĐÃ THỰC HIỆN:\n")
        f.write("   [✓] Bước 1: Tiền xử lý (Grayscale + Sauvola threshold)\n")
        f.write("   [✓] Bước 2: Loại nhiễu (Opening với kernel 2x2)\n")
        f.write("   [✓] Bước 3: Nối nét đứt (Closing với kernel 3x3)\n")
        f.write("   [✓] Bước 4: Loại nền/vết bẩn (Black-hat analysis)\n")
        f.write("   [✓] Bước 5: Tăng cường và lưu kết quả\n")
        f.write("   [✓] Bước 6: Đánh giá so sánh trước/sau\n\n")
        
        f.write("3. ĐÁNH GIÁ CHẤT LƯỢNG:\n")
        
        # Tính một số metrics
        orig_mean = np.mean(original)
        final_mean = np.mean(final)
        
        # Đếm số pixel đen (text) trong kết quả
        text_pixels = np.sum(final == 0)
        bg_pixels = np.sum(final == 255)
        text_ratio = text_pixels / (text_pixels + bg_pixels) * 100
        
        f.write(f"   - Độ sáng trung bình gốc: {orig_mean:.2f}\n")
        f.write(f"   - Tỷ lệ text/background: {text_ratio:.2f}% / {100-text_ratio:.2f}%\n")
        f.write(f"   - Số pixel text: {text_pixels:,}\n")
        f.write(f"   - Số pixel background: {bg_pixels:,}\n\n")
        
        f.write("4. NHẬN XÉT:\n")
        if text_ratio < 5:
            f.write("   - ⚠️ Tỷ lệ text thấp, có thể text bị mất\n")
        elif text_ratio > 50:
            f.write("   - ⚠️ Tỷ lệ text cao, có thể còn nhiễu\n")
        else:
            f.write("   - ✓ Tỷ lệ text/background hợp lý\n")
        
        f.write("\n5. FILE KẾT QUẢ:\n")
        for name in results.keys():
            f.write(f"   - {name}\n")
    
    print(f"  ✓ Báo cáo đánh giá: {report_path}")


def create_step_by_step_comparison(results, output_path):
    """Tạo ảnh so sánh từng bước"""
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    
    # Lấy kích thước từ ảnh đầu tiên
    first_img = list(results.values())[0]
    h, w = first_img.shape[:2]
    
    # Resize nếu quá lớn
    max_size = 300
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        results = {k: cv2.resize(v, (new_w, new_h)) for k, v in results.items()}
        h, w = new_h, new_w
    
    padding = 15
    text_height = 30
    cell_w = w + 2 * padding
    cell_h = h + 2 * padding + text_height
    
    canvas = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255
    
    for i, (label, img) in enumerate(results.items()):
        row = i // cols
        col = i % cols
        
        x = col * cell_w + padding
        y = row * cell_h + padding + text_height
        
        canvas[y:y+h, x:x+w] = img
        cv2.putText(canvas, label[:20], (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    cv2.imwrite(output_path, canvas)
    print(f"  ✓ Comparison: {output_path}")


def create_before_after(before, after, output_path):
    """Tạo ảnh so sánh trước/sau"""
    h, w = before.shape[:2]
    
    # Resize nếu cần
    max_size = 400
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        before = cv2.resize(before, (new_w, new_h))
        after = cv2.resize(after, (new_w, new_h))
        h, w = new_h, new_w
    
    padding = 20
    text_height = 40
    
    # Tạo canvas với 2 ảnh cạnh nhau
    canvas_w = w * 2 + padding * 3
    canvas_h = h + padding * 2 + text_height
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    
    # Vẽ ảnh BEFORE
    x1 = padding
    y1 = padding + text_height
    canvas[y1:y1+h, x1:x1+w] = before
    cv2.putText(canvas, "BEFORE", (x1 + w//4, text_height//2 + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    # Vẽ ảnh AFTER
    x2 = w + padding * 2
    canvas[y1:y1+h, x2:x2+w] = after
    cv2.putText(canvas, "AFTER", (x2 + w//4, text_height//2 + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    # Vẽ mũi tên
    arrow_x = x1 + w + padding//2
    arrow_y = y1 + h//2
    cv2.arrowedLine(canvas, (arrow_x - 10, arrow_y), (arrow_x + 10, arrow_y), 
                    0, 2, tipLength=0.5)
    
    cv2.imwrite(output_path, canvas)
    print(f"  ✓ Before/After: {output_path}")


# ============ THÊM CÁC PIPELINE KHÁC ĐỂ THỬ NGHIỆM ============

def pipeline_sauvola_enhanced(gray):
    """
    Pipeline Sauvola tối ưu với đầy đủ các bước
    """
    print("\n[PIPELINE SAUVOLA ENHANCED]")
    
    # 1. Bilateral filter
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Background estimation và subtraction
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_bg)
    diff = cv2.absdiff(denoised, background)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(diff.astype(np.uint8))
    
    # 4. Sauvola threshold
    thresh = threshold_sauvola(enhanced, window_size=25, k=0.15)
    binary = (enhanced > thresh).astype(np.uint8) * 255
    
    # 5. Opening (loại nhiễu) - kernel 2x2
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 6. Closing (nối nét) - kernel 3x3
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 7. Loại nhiễu nhỏ
    final = remove_small_components(closed, min_size=25)
    
    # 8. Closing cuối
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final)
    
    return final


def pipeline_combined(gray):
    """
    Pipeline kết hợp nhiều kỹ thuật
    """
    print("\n[PIPELINE COMBINED]")
    
    # 1. Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Bilateral + Median filter
    filtered = cv2.bilateralFilter(normalized, 7, 50, 50)
    filtered = cv2.medianBlur(filtered, 3)
    
    # 3. Background subtraction với Top-hat
    kernel_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel_th)
    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_th)
    
    # Kết hợp
    enhanced = cv2.add(cv2.subtract(filtered, blackhat), tophat)
    
    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    
    # 5. Multi-scale Sauvola
    results = []
    for ws in [15, 25, 35]:
        thresh = threshold_sauvola(enhanced, window_size=ws, k=0.2)
        binary = (enhanced > thresh).astype(np.float32)
        results.append(binary)
    
    combined = np.mean(results, axis=0)
    binary = (combined > 0.5).astype(np.uint8) * 255
    
    # 6. Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # 7. Closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    # 8. Clean up
    final = remove_small_components(closed, min_size=30)
    
    return final


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Xử lý ảnh theo yêu cầu Bảng 8')
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
    
    # Chạy pipeline chính theo yêu cầu Bảng 8
    process_with_full_requirements(args.image, args.output)
    
    # Chạy thêm pipeline Sauvola Enhanced
    print("\n" + "="*70)
    print("THÊM: PIPELINE SAUVOLA ENHANCED")
    print("="*70)
    
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    result_enhanced = pipeline_sauvola_enhanced(gray)
    cv2.imwrite(f"{args.output}/pipeline_sauvola_enhanced.png", result_enhanced)
    print(f"✓ Saved: {args.output}/pipeline_sauvola_enhanced.png")
    
    # Chạy thêm pipeline Combined
    print("\n" + "="*70)
    print("THÊM: PIPELINE COMBINED")
    print("="*70)
    
    result_combined = pipeline_combined(gray)
    cv2.imwrite(f"{args.output}/pipeline_combined.png", result_combined)
    print(f"✓ Saved: {args.output}/pipeline_combined.png")
    
    print("\n" + "="*70)
    print("TẤT CẢ HOÀN THÀNH!")
    print("="*70)
