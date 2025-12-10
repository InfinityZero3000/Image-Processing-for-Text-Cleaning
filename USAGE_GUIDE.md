# Hướng Dẫn Xử Lý Ảnh Văn Bản - Version 2.1

## 🎯 CẢI TIẾN MỚI (v2.1 - 09/12/2025)

### ✅ Sửa lỗi làm gãy chữ thêm

**Vấn đề cũ:**
- Opening được thực hiện TRƯỚC Closing → làm gãy chữ trước khi nối
- Kernel quá lớn phá vỡ cấu trúc chữ

**Giải pháp:**
```
Trước: [Threshold] → [Opening ❌] → [Closing] 
Sau:   [Threshold] → [Closing ✅] → [Opening] → [BG Removal] → [Closing lại]
```

- ✅ Đổi thứ tự: Closing TRƯỚC, Opening SAU
- ✅ Dùng ELLIPSE kernel thay vì RECT (mềm mại hơn)
- ✅ Kernel nhỏ (2×2) không phá chữ
- ✅ Closing lại sau loại nền để đảm bảo chữ liền

### ✅ OCR được tích hợp đầy đủ

- Frontend: Tesseract.js (client-side)
- Backend: Tesseract OCR API (mạnh hơn, nhanh hơn)
- Hỗ trợ tiếng Việt và nhiều ngôn ngữ khác

---

## Tổng Quan

Hệ thống xử lý ảnh văn bản theo đúng yêu cầu task với các bước sau:

### Pipeline Xử Lý V2.1 (Cải tiến)

1. **Tiền xử lý:**
   - Chuyển ảnh sang thang xám (Grayscale)
   - Dùng ngưỡng (Otsu hoặc Adaptive Threshold) để nhị phân ảnh

2. **Làm liền nét chữ:** ⭐ **THỰC HIỆN TRƯỚC**
   - Dùng phép đóng (Closing) để lấp khoảng trống, nối các đoạn chữ đứt gãy
   - Kernel nhỏ (2×2), kiểu ELLIPSE

3. **Làm sạch nhiễu:** ⭐ **THỰC HIỆN SAU**
   - Sử dụng phép mở (Opening) để loại bỏ các điểm trắng nhỏ (nhiễu)
   - Kernel nhỏ (2×2), kiểu ELLIPSE

4. **Loại bỏ nền và vết bẩn:**
   - Dùng Black-hat hoặc Top-hat tùy loại nền:
     - Nếu nền tối → dùng Top-hat
     - Nếu nền sáng có vết đen → dùng Black-hat

5. **Tăng cường hiển thị và lưu kết quả**

6. **Đánh giá chất lượng (PSNR, SSIM, MSE)**

## Cài Đặt

### 1. Cài đặt các thư viện Python

```bash
cd Backend
pip install -r requirements.txt
```

### 2. Cài đặt Tesseract OCR (nếu cần OCR)

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-vie

# Windows
# Download từ: https://github.com/UB-Mannheim/tesseract/wiki
```

## Sử Dụng

### 1. Xử Lý Ảnh Đơn Lẻ (Single Image)

#### Qua Web Interface

```bash
# Khởi động backend
cd Backend
python app.py

# Khởi động frontend (terminal khác)
cd Frontend
npm install
npm run dev
```

Truy cập: `http://localhost:5173`

#### Qua API

```python
import requests
import base64

# Đọc ảnh
with open('image.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

# Cấu hình xử lý
settings = {
    'thresholdMethod': 'otsu',       # hoặc 'adaptive_mean', 'adaptive_gaussian'
    'kernelOpening': 2,               # Kernel cho opening (2x2)
    'kernelClosing': 3,               # Kernel cho closing (3x3)
    'backgroundRemoval': 'auto',      # 'auto', 'blackhat', 'tophat', 'none'
    'backgroundKernel': 15,           # Kernel cho background removal
    'contrastMethod': 'none'          # 'none', 'clahe', 'clahe_masked'
}

# Gửi request
response = requests.post('http://localhost:5001/api/process', json={
    'image': f'data:image/jpeg;base64,{img_data}',
    'settings': settings
})

result = response.json()
print(f"PSNR: {result['evaluation']['psnr']}")
print(f"SSIM: {result['evaluation']['ssim']}")
```

### 2. Xử Lý Hàng Loạt (Batch Processing)

#### Xử lý thư mục ảnh

```bash
cd Backend

# Xử lý với cấu hình mặc định
python batch_processor.py /path/to/images

# Xử lý với cấu hình tùy chỉnh
python batch_processor.py /path/to/images \
    --output-dir logs/my_experiment \
    --threshold otsu \
    --bg-removal auto
```

**Tham số:**
- `input_dir`: Thư mục chứa ảnh đầu vào (bắt buộc)
- `--output-dir`: Thư mục lưu kết quả (mặc định: `logs/batch_experiments`)
- `--threshold`: Phương pháp threshold - `otsu`, `adaptive_mean`, `adaptive_gaussian` (mặc định: `otsu`)
- `--bg-removal`: Phương pháp loại nền - `auto`, `blackhat`, `tophat`, `none` (mặc định: `auto`)

#### Kết quả

Sau khi xử lý, hệ thống tạo:

1. **Thư mục `processed/`**: Chứa tất cả ảnh đã xử lý
2. **Thư mục `comparisons/`**: Chứa ảnh so sánh trước/sau
3. **File `results_YYYYMMDD_HHMMSS.json`**: Kết quả chi tiết dạng JSON
4. **File `results_YYYYMMDD_HHMMSS.csv`**: Bảng kết quả dạng CSV
5. **File `report_YYYYMMDD_HHMMSS.html`**: Báo cáo HTML đầy đủ

### 3. Xử Lý Batch qua API

```python
import requests
import base64
import os

# Đọc nhiều ảnh
images = []
for filename in os.listdir('/path/to/images'):
    if filename.endswith(('.jpg', '.png')):
        with open(os.path.join('/path/to/images', filename), 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
            images.append(f'data:image/jpeg;base64,{img_data}')

# Gửi batch request
response = requests.post('http://localhost:5001/api/batch/process', json={
    'images': images,
    'settings': {
        'thresholdMethod': 'otsu',
        'kernelOpening': 2,
        'kernelClosing': 3,
        'backgroundRemoval': 'auto',
        'backgroundKernel': 15
    }
})

result = response.json()
print(f"Tổng số: {result['total']}")
print(f"Thành công: {result['successful']}")
print(f"Thất bại: {result['failed']}")
print(f"PSNR trung bình: {result['statistics']['psnr']['mean']:.2f}")
print(f"SSIM trung bình: {result['statistics']['ssim']['mean']:.4f}")
```

## Các Phương Pháp Xử Lý

### 1. Threshold Methods

- **`otsu`**: Tự động tìm ngưỡng tối ưu (Otsu's method)
  - Tốt cho ảnh có phân bố histogram rõ ràng
  - Nhanh và ổn định

- **`adaptive_mean`**: Ngưỡng thích ứng dựa trên trung bình
  - Tốt cho ảnh có độ sáng không đồng đều
  - Xử lý tốt với ánh sáng thay đổi

- **`adaptive_gaussian`**: Ngưỡng thích ứng dựa trên Gaussian
  - Tốt nhất cho ảnh có nhiễu và độ sáng không đều
  - Chậm hơn nhưng chất lượng cao

### 2. Background Removal Methods

- **`auto`**: Tự động chọn phương pháp phù hợp
  - Phân tích độ sáng trung bình
  - Nền sáng (>127) → Black-hat
  - Nền tối (≤127) → Top-hat

- **`blackhat`**: Loại vết đen trên nền sáng
  - Phát hiện và loại bỏ nhiễu pepper (điểm đen)
  - Phù hợp với ảnh scan có vết mực, bụi bẩn

- **`tophat`**: Loại nền tối, làm nổi text sáng
  - Trích xuất vùng sáng (text) từ nền tối
  - Phù hợp với ảnh chụp có nền tối

- **`none`**: Không loại nền
  - Chỉ áp dụng threshold và morphology

### 3. Kernel Sizes

- **Opening Kernel (2-5)**: Càng lớn càng loại nhiễu mạnh, nhưng có thể mất chi tiết
  - 2×2: Loại nhiễu nhỏ, giữ chi tiết
  - 3×3: Cân bằng
  - 5×5: Loại nhiễu mạnh

- **Closing Kernel (2-5)**: Càng lớn càng nối chữ tốt, nhưng có thể làm dính chữ
  - 2×2: Nối đứt gãy nhỏ
  - 3×3: Cân bằng (khuyến nghị)
  - 5×5: Nối mạnh

- **Background Kernel (10-30)**: Càng lớn càng loại nền lớn
  - 15×15: Mặc định, cân bằng
  - 20-30: Loại nền lớn, vết bẩn lớn

## Đánh Giá Kết Quả

### Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Đo tỷ lệ tín hiệu/nhiễu
   - Càng cao càng tốt (>30 dB là tốt)

2. **SSIM (Structural Similarity Index)**
   - Đo độ tương đồng cấu trúc
   - Giá trị 0-1, càng gần 1 càng giống

3. **MSE (Mean Squared Error)**
   - Đo lỗi trung bình bình phương
   - Càng thấp càng tốt

### Ví dụ Đọc Báo Cáo

```python
import json

# Đọc kết quả JSON
with open('logs/batch_experiments/results_20231205_143022.json', 'r') as f:
    data = json.load(f)

# Xem thống kê
stats = data['statistics']
print(f"PSNR - Mean: {stats['psnr']['mean']:.2f}, Std: {stats['psnr']['std']:.2f}")
print(f"SSIM - Mean: {stats['ssim']['mean']:.4f}, Std: {stats['ssim']['std']:.4f}")

# Xem từng ảnh
for result in data['detailed_results']:
    print(f"{result['filename']}: PSNR={result['metrics']['psnr']:.2f}")
```

## Ví Dụ Thực Nghiệm

### Thí nghiệm 1: So sánh các phương pháp threshold

```bash
# Otsu
python batch_processor.py data/test_images \
    --output-dir experiments/exp1_otsu \
    --threshold otsu

# Adaptive Mean
python batch_processor.py data/test_images \
    --output-dir experiments/exp1_adaptive_mean \
    --threshold adaptive_mean

# Adaptive Gaussian
python batch_processor.py data/test_images \
    --output-dir experiments/exp1_adaptive_gaussian \
    --threshold adaptive_gaussian
```

### Thí nghiệm 2: So sánh các phương pháp loại nền

```bash
# Black-hat
python batch_processor.py data/test_images \
    --output-dir experiments/exp2_blackhat \
    --bg-removal blackhat

# Top-hat
python batch_processor.py data/test_images \
    --output-dir experiments/exp2_tophat \
    --bg-removal tophat

# Auto
python batch_processor.py data/test_images \
    --output-dir experiments/exp2_auto \
    --bg-removal auto
```

## Troubleshooting

### Lỗi thường gặp

1. **"No module named cv2"**
   ```bash
   pip install opencv-python
   ```

2. **"No module named skimage"**
   ```bash
   pip install scikit-image
   ```

3. **OCR không hoạt động**
   - Kiểm tra Tesseract đã cài đặt: `tesseract --version`
   - Cài đặt ngôn ngữ: `brew install tesseract-lang` (macOS)

4. **Backend không chạy**
   - Kiểm tra port 5001 đã được sử dụng chưa
   - Thay đổi port: `PORT=5002 python app.py`

## API Endpoints

### POST /api/process
Xử lý một ảnh đơn lẻ

### POST /api/batch/process
Xử lý nhiều ảnh cùng lúc

### POST /api/ocr
Trích xuất text từ ảnh

### POST /api/evaluate
Đánh giá chất lượng ảnh

## Tài Liệu Tham Khảo

- OpenCV Morphological Operations: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- Otsu Threshold: https://en.wikipedia.org/wiki/Otsu%27s_method
- SSIM: https://en.wikipedia.org/wiki/Structural_similarity
- Image Quality Metrics: https://scikit-image.org/docs/stable/api/skimage.metrics.html

## License

MIT License
