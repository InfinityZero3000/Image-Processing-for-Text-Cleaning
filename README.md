# Image Processing for Text Cleaning

Hệ thống xử lý ảnh tài liệu chuyên nghiệp với Pipeline V2 - Morphological Operations

## 📁 Cấu trúc Project

```
Image-Processing-for-Text-Cleaning/
├── Frontend/                          # React Application
│   ├── src/
│   │   ├── components/               # React Components
│   │   │   ├── Header.jsx
│   │   │   ├── UploadArea.jsx
│   │   │   ├── ImageViewer.jsx
│   │   │   └── SettingsPanel.jsx
│   │   ├── utils/
│   │   │   └── imageProcessing.js   # Canvas API processing
│   │   └── DocumentCleanerApp.jsx   # Main App
│   ├── public/
│   │   └── image/
│   ├── package.json
│   └── README.md
│
├── Backend/                           # Python Flask API
│   ├── api/
│   │   └── app.py                    # Flask application
│   ├── utils/
│   │   ├── image_processing.py      # Pipeline V2 (OpenCV)
│   │   ├── ocr_engine.py            # Tesseract OCR
│   │   └── config.py                # Configuration
│   ├── requirements.txt
│   └── README.md
│
├── Image_Processing_Implementation.ipynb  # Jupyter Notebook (Research)
├── SRS_Document_Image_Processing.md      # Requirements Document
└── README.md                              # This file
```

## 🎯 Tính năng chính

### Frontend (React + Canvas API)
- ✅ Upload/Camera interface
- ✅ Real-time image processing
- ✅ 4 tabs: Kết quả, Các bước, So sánh, OCR
- ✅ Interactive settings panel
- ✅ 6-step pipeline visualization
- ✅ Processing statistics
- ✅ Download processed images
- ✅ Responsive design

### Backend (Python Flask + OpenCV)
- ✅ RESTful API
- ✅ Pipeline V2 implementation
- ✅ Background removal (Fixed - 15×15 kernel)
- ✅ CLAHE Masked contrast enhancement
- ✅ Tesseract OCR integration
- ✅ Quality metrics (PSNR, SSIM, MSE)
- ✅ 5 configuration presets
- ✅ CORS enabled

## 🚀 Quick Start

### 1. Setup Backend

```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Cài đặt Tesseract OCR
# macOS: brew install tesseract tesseract-lang
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-vie

# Run server
python api/app.py
```

Server chạy tại: `http://localhost:5000`

### 2. Setup Frontend

```bash
cd Frontend
npm install

# Development
npm run dev

# Production build
npm run build
```

## 🧪 Pipeline V2 - Fixed

### 6 bước xử lý:

1. **Grayscale** - Chuyển sang thang xám
2. **Background Removal** (V2 Fixed) - Loại vết bẩn
   - Kernel: 15×15 (tăng từ 9×9)
   - Methods: Auto/Blackhat/Tophat
3. **Contrast Enhancement** - CLAHE Masked
   - Apply only to text regions
   - Avoid enhancing stains
4. **Threshold** - Otsu/Adaptive
   - Binary image
5. **Opening** (2×2) - Loại nhiễu nhỏ
   - Erosion → Dilation
6. **Closing** (3×3) - Nối nét chữ
   - Dilation → Erosion

### Bug Fix (V2)

**Problem**: Vết bẩn bị làm đậm thay vì mờ đi

**Root Cause**:
```python
# OLD (Wrong):
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
result = cv2.add(gray, tophat)  # ❌ Doubles bright stains
```

**Solution**:
```python
# V2 (Fixed):
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
result = cv2.subtract(gray, blackhat)  # ✅ Removes stains
result = np.clip(result + 10, 0, 255)  # Brightness compensation
```

## 📊 Metrics

- **PSNR**: +3-5 dB improvement
- **SSIM**: 0.75 → 0.85+
- **Stains**: 80% lighter
- **Processing Time**: ~100-300ms (depends on image size)

## 🎨 Configuration Presets

1. **Default**: Tài liệu scan thông thường
2. **Heavy Stains**: Vết bẩn nặng (kernel 21×21)
3. **Broken Strokes**: Nét chữ đứt gãy (closing 5×5)
4. **Faded Text**: Chữ mờ nhạt (clip limit 3.5)
5. **Low Noise**: Ảnh sạch (minimal processing)

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/process` | Process image with Pipeline V2 |
| POST | `/api/ocr` | Extract text (Tesseract) |
| POST | `/api/evaluate` | Calculate quality metrics |
| GET | `/api/config` | Get default config |
| GET | `/api/config/presets` | Get all presets |

## 🔧 Tech Stack

### Frontend
- React 18
- Lucide React (icons)
- Canvas API (image processing)
- TailwindCSS (styling)
- Vite (bundler)

### Backend
- Python 3.9+
- Flask (API framework)
- OpenCV (image processing)
- Tesseract OCR
- scikit-image (metrics)
- NumPy, Pillow

## 📚 Documentation

- [Frontend README](Frontend/README.md) - Component details
- [Backend README](Backend/README.md) - API documentation
- [SRS Document](SRS_Document_Image_Processing.md) - Requirements (FR1-FR11)
- [Jupyter Notebook](Image_Processing_Implementation.ipynb) - Research & experiments

## 🧪 Testing

### Backend
```bash
cd Backend
python -m pytest tests/
```

### Frontend
```bash
cd Frontend
npm test
```

## 📦 Deployment

### Backend (Docker)
```bash
cd Backend
docker build -t doccleaner-backend .
docker run -p 5000:5000 doccleaner-backend
```

### Frontend (Vercel/Netlify)
```bash
cd Frontend
npm run build
# Deploy dist/ folder
```

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License

## 👥 Authors

- Nguyễn Hữu Thắng - Initial work

## 🙏 Acknowledgments

- OpenCV documentation
- Tesseract OCR project
- React community
- Flask community
