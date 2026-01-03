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

### Option 1: Automated Script (Recommended ⭐)

**For Mac/Linux:**
```bash
# Make scripts executable
chmod +x start.sh stop.sh

# Start all services
./start.sh

# Stop all services
./stop.sh
```

**For Windows:**
```cmd
# Start all services
start.bat

# Stop all services
stop.bat
```

📖 **Windows users:** See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed guide

### Option 2: Manual Setup

#### 1. Setup Backend

```bash
cd Backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Cài đặt Tesseract OCR
# macOS: brew install tesseract tesseract-lang
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-vie
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Run server
python app.py  # Changed from api/app.py
```

Server chạy tại: `http://localhost:5001`

#### 2. Setup Frontend

```bash
cd Frontend
npm install

# Development
npm run dev

# Production build
npm run build
```

Frontend chạy tại: `http://localhost:3000`

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

### Deploy lên Vercel (Recommended ⭐)

#### Prerequisites
1. Cài đặt Vercel CLI (nếu chưa có):
```bash
npm install -g vercel
```

2. Đăng nhập Vercel:
```bash
vercel login
```

#### Deploy
1. **Tại root folder của project, chạy:**
```bash
vercel
```

2. **Làm theo hướng dẫn:**
   - Set up and deploy? → Yes
   - Which scope? → Chọn account của bạn
   - Link to existing project? → No (lần đầu)
   - What's your project's name? → image-processing-text-cleaning (hoặc tên bạn muốn)
   - In which directory is your code located? → `./` (để mặc định)

3. **Deployment settings (quan trọng):**
   - Build Command: `cd Frontend && npm install && npm run build`
   - Output Directory: `Frontend/dist`
   - Install Command: `npm install`

4. **Environment Variables (cần thiết cho production):**
```bash
vercel env add PYTHONPATH
# Nhập: Backend

vercel env add DEBUG
# Nhập: False
```

#### Deploy Production
```bash
vercel --prod
```

#### Cấu trúc files đã tạo:
- ✅ `vercel.json` - Cấu hình routing và builds
- ✅ `api/index.py` - Entry point cho Backend API
- ✅ `.vercelignore` - Files không deploy

#### URLs sau khi deploy:
- **Frontend**: `https://your-project.vercel.app`
- **Backend API**: `https://your-project.vercel.app/api/*`

#### ⚠️ Lưu ý với Vercel:
1. **Serverless Functions Limits:**
   - Execution time: 10s (Hobby), 60s (Pro)
   - Memory: 1024MB (Hobby), 3008MB (Pro)
   - OpenCV và image processing có thể chậm → Cân nhắc dùng Vercel Pro hoặc deploy Backend riêng

2. **Large Dependencies:**
   - OpenCV, NumPy khá nặng
   - Có thể bị timeout với ảnh lớn
   - Giải pháp: Deploy Backend lên Railway/Render, Frontend lên Vercel

### Alternative: Deploy Backend riêng

#### Backend (Railway/Render)
```bash
cd Backend
# Tạo Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy lên Railway hoặc Render
# Cập nhật BACKEND_URL trong Frontend
```

#### Frontend (Vercel)
```bash
cd Frontend
# Cập nhật API URL trong code
# Deploy chỉ Frontend
vercel --prod
```

### Docker Deployment (Self-hosted)

#### Backend
```bash
cd Backend
docker build -t doccleaner-backend .
docker run -p 5000:5000 doccleaner-backend
```

#### Frontend
```bash
cd Frontend
docker build -t doccleaner-frontend .
docker run -p 3000:3000 doccleaner-frontend
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
