# Backend - DocCleaner AI

Python Flask API for document image processing using OpenCV and Tesseract OCR.

## 📁 Cấu trúc thư mục

```
Backend/
├── api/
│   └── app.py                   # Flask application (main API)
├── utils/
│   ├── image_processing.py      # Pipeline V2 implementation
│   ├── ocr_engine.py           # Tesseract OCR wrapper
│   └── config.py               # Configuration & presets
├── uploads/                     # Temporary upload folder
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Setup

### 1. Cài đặt dependencies

```bash
cd Backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cài đặt Tesseract OCR

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # Vietnamese language pack
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-vie
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Run Development Server

```bash
python api/app.py
```

Server chạy tại: `http://localhost:5000`

## 📡 API Endpoints

### 1. Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "running",
  "version": "1.0.0",
  "service": "DocCleaner AI Backend",
  "timestamp": "2025-11-29T..."
}
```

---

### 2. Process Image
```http
POST /api/process
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/png;base64,...",
  "settings": {
    "backgroundRemoval": "auto",
    "backgroundKernel": 15,
    "contrastMethod": "clahe_masked",
    "thresholdMethod": "otsu",
    "kernelOpening": 2,
    "kernelClosing": 3
  }
}
```

**Response:**
```json
{
  "success": true,
  "processedImage": "data:image/png;base64,...",
  "intermediateSteps": {
    "grayscale": "data:image/png;base64,...",
    "bgRemoved": "...",
    "enhanced": "...",
    "binary": "...",
    "cleaned": "..."
  },
  "stats": {
    "time": 123.45,
    "width": 1920,
    "height": 1080,
    "steps": 6
  }
}
```

---

### 3. OCR Extraction
```http
POST /api/ocr
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/png;base64,...",
  "lang": "vie"
}
```

**Response:**
```json
{
  "success": true,
  "text": "Extracted text from image...",
  "confidence": 85.6,
  "time": 456.78
}
```

---

### 4. Evaluate Quality
```http
POST /api/evaluate
Content-Type: application/json
```

**Request Body:**
```json
{
  "original": "data:image/png;base64,...",
  "processed": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "psnr": 25.34,
    "ssim": 0.8567,
    "mse": 123.45
  }
}
```

---

### 5. Get Default Config
```http
GET /api/config
```

**Response:**
```json
{
  "success": true,
  "config": {
    "backgroundRemoval": "auto",
    "backgroundKernel": 15,
    ...
  }
}
```

---

### 6. Get Presets
```http
GET /api/config/presets
```

**Response:**
```json
{
  "success": true,
  "presets": {
    "default": {
      "name": "Pipeline V2 - Default",
      "description": "...",
      "config": {...}
    },
    "heavy_stains": {...},
    "broken_strokes": {...},
    "faded_text": {...},
    "low_noise": {...}
  }
}
```

## 🧪 Testing

```bash
# Test health check
curl http://localhost:5000/

# Test với sample image
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

## 🔧 Configuration

Edit `.env` file:

```env
PORT=5000
DEBUG=True
MAX_FILE_SIZE=16777216  # 16MB
TESSERACT_PATH=/usr/local/bin/tesseract
```

## 📦 Pipeline V2 Features

### Background Removal (Fixed)
- **Blackhat**: Loại vết tối trên nền sáng (coffee stains)
- **Tophat**: Làm nổi text trên nền tối
- **Auto**: Kết hợp cả hai (recommended)
- Kernel: 15×15 (V2 - increased from 9×9)

### Contrast Enhancement
- **CLAHE Masked**: Apply CLAHE only to text regions (best for stain removal)
- **CLAHE**: Global adaptive histogram equalization
- **Histogram EQ**: Standard equalization

### Threshold
- **Otsu**: Automatic threshold (recommended)
- **Adaptive Mean**: Local thresholding with mean
- **Adaptive Gaussian**: Local thresholding with Gaussian

### Morphological Operations
- **Opening**: Erosion → Dilation (noise removal)
- **Closing**: Dilation → Erosion (connect strokes)

## 🎯 Presets

1. **Default**: Tài liệu scan thông thường
2. **Heavy Stains**: Vết bẩn nặng (kernel 21×21, clip 3.0)
3. **Broken Strokes**: Nét chữ đứt gãy (closing 5×5)
4. **Faded Text**: Chữ mờ nhạt (clip 3.5)
5. **Low Noise**: Ảnh sạch ít nhiễu (minimal processing)

## 📊 Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error

## 🚀 Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app

# Using Docker
docker build -t doccleaner-backend .
docker run -p 5000:5000 doccleaner-backend
```

## 📝 TODO

- [ ] Add batch processing endpoint
- [ ] Implement caching for processed images
- [ ] Add rate limiting
- [ ] Integrate with cloud storage (S3/GCS)
- [ ] Add WebSocket for real-time processing
- [ ] Implement PDF processing
