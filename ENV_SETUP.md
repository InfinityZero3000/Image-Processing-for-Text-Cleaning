# Hướng Dẫn Cấu Hình Environment Variables

File `.env` đã được tạo để lưu trữ các API keys và cấu hình môi trường.

## 📁 Cấu trúc files

- **`.env`** - File chính chứa API keys (KHÔNG commit lên Git)
- **`.env.example`** - File mẫu để tham khảo (có thể commit)

## 🔑 API Keys Cần Thiết

### 1. OCR.space API (Miễn phí)

**Bước 1:** Đăng ký miễn phí tại https://ocr.space/ocrapi

**Bước 2:** Nhận API key qua email

**Bước 3:** Thêm vào file `.env`:
```bash
OCRSPACE_API_KEY=K87654321088957
```

**Free tier:**
- 25,000 requests/tháng
- Hỗ trợ tiếng Việt
- Không cần credit card

---

### 2. Google Cloud Vision API (Cần credit card)

**Bước 1:** Tạo project tại https://console.cloud.google.com/

**Bước 2:** Bật Cloud Vision API:
- Vào [API Library](https://console.cloud.google.com/apis/library)
- Tìm "Cloud Vision API"
- Click "Enable"

**Bước 3:** Tạo Service Account:
- Vào [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
- Click "Create Service Account"
- Tên: `document-ocr`
- Role: `Cloud Vision AI Service Agent`

**Bước 4:** Tạo JSON key:
- Click vào service account vừa tạo
- Tab "Keys" > "Add Key" > "Create new key"
- Chọn JSON
- Download file (ví dụ: `my-project-123456-abc123.json`)

**Bước 5:** Lưu file JSON vào thư mục an toàn:
```bash
# Tạo thư mục credentials
mkdir -p ~/credentials

# Copy file JSON vào đây
cp ~/Downloads/my-project-123456-abc123.json ~/credentials/google-vision.json

# Set permissions
chmod 600 ~/credentials/google-vision.json
```

**Bước 6:** Thêm đường dẫn vào `.env`:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/Users/yourname/credentials/google-vision.json
```

**Pricing:**
- 1,000 requests đầu tiên MIỄN PHÍ mỗi tháng
- Sau đó: $1.50/1,000 requests
- Cần credit card để verify

---

## 🚀 Cách Sử Dụng

### 1. Copy file `.env.example` thành `.env`:
```bash
cp .env.example .env
```

### 2. Mở file `.env` và điền API keys:
```bash
nano .env
# hoặc
code .env
```

### 3. Restart Backend:
```bash
./stop.sh
./start.sh
```

---

## ✅ Kiểm Tra Cấu Hình

### Test OCR.space:
```bash
cd Backend
source ../.venv/bin/activate
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('OCR.space Key:', os.getenv('OCRSPACE_API_KEY')[:10] + '...' if os.getenv('OCRSPACE_API_KEY') else 'NOT SET')
"
```

### Test Google Vision:
```bash
cd Backend
source ../.venv/bin/activate
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
print('Google Credentials:', 'SET' if path and os.path.exists(path) else 'NOT SET or FILE NOT FOUND')
"
```

---

## 🎯 OCR Providers Trong App

Sau khi cấu hình, bạn có thể chọn OCR provider trong Frontend:

1. **Tesseract (Local)** - Không cần API key, miễn phí, offline
2. **OCR.space (Cloud)** - Cần `OCRSPACE_API_KEY`
3. **Google Vision (Cloud)** - Cần `GOOGLE_APPLICATION_CREDENTIALS`

---

## ⚠️ Lưu Ý Bảo Mật

- ❌ **KHÔNG** commit file `.env` lên Git
- ❌ **KHÔNG** share API keys công khai
- ✅ File `.env` đã được thêm vào `.gitignore`
- ✅ Chỉ commit `.env.example` (không chứa keys thật)

---

## 🐛 Troubleshooting

### Lỗi: "OCRSPACE_API_KEY is not set"
→ Chưa cấu hình `.env` hoặc API key sai

**Fix:**
```bash
# Kiểm tra file .env tồn tại
ls -la .env

# Xem nội dung (che key)
cat .env | grep OCRSPACE_API_KEY
```

### Lỗi: "Google credentials not found"
→ Đường dẫn file JSON sai hoặc file không tồn tại

**Fix:**
```bash
# Kiểm tra file tồn tại
test -f "$GOOGLE_APPLICATION_CREDENTIALS" && echo "OK" || echo "NOT FOUND"

# Kiểm tra permissions
ls -l "$GOOGLE_APPLICATION_CREDENTIALS"
```

### Backend không load .env
→ Chưa cài `python-dotenv`

**Fix:**
```bash
source .venv/bin/activate
pip install python-dotenv
```

---

## 📚 Tài Liệu Tham Khảo

- OCR.space API: https://ocr.space/ocrapi
- Google Cloud Vision: https://cloud.google.com/vision/docs/ocr
- python-dotenv: https://pypi.org/project/python-dotenv/
