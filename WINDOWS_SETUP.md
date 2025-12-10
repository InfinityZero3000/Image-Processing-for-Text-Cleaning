# DocCleaner AI - Quick Start Guide for Windows

## 📋 Yêu cầu hệ thống

- **Node.js** (v16 hoặc mới hơn) - [Tải tại đây](https://nodejs.org/)
- **Python** (v3.8 hoặc mới hơn) - [Tải tại đây](https://www.python.org/downloads/)
- **Git** (tùy chọn) - [Tải tại đây](https://git-scm.com/)

## 🚀 Khởi động nhanh

### Cách 1: Sử dụng script tự động (Khuyến nghị)

1. **Khởi động ứng dụng:**
   ```cmd
   start.bat
   ```
   Script sẽ tự động:
   - Kiểm tra và cài đặt dependencies cho Frontend và Backend
   - Khởi động Backend trên port 5001
   - Khởi động Frontend trên port 3000
   - Tạo log files trong thư mục `logs/`

2. **Truy cập ứng dụng:**
   - Mở trình duyệt và truy cập: http://localhost:3000

3. **Dừng ứng dụng:**
   ```cmd
   stop.bat
   ```

### Cách 2: Khởi động thủ công

#### Backend:
```cmd
cd Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

#### Frontend (cửa sổ mới):
```cmd
cd Frontend
npm install
npm run dev
```

## 📂 Cấu trúc thư mục

```
Image-Processing-for-Text-Cleaning/
├── start.bat              # Script khởi động cho Windows
├── stop.bat               # Script dừng cho Windows  
├── start.sh               # Script khởi động cho Mac/Linux
├── stop.sh                # Script dừng cho Mac/Linux
├── Backend/
│   ├── app.py            # API server chính
│   ├── requirements.txt  # Python dependencies
│   └── utils/
│       └── image_processing.py  # Core xử lý ảnh
├── Frontend/
│   ├── package.json      # Node.js dependencies
│   └── src/              # Source code React
└── logs/                 # Log files (tự động tạo)
```

## 🔧 Xử lý sự cố

### Lỗi: "python không được nhận diện"
- Đảm bảo Python đã được cài đặt và thêm vào PATH
- Thử dùng `py` thay vì `python`:
  ```cmd
  py app.py
  ```

### Lỗi: "npm không được nhận diện"
- Đảm bảo Node.js đã được cài đặt và thêm vào PATH
- Khởi động lại Command Prompt sau khi cài đặt Node.js

### Port 3000 hoặc 5001 đã được sử dụng
```cmd
# Kiểm tra process đang dùng port
netstat -ano | findstr :3000
netstat -ano | findstr :5001

# Kill process theo PID
taskkill /F /PID <PID>
```

### Backend không khởi động
- Kiểm tra Python version: `python --version` (cần >= 3.8)
- Kiểm tra log file trong thư mục `logs/`
- Thử cài lại dependencies:
  ```cmd
  cd Backend
  rmdir /s /q venv
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```

### Frontend không khởi động
- Kiểm tra Node.js version: `node --version` (cần >= 16)
- Kiểm tra log file trong thư mục `logs/`
- Thử cài lại dependencies:
  ```cmd
  cd Frontend
  rmdir /s /q node_modules
  npm install
  ```

## 📝 Xem logs

```cmd
# Xem log Backend
type logs\backend_YYYYMMDD_HHMMSS.log

# Xem log Frontend
type logs\frontend_YYYYMMDD_HHMMSS.log

# Xem log real-time (PowerShell)
Get-Content logs\backend_YYYYMMDD_HHMMSS.log -Wait
```

## 🎯 Sử dụng ứng dụng

1. **Upload ảnh:** Kéo thả hoặc click để chọn ảnh
2. **Cấu hình xử lý:** Điều chỉnh các tham số trong Settings Panel
3. **Xử lý:** Click "Process Image" để làm sạch ảnh
4. **Tải xuống:** Click "Download" để lưu kết quả

## 🔗 Links hữu ích

- Frontend: http://localhost:3000
- Backend API: http://localhost:5001
- API Documentation: http://localhost:5001/docs (nếu có Swagger)

## 💡 Tips

- Sử dụng `start.bat` để khởi động nhanh mà không cần cài đặt thủ công
- Log files được lưu tự động với timestamp để dễ debug
- Services chạy trong cửa sổ riêng (minimized) để không làm lộn xộn desktop
- Dùng `stop.bat` để dừng tất cả services một cách an toàn

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra log files trong thư mục `logs/`
2. Đảm bảo đã cài đặt đủ requirements
3. Kiểm tra ports 3000 và 5001 không bị chiếm dụng
