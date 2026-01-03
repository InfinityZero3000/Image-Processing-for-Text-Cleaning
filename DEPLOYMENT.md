# Deployment Guide - Separate Frontend & Backend

## Vấn đề với Vercel Full-Stack

❌ **Vercel Serverless Functions có giới hạn:**
- Execution time: 10s (Hobby plan)
- Memory: 1024MB
- Build size: 250MB
- OpenCV + NumPy + dependencies khá nặng → Dễ timeout

✅ **Giải pháp: Deploy riêng**
- Frontend → Vercel (miễn phí, nhanh)
- Backend → Railway/Render (miễn phí, unlimited time)

---

## Option 1: Deploy Backend lên Railway (Recommended ⭐)

### 1. Chuẩn bị Backend

Tạo file `runtime.txt` trong thư mục `Backend/`:
```
python-3.11
```

Tạo file `Procfile` trong thư mục `Backend/`:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### 2. Deploy lên Railway

1. Tạo tài khoản tại https://railway.app
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Chọn repo của bạn
4. **Root Directory**: `Backend`
5. **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
6. **Environment Variables**:
   - `PORT`: 5001 (Railway tự set)
   - `DEBUG`: False
   - `PYTHONPATH`: .

7. Deploy! Railway sẽ tự động build và deploy
8. Lấy URL (ví dụ: `https://your-app.railway.app`)

---

## Option 2: Deploy Backend lên Render

### 1. Chuẩn bị Backend

Tạo file `render.yaml` ở root:
```yaml
services:
  - type: web
    name: doccleaner-backend
    env: python
    region: oregon
    plan: free
    buildCommand: "cd Backend && pip install -r requirements.txt"
    startCommand: "cd Backend && gunicorn app:app --bind 0.0.0.0:$PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
      - key: DEBUG
        value: False
```

### 2. Deploy lên Render

1. Tạo tài khoản tại https://render.com
2. **New** → **Web Service**
3. Connect GitHub repo
4. **Root Directory**: `Backend`
5. **Build Command**: `pip install -r requirements.txt`
6. **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
7. Deploy! Lấy URL (ví dụ: `https://your-app.onrender.com`)

---

## Deploy Frontend lên Vercel

### 1. Cấu hình Backend URL

Trong folder Frontend, tạo file `.env.production`:
```
VITE_BACKEND_URL=https://your-backend.railway.app
```

### 2. Deploy lên Vercel

```bash
cd /path/to/project
vercel
```

Hoặc từ Vercel Dashboard:
1. Import GitHub repo
2. **Root Directory**: `/`
3. **Build Command**: `cd Frontend && npm install && npm run build`
4. **Output Directory**: `Frontend/dist`
5. **Install Command**: `npm install`
6. **Environment Variables**:
   - `VITE_BACKEND_URL`: `https://your-backend.railway.app`

---

## Testing

### Test Backend (Railway/Render):
```bash
curl https://your-backend.railway.app/
# Expected: {"message": "DocCleaner API is running", ...}
```

### Test Frontend (Vercel):
```bash
open https://your-frontend.vercel.app
# Upload ảnh và xử lý
```

---

## Environment Variables Summary

### Backend (Railway/Render):
- `PORT`: Auto-set by platform
- `DEBUG`: False
- `PYTHONPATH`: . (current directory)

### Frontend (Vercel):
- `VITE_BACKEND_URL`: https://your-backend.railway.app

---

## Troubleshooting

### Backend không khởi động
- Check logs: Railway/Render dashboard
- Verify `gunicorn` trong requirements.txt
- Check Python version (3.9-3.11)

### Frontend không kết nối Backend
- Check CORS trong Backend/app.py
- Verify VITE_BACKEND_URL đúng
- Check Network tab trong browser DevTools

### OpenCV errors
- Railway/Render tự cài system dependencies
- Nếu lỗi: thêm `opencv-python-headless` thay vì `opencv-python`

---

## Cost

| Service | Plan | Cost | Limits |
|---------|------|------|--------|
| Railway | Free | $0 | 500 hours/month, 100GB bandwidth |
| Render | Free | $0 | 750 hours/month |
| Vercel | Hobby | $0 | Unlimited bandwidth, 100GB storage |

**Total: $0/month** 🎉

---

## Rollback

### Railway/Render:
- Dashboard → Deployments → Click previous version

### Vercel:
```bash
vercel rollback
```
