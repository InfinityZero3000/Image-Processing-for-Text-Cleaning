# Frontend - DocCleaner AI

Modern React application for document image processing using morphological operations.

## 📁 Cấu trúc thư mục

```
Frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx              # Header với logo và reset button
│   │   ├── UploadArea.jsx          # Upload/Camera interface
│   │   ├── ImageViewer.jsx         # Hiển thị ảnh với tabs (Kết quả, Các bước, So sánh, OCR)
│   │   └── SettingsPanel.jsx       # Panel cấu hình Pipeline V2
│   ├── utils/
│   │   └── imageProcessing.js      # Computer Vision functions
│   └── DocumentCleanerApp.jsx      # Main App component
└── public/
    └── image/                       # Static images

```

## 🎨 Components

### 1. **Header.jsx**
- Sticky header với branding
- Reset button để làm mới toàn bộ state

### 2. **UploadArea.jsx**
- Upload file từ thư viện
- Chụp ảnh trực tiếp (mobile-friendly)
- Info box với hướng dẫn

### 3. **ImageViewer.jsx**
- **Tab "Kết quả"**: Hiển thị ảnh đã xử lý
- **Tab "Các bước"**: Grid view của 6 bước trung gian
- **Tab "So sánh"**: Side-by-side comparison
- **Tab "OCR"**: Textarea với kết quả OCR
- Processing stats: Thời gian, kích thước, số bước

### 4. **SettingsPanel.jsx**
- Cấu hình 6 bước Pipeline V2:
  1. Background Removal (Auto/Blackhat/Tophat)
  2. Contrast Enhancement (CLAHE Masked/CLAHE/Histogram EQ)
  3. Threshold (Otsu/Adaptive Mean/Gaussian)
  4. Opening - Khử nhiễu (kernel 2-5)
  5. Closing - Nối nét chữ (kernel 2-7)
- Action buttons: OCR, Download, Reset
- Info box với tips

## 🛠️ Utilities

### imageProcessing.js
Các hàm xử lý ảnh với Canvas API:

- `applyGrayscale()` - Chuyển RGB sang grayscale
- `applyBackgroundRemoval()` - Loại bỏ nền (Morphological)
- `applyContrastEnhancement()` - Histogram Equalization
- `applyThreshold()` - Otsu's method
- `applyErosion()` / `applyDilation()` - Morphological operations
- `applyMorphologicalOpening()` - Erosion → Dilation
- `applyMorphologicalClosing()` - Dilation → Erosion

## 🚀 Pipeline V2

1. **Grayscale** - Chuyển sang thang xám
2. **Background Removal** (15×15 kernel) - Loại vết bẩn
3. **Contrast Enhancement** (CLAHE Masked) - Tăng độ tương phản vùng text
4. **Threshold** (Otsu) - Nhị phân hóa
5. **Opening** (2×2) - Loại nhiễu nhỏ
6. **Closing** (3×3) - Nối nét chữ đứt gãy

## 🎯 Features

✅ Real-time processing với auto-debounce (500ms)  
✅ Intermediate steps visualization  
✅ Before/After comparison  
✅ Processing statistics  
✅ OCR integration ready (Tesseract.js)  
✅ Download processed images  
✅ Responsive design (Mobile + Desktop)  
✅ Modern UI với TailwindCSS  

## 🎨 Theme

- **Primary Color**: `#800020` (Bordeaux Red)
- **Background**: Gradient slate
- **Checkered pattern** cho canvas background

## 📦 Dependencies

```json
{
  "react": "^18.x",
  "lucide-react": "latest"
}
```

## 🔧 Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev
```

## 📝 TODO

- [ ] Tích hợp Tesseract.js cho OCR thực tế
- [ ] Export intermediate steps as ZIP
- [ ] Batch processing multiple images
- [ ] Save/Load configuration presets
- [ ] Kết nối với Backend API
