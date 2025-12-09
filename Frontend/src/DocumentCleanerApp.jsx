import React, { useState, useRef, useEffect } from 'react';
import Tesseract from 'tesseract.js';

// Import Components
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import ImageViewer from './components/ImageViewer';
import SettingsPanel from './components/SettingsPanel';

// Import Utility Functions
import {
  applyGrayscale,
  applyBackgroundRemoval,
  applyContrastEnhancement,
  applyThreshold,
  applyMorphologicalOpening,
  applyMorphologicalClosing,
  applyDilation,
  applyGaussianBlur
} from './utils/imageProcessing';

const DocumentCleanerApp = () => {
  // State quản lý ảnh và trạng thái
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [intermediateSteps, setIntermediateSteps] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('process'); // 'process', 'steps', 'compare', or 'ocr'
  const [extractedText, setExtractedText] = useState("");
  const [processingStats, setProcessingStats] = useState(null);
  const canvasRef = useRef(null);

    // State cấu hình xử lý ảnh - AGGRESSIVE CLEANING (như ảnh mẫu)
  const [settings, setSettings] = useState({
    // Threshold - OTSU với tương phản cực cao
    thresholdMethod: 'otsu', // 'otsu', 'adaptive_mean', 'sauvola'
    sauvolaK: 0.5,
    
    // Median Filter - giảm nhiễu mạnh
    medianKernel: 5, // 5x5 - MẠH HƠN
    
    // Bilateral Filter - làm mịn nền giữ cạnh
    bilateralD: 9,
    bilateralSigmaColor: 75,
    bilateralSigmaSpace: 75,
    
    // Opening - làm sạch nhiễu
    kernelOpening: 3, // 3x3
    
    // Closing - nối nét chữ
    kernelClosing: 3, // 3x3
    
    // Background Removal
    backgroundRemoval: 'none',
    backgroundKernel: 15,
    
    // Contrast Enhancement - BẬT MẠNH
    contrastMethod: 'clahe',
    claheClipLimit: 4.0, // TĂNG LÊN 4.0 - rất mạnh
    claheTileGrid: 8,
  });  // Xử lý ảnh thực tế với Canvas API (Simplified Computer Vision)
  const processImage = async () => {
    if (!originalImage) return;
    
    setIsProcessing(true);
    const steps = {};
    
    try {
      // Tạo canvas để xử lý
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      
      // Load ảnh gốc
      const img = new Image();
      img.crossOrigin = "anonymous";
      
      await new Promise((resolve) => {
        img.onload = resolve;
        img.src = originalImage;
      });
      
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      const startTime = performance.now();
      
      // === PIPELINE TỐI ƯU CHO CHỮ VIẾT TAY ===
      
      // BƯỚC 1: Grayscale
      let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      imageData = applyGrayscale(imageData);
      ctx.putImageData(imageData, 0, 0);
      steps['1_grayscale'] = canvas.toDataURL();
      
      // BƯỚC 2: Gaussian Blur - Giảm nhiễu nhẹ nhàng
      if (settings.gaussianKernel >= 3) {
        imageData = applyGaussianBlur(imageData, settings.gaussianKernel, settings.gaussianSigma);
        ctx.putImageData(imageData, 0, 0);
        steps['2_blurred'] = canvas.toDataURL();
      }
      
      // BƯỚC 3: Sauvola Threshold - Adaptive binarization
      imageData = applyThreshold(
        imageData, 
        settings.thresholdMethod, 
        settings.sauvolaK, 
        settings.niblackK,
        settings.windowSize
      );
      ctx.putImageData(imageData, 0, 0);
      steps['3_threshold'] = canvas.toDataURL();
      
      // BƯỚC 4: Opening - Làm sạch nhiễu nhỏ (NHẸ)
      if (settings.kernelOpening > 1) {
        imageData = applyMorphologicalOpening(imageData, settings.kernelOpening);
        ctx.putImageData(imageData, 0, 0);
        steps['4_opening'] = canvas.toDataURL();
      }
      
      // BƯỚC 5: Closing - Nối nét chữ gãy (NHẸ)
      if (settings.kernelClosing > 1) {
        imageData = applyMorphologicalClosing(imageData, settings.kernelClosing);
        ctx.putImageData(imageData, 0, 0);
        steps['5_closing'] = canvas.toDataURL();
      }
      
      // BƯỚC 6: Kết quả cuối
      steps['6_final'] = canvas.toDataURL();
      
      const finalImage = canvas.toDataURL();
      const processingTime = performance.now() - startTime;
      
      setProcessedImage(finalImage);
      setIntermediateSteps(steps);
      setProcessingStats({
        time: processingTime.toFixed(2),
        width: canvas.width,
        height: canvas.height,
        steps: Object.keys(steps).length + 1
      });
      
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Lỗi xử lý ảnh: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  // Tự động xử lý khi có ảnh hoặc settings thay đổi
  useEffect(() => {
    if (originalImage) {
      const debounceTimer = setTimeout(() => {
        processImage();
      }, 500); // Debounce 500ms
      return () => clearTimeout(debounceTimer);
    }
  }, [originalImage, settings]);

  // Xử lý upload ảnh
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setOriginalImage(e.target.result);
        setProcessedImage(null);
        setIntermediateSteps(null);
        setExtractedText("");
        setProcessingStats(null);
      };
      reader.readAsDataURL(file);
    }
  };

  // OCR thực tế với Tesseract.js
  const handleOCR = async () => {
    if (!processedImage) {
      alert('Vui lòng xử lý ảnh trước khi chạy OCR');
      return;
    }
    
    setIsProcessing(true);
    setExtractedText(''); // Clear previous text
    
    try {
      // Tạo worker Tesseract
      const worker = await Tesseract.createWorker('vie', 1, {
        logger: (m) => {
          // Log progress
          if (m.status === 'recognizing text') {
            console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
          }
        }
      });

      // Recognize text từ ảnh đã xử lý
      const { data: { text, confidence } } = await Tesseract.recognize(
        processedImage,
        'vie',
        {
          tessjs_create_pdf: '0',
          tessjs_create_hocr: '0'
        }
      );

      // Terminate worker
      await worker.terminate();

      // Format kết quả
      const result = `=== KẾT QUẢ OCR (Tesseract.js) ===

${text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 THÔNG TIN XỬ LÝ:

Pipeline đã áp dụng:
1. Grayscale - Chuyển sang thang xám
2. Threshold (${settings.thresholdMethod}) - Nhị phân hóa
3. Opening (${settings.kernelOpening}×${settings.kernelOpening}) - Làm sạch nhiễu
4. Closing (${settings.kernelClosing}×${settings.kernelClosing}) - Nối nét chữ
5. Background Removal (${settings.backgroundRemoval}) - Loại vết bẩn
6. Contrast Enhancement (${settings.contrastMethod}) - Tăng cường

Thời gian xử lý ảnh: ${processingStats?.time || 0}ms
Kích thước: ${processingStats?.width || 0}×${processingStats?.height || 0}px
Độ tin cậy OCR: ${Math.round(confidence)}%
Ngôn ngữ: Tiếng Việt (vie)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 Lưu ý: 
- Kết quả OCR phụ thuộc vào chất lượng ảnh sau xử lý
- Độ tin cậy cao (>80%) cho thấy văn bản được nhận diện tốt
- Có thể thử điều chỉnh các tham số pipeline để cải thiện kết quả`;

      setExtractedText(result);
      setActiveTab('ocr');
      
    } catch (error) {
      console.error('OCR Error:', error);
      alert('Lỗi OCR: ' + error.message + '\n\nVui lòng kiểm tra kết nối internet để tải language data.');
      setExtractedText(`❌ LỖI OCR

${error.message}

Có thể do:
- Chưa tải được language data (cần internet lần đầu)
- Ảnh không phù hợp cho OCR
- Lỗi hệ thống

Vui lòng thử lại hoặc kiểm tra console để biết thêm chi tiết.`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset toàn bộ
  const handleReset = () => {
    setOriginalImage(null);
    setProcessedImage(null);
    setIntermediateSteps(null);
    setExtractedText("");
    setProcessingStats(null);
    setActiveTab('process');
  };

  // Download ảnh đã xử lý
  const handleDownload = () => {
    if (!processedImage) return;
    
    const link = document.createElement('a');
    link.href = processedImage;
    link.download = `cleaned_document_${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 font-sans text-slate-800">
      {/* Hidden Canvas for Image Processing */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Header Component */}
      <Header onReset={handleReset} />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {!originalImage ? (
          // Upload Area Component
          <UploadArea onFileSelect={handleFileUpload} />
        ) : (
          // Editor Workspace
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Image Viewer (2/3 width) */}
            <div className="lg:col-span-2">
              <ImageViewer
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                processedImage={processedImage}
                originalImage={originalImage}
                isProcessing={isProcessing}
                extractedText={extractedText}
                intermediateSteps={intermediateSteps}
                processingStats={processingStats}
              />
            </div>

            {/* Right Column: Settings Panel (1/3 width) */}
            <div className="lg:col-span-1">
              <SettingsPanel
                settings={settings}
                setSettings={setSettings}
                isProcessing={isProcessing}
                onOCR={handleOCR}
                onDownload={handleDownload}
                onReset={handleReset}
                hasProcessedImage={!!processedImage}
              />
            </div>
          </div>
        )}
      </main>

      {/* Global Styles */}
      <style>{`
        .bg-checkered {
          background-color: #ffffff;
          background-image: linear-gradient(45deg, #f1f5f9 25%, transparent 25%), 
                            linear-gradient(-45deg, #f1f5f9 25%, transparent 25%), 
                            linear-gradient(45deg, transparent 75%, #f1f5f9 75%), 
                            linear-gradient(-45deg, transparent 75%, #f1f5f9 75%);
          background-size: 20px 20px;
          background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }
        
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.5s ease-out;
        }
      `}</style>
    </div>
  );
};

export default DocumentCleanerApp;