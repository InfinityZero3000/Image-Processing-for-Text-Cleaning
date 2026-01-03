import React, { useState, useEffect } from 'react';
import Tesseract from 'tesseract.js';

// Import Components
import Header from './components/Header';
import UploadArea from './components/UploadArea';
import ImageViewer from './components/ImageViewer';
import SettingsPanel from './components/SettingsPanel';

// Backend API URL - Sử dụng environment variable hoặc fallback về localhost
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

const DocumentCleanerApp = () => {
  // State quản lý ảnh và trạng thái
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [intermediateSteps, setIntermediateSteps] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('process'); // 'process', 'steps', 'compare', 'ocr', or 'ocr-compare'
  const [extractedText, setExtractedText] = useState("");
  const [ocrComparison, setOcrComparison] = useState(null); // Kết quả so sánh OCR
  const [processingStats, setProcessingStats] = useState(null);
  const [ocrProvider, setOcrProvider] = useState('tesseract'); // 'tesseract', 'ocrspace', 'google_vision'
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online', 'offline', 'checking'

  // Kiểm tra Backend status
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/config`);
        if (response.ok) {
          setBackendStatus('online');
        } else {
          setBackendStatus('offline');
        }
      } catch (error) {
        setBackendStatus('offline');
      }
    };
    checkBackend();
  }, []);

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
  });  
  const [backendPipeline, setBackendPipeline] = useState('ai'); // 'ai', 'simple', 'premium'
  
  // ========= XỬ LÝ ẢNH VỚI BACKEND API =========
  const processImageBackend = async () => {
    if (!originalImage) return;
    
    setIsProcessing(true);
    const startTime = performance.now();
    
    try {
      // Chuyển base64 thành Blob
      const response = await fetch(originalImage);
      const blob = await response.blob();
      
      // Tạo FormData
      const formData = new FormData();
      formData.append('image', blob, 'image.png');
      
      // Settings cho Backend dựa trên pipeline được chọn
      let backendSettings;
      if (backendPipeline === 'ai') {
        // AI Pipeline - Local Advanced (theo yêu cầu task)
        backendSettings = {
          pipeline: 'ai_local',
          denoiseStrength: 10,
          bgMode: 'auto',  // auto detect: blackhat hoặc tophat
          bgKernel: 25,
          claheClip: 2.0,
          thresholdMethod: 'otsu',
          openingKernel: 2,
          closingKernel: 2,
        };
      } else if (backendPipeline === 'ai_cloud') {
        // AI Pipeline - Cloud (Hugging Face)
        // Available tasks: dewarping, deshadowing, appearance, deblurring, binarization
        backendSettings = {
          pipeline: 'ai_cloud',
          tasks: ['appearance', 'deshadowing', 'binarization'],
        };
      } else if (backendPipeline === 'simple') {
        backendSettings = {
          pipeline: 'simple',
          blurSize: 3,
          thresholdMethod: 'otsu',
          openingKernel: 2,
          closingKernel: 2,
        };
      } else if (backendPipeline === 'premium') {
        // Premium Pipeline - Theo task requirements
        // 1. Grayscale → 2. Threshold → 3. Opening → 4. Closing → 5. Black/Top-hat → 6. CLAHE
        backendSettings = {
          pipeline: 'premium',
          thresholdMethod: 'otsu',  // 'otsu', 'adaptive', 'adaptive_gaussian'
          adaptiveBlock: 31,
          adaptiveC: 10,
          openingKernel: 2,  // Làm sạch nhiễu
          closingKernel: 2,  // Nối nét chữ
          bgMode: 'auto',  // 'auto', 'blackhat', 'tophat', 'none'
          bgKernel: 25,
          contrastMethod: 'clahe',  // 'clahe', 'histogram', 'none'
          claheClip: 2.0,
          claheTileGrid: 8,
        };
      } else {
        backendSettings = {
          pipeline: backendPipeline,
          thresholdMethod: 'otsu',
        };
      }
      
      formData.append('settings', JSON.stringify(backendSettings));
      
      // Gọi Backend API
      const apiResponse = await fetch(`${BACKEND_URL}/api/process`, {
        method: 'POST',
        body: formData,
      });
      
      if (!apiResponse.ok) {
        throw new Error(`Backend error: ${apiResponse.status}`);
      }
      
      const result = await apiResponse.json();
      const processingTime = performance.now() - startTime;
      
      // Xử lý kết quả
      if (result.success) {
        setProcessedImage(`data:image/png;base64,${result.processed_image}`);
        
        // Convert intermediate steps
        if (result.intermediate_steps) {
          const steps = {};
          Object.entries(result.intermediate_steps).forEach(([key, value]) => {
            steps[key] = `data:image/png;base64,${value}`;
          });
          setIntermediateSteps(steps);
        }
        
        setProcessingStats({
          time: processingTime.toFixed(2),
          width: result.width || 0,
          height: result.height || 0,
          steps: Object.keys(result.intermediate_steps || {}).length,
          pipeline: 'Premium V4.0',
          metrics: result.metrics,
        });
        
        // Hiển thị cảnh báo nếu có (VD: fallback từ AI Cloud)
        if (result.warning) {
          alert(`⚠️ ${result.warning}`);
        }
      } else {
        throw new Error(result.error || 'Unknown error');
      }
      
    } catch (error) {
      console.error('Backend processing error:', error);
      alert(`Lỗi xử lý Backend: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };
  

  
  // ========= HÀM XỬ LÝ CHÍNH =========
  const processImage = async () => {
    if (backendStatus === 'online') {
      await processImageBackend();
    } else {
      alert('Backend không hoạt động. Vui lòng khởi động Backend.');
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

  // OCR với nhiều provider
  const handleOCR = async () => {
    if (!processedImage) {
      alert('Vui lòng xử lý ảnh trước khi chạy OCR');
      return;
    }
    
    setIsProcessing(true);
    setExtractedText(''); // Clear previous text
    
    try {
      let result;
      
      if (ocrProvider === 'tesseract') {
        // Tesseract.js - Local OCR
        result = await runTesseractOCR();
      } else {
        // Cloud OCR via Backend API
        result = await runCloudOCR();
      }
      
      setExtractedText(result);
      setActiveTab('ocr');
      
    } catch (error) {
      console.error('OCR Error:', error);
      alert('Lỗi OCR: ' + error.message);
      setExtractedText(`❌ LỖI OCR\n\n${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // So sánh OCR giữa ảnh gốc và ảnh đã xử lý
  const handleCompareOCR = async () => {
    if (!originalImage || !processedImage) {
      alert('Cần có cả ảnh gốc và ảnh đã xử lý để so sánh');
      return;
    }
    
    if (backendStatus !== 'online') {
      alert('Tính năng này yêu cầu Backend đang hoạt động');
      return;
    }
    
    setIsProcessing(true);
    setOcrComparison(null);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/compare-ocr`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          original: originalImage,
          processed: processedImage,
          lang: 'vie',
          provider: ocrProvider === 'tesseract' ? 'tesseract' : ocrProvider
        })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Lỗi khi so sánh OCR');
      }
      
      const result = await response.json();
      
      if (result.success) {
        setOcrComparison(result);
        setActiveTab('ocr-compare');
      } else {
        throw new Error(result.error || 'So sánh OCR thất bại');
      }
      
    } catch (error) {
      console.error('Compare OCR Error:', error);
      alert('Lỗi so sánh OCR: ' + error.message);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Tesseract.js OCR (Local)
  const runTesseractOCR = async () => {
    const worker = await Tesseract.createWorker('vie+eng', 1, {
      logger: (m) => {
        if (m.status === 'recognizing text') {
          console.log(`OCR Progress: ${Math.round(m.progress * 100)}%`);
        }
      }
    });

    // Cấu hình Tesseract.js với PSM mode tốt hơn
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.AUTO,
      preserve_interword_spaces: '1',
    });

    const { data: { text, confidence } } = await Tesseract.recognize(
      processedImage,
      'vie+eng',
      { tessjs_create_pdf: '0', tessjs_create_hocr: '0' }
    );

    await worker.terminate();

    return `=== KẾT QUẢ OCR (Tesseract.js - Local) ===

${text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 THÔNG TIN:
• Pipeline: ${processingStats?.pipeline || 'Local JS'}
• Thời gian xử lý ảnh: ${processingStats?.time || 0}ms
• Kích thước: ${processingStats?.width || 0}×${processingStats?.height || 0}px
• Độ tin cậy OCR: ${Math.round(confidence)}%
• Ngôn ngữ: Tiếng Việt + English (vie+eng)`;
  };
  
  // Cloud OCR via Backend
  const runCloudOCR = async () => {
    if (backendStatus !== 'online') {
      throw new Error('Backend không hoạt động. Vui lòng dùng Tesseract (Local).');
    }
    
    // Chuyển base64 thành Blob
    const response = await fetch(processedImage);
    const blob = await response.blob();
    
    const formData = new FormData();
    formData.append('image', blob, 'image.png');
    formData.append('provider', ocrProvider);
    formData.append('language', 'vie');
    
    const apiResponse = await fetch(`${BACKEND_URL}/api/ocr`, {
      method: 'POST',
      body: formData,
    });
    
    if (!apiResponse.ok) {
      const error = await apiResponse.json();
      throw new Error(error.error || `Backend error: ${apiResponse.status}`);
    }
    
    const result = await apiResponse.json();
    
    if (!result.success) {
      throw new Error(result.error || 'OCR failed');
    }
    
    const providerName = {
      'ocrspace': 'OCR.space',
      'google_vision': 'Google Cloud Vision',
      'easyocr': 'EasyOCR',
      'vietocr': 'VietOCR',
    }[ocrProvider] || ocrProvider;
    
    return `=== KẾT QUẢ OCR (${providerName} - Cloud) ===

${result.text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THÔNG TIN:
• Provider: ${providerName}
• Pipeline: ${processingStats?.pipeline || 'Unknown'}
• Thời gian xử lý ảnh: ${processingStats?.time || 0}ms
• Kích thước: ${processingStats?.width || 0}×${processingStats?.height || 0}px
• Ngôn ngữ: Tiếng Việt`;
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
      {/* Header Component */}
      <Header onReset={handleReset} />
      
      {/* Backend Status Bar */}
      <div className="max-w-7xl mx-auto px-4 py-2">
        <div className="flex items-center justify-between bg-white rounded-lg shadow-sm p-3">
          {/* Backend Status */}
          <div className="flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${
              backendStatus === 'online' ? 'bg-green-500' : 
              backendStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
            }`}></span>
            <span className="text-sm text-slate-600">
              Backend: {backendStatus === 'online' ? 'Đang hoạt động' : 
                       backendStatus === 'offline' ? 'Offline' : 'Đang kiểm tra...'}
            </span>
          </div>
          
          {/* Backend Pipeline Selection */}
          {backendStatus === 'online' && (
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-600">Pipeline:</label>
              <select
                value={backendPipeline}
                onChange={(e) => setBackendPipeline(e.target.value)}
                className="text-sm px-2 py-1.5 rounded-lg border border-slate-200 bg-white focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ai">AI Local (Khuyến nghị)</option>
                <option value="ai_cloud">AI Cloud (Hugging Face)</option>
                <option value="simple">Simple</option>
                <option value="premium">Premium</option>
              </select>
            </div>
          )}
          
          {/* OCR Provider */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-600">OCR:</label>
            <select
              value={ocrProvider}
              onChange={(e) => setOcrProvider(e.target.value)}
              className="text-sm px-2 py-1.5 rounded-lg border border-slate-200 bg-white focus:ring-2 focus:ring-indigo-500"
            >
              <option value="tesseract">Tesseract (Local)</option>
              <option value="ocrspace">OCR.space (Cloud)</option>
            </select>
          </div>
        </div>
      </div>

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
                ocrComparison={ocrComparison}
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
                onCompareOCR={handleCompareOCR}
                onDownload={handleDownload}
                onReset={handleReset}
                hasProcessedImage={!!processedImage}
                backendStatus={backendStatus}
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