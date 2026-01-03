import React, { useRef, useState } from 'react';
import { Camera, Upload, Image as ImageIcon, X } from 'lucide-react';

const UploadArea = ({ onFileSelect }) => {
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const videoRef = useRef(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  
  const themeColor = "bg-[#800020]";
  const themeBorder = "border-[#800020]";
  
  // Mở camera
  const openCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      setStream(mediaStream);
      setShowCamera(true);
      
      // Đợi một chút để video element render
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (err) {
      console.error('Error accessing camera:', err);
      // Fallback: mở file picker nếu camera không khả dụng
      alert('Không thể truy cập camera. Vui lòng chọn ảnh từ thư viện.');
      cameraInputRef.current.click();
    }
  };
  
  // Đóng camera
  const closeCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setShowCamera(false);
  };
  
  // Chụp ảnh từ camera
  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);
      
      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        const event = {
          target: {
            files: [file]
          }
        };
        onFileSelect(event);
        closeCamera();
      }, 'image/jpeg', 0.95);
    }
  };
  
  // Camera Modal
  if (showCamera) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-90 z-50 flex items-center justify-center">
        <div className="relative w-full h-full max-w-4xl max-h-screen p-4">
          <button
            onClick={closeCamera}
            className="absolute top-6 right-6 z-10 bg-white p-2 rounded-full shadow-lg hover:bg-gray-100 transition-colors"
          >
            <X size={24} className="text-gray-800" />
          </button>
          
          <div className="flex flex-col items-center justify-center h-full space-y-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="max-w-full max-h-[70vh] rounded-lg shadow-2xl"
            />
            
            <button
              onClick={capturePhoto}
              className="bg-[#800020] text-white px-8 py-4 rounded-full font-bold text-lg shadow-lg hover:bg-[#600018] transition-all flex items-center space-x-2"
            >
              <Camera size={24} />
              <span>Chụp ảnh</span>
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] space-y-8 animate-fade-in">
      <div className="text-center space-y-3 max-w-2xl px-4">
        <div className="inline-block p-3 bg-red-50 rounded-full mb-4">
          <ImageIcon className="w-12 h-12 text-[#800020]" />
        </div>
        <h2 className="text-3xl font-bold text-[#800020]">
          Xử lý ảnh tài liệu chuyên nghiệp
        </h2>
        <p className="text-slate-600 text-lg">
          Làm sạch vết bẩn, nhiễu và tăng cường chất lượng văn bản bằng thuật toán hình thái học
        </p>
        <div className="flex items-center justify-center space-x-2 text-sm text-slate-500 mt-4">
          <span className="px-3 py-1 bg-slate-100 rounded-full">✓ Loại bỏ nền</span>
          <span className="px-3 py-1 bg-slate-100 rounded-full">✓ Nối nét chữ</span>
          <span className="px-3 py-1 bg-slate-100 rounded-full">✓ Khử nhiễu</span>
          <span className="px-3 py-1 bg-slate-100 rounded-full">✓ OCR</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-2xl px-4">
        {/* Nút Chụp Ảnh */}
        <div 
          onClick={openCamera}
          className={`cursor-pointer group relative overflow-hidden bg-white p-8 rounded-2xl shadow-lg border-2 border-dashed ${themeBorder} hover:bg-red-50 hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1`}
        >
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className={`${themeColor} p-5 rounded-full text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
              <Camera size={36} />
            </div>
            <div className="text-center">
              <span className="font-bold text-lg block text-slate-800">Chụp ảnh mới</span>
              <span className="text-sm text-slate-500">Sử dụng camera thiết bị</span>
            </div>
          </div>
          {/* Hidden file input as fallback */}
          <input 
            type="file" 
            accept="image/*" 
            capture="environment" 
            ref={cameraInputRef}
            className="hidden"
            onChange={onFileSelect}
          />
        </div>

        {/* Nút Tải Ảnh */}
        <div 
          onClick={() => fileInputRef.current.click()}
          className="cursor-pointer group bg-white p-8 rounded-2xl shadow-lg border-2 border-slate-200 hover:border-[#800020] hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1"
        >
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="bg-slate-100 p-5 rounded-full text-slate-600 group-hover:bg-slate-200 group-hover:text-[#800020] transition-all duration-300">
              <Upload size={36} />
            </div>
            <div className="text-center">
              <span className="font-bold text-lg block text-slate-800">Tải ảnh lên</span>
              <span className="text-sm text-slate-500">Từ thư viện hoặc máy tính</span>
            </div>
          </div>
          <input 
            type="file" 
            accept="image/*" 
            ref={fileInputRef}
            className="hidden"
            onChange={onFileSelect}
          />
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 max-w-2xl">
        <p className="text-sm text-blue-900">
          <span className="font-bold"> Lưu ý:</span> Để có kết quả tốt nhất, hãy chụp ảnh trong điều kiện ánh sáng đầy đủ và giữ camera ổn định. Hỗ trợ định dạng JPG, PNG.
        </p>
      </div>
    </div>
  );
};

export default UploadArea;
