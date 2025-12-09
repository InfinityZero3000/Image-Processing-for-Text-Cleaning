import React from 'react';
import { Settings, Sun, Eraser, Link2, Contrast, ScanText, Download, Wand2, Trash2, Info } from 'lucide-react';

const SettingsPanel = ({ 
  settings, 
  setSettings, 
  isProcessing, 
  onOCR, 
  onDownload, 
  onReset,
  hasProcessedImage 
}) => {
  const themeColor = "bg-[#800020]";
  const themeRing = "focus:ring-[#800020]";
  
  return (
    <div className="space-y-4">
      {/* Card 1: Cấu hình xử lý */}
      <div className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden">
        <div className={`${themeColor} px-4 py-3 flex items-center justify-between`}>
          <h3 className="font-bold text-white flex items-center">
            <Settings size={20} className="mr-2" />
            Cấu hình Pipeline V2
          </h3>
          <span className="text-xs bg-white/20 px-2 py-1 rounded text-white">6 bước</span>
        </div>
        
        <div className="p-5 space-y-6">
          {/* FR4: Background Removal (Task: Loại bỏ nền và vết bẩn) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold text-slate-800 flex items-center">
                <Eraser size={16} className="mr-2 text-orange-500" />
                4. Loại bỏ nền và vết bẩn
              </label>
              <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full font-semibold">
                {settings.backgroundRemoval}
              </span>
            </div>
            
            <select 
              value={settings.backgroundRemoval}
              onChange={(e) => setSettings({...settings, backgroundRemoval: e.target.value})}
              className={`w-full p-2.5 text-sm border-2 border-slate-200 rounded-lg focus:outline-none focus:ring-2 ${themeRing} bg-white`}
            >
              <option value="blackhat">Blackhat (Nền sáng có vết đen) ⭐</option>
              <option value="tophat">Tophat (Nền tối)</option>
              <option value="auto">Auto (Hybrid)</option>
              <option value="none">Không xử lý</option>
            </select>
            <p className="text-xs text-slate-500">
              <strong>Task:</strong> Nếu nền sáng có vết đen → dùng black-hat. Nếu nền tối → dùng top-hat
            </p>
            
            <div className="flex items-center justify-between text-xs text-slate-600">
              <span>Kernel Size: {settings.backgroundKernel}×{settings.backgroundKernel}</span>
              <input 
                type="range" 
                min="9" max="21" step="2"
                value={settings.backgroundKernel}
                onChange={(e) => setSettings({...settings, backgroundKernel: Number(e.target.value)})}
                className="w-32 accent-[#800020]"
              />
            </div>
          </div>

          <div className="border-t border-slate-200"></div>

          {/* FR5: Contrast Enhancement (Task: Tăng cường hiển thị) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold text-slate-800 flex items-center">
                <Contrast size={16} className="mr-2 text-blue-500" />
                5. Tăng cường hiển thị (Optional)
              </label>
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-semibold">
                {settings.contrastMethod}
              </span>
            </div>
            
            <select 
              value={settings.contrastMethod}
              onChange={(e) => setSettings({...settings, contrastMethod: e.target.value})}
              className={`w-full p-2.5 text-sm border-2 border-slate-200 rounded-lg focus:outline-none focus:ring-2 ${themeRing}`}
            >
              <option value="clahe_masked">CLAHE Masked (Text Only)</option>
              <option value="clahe">CLAHE (Global)</option>
              <option value="histogram_eq">Histogram Equalization</option>
              <option value="none">Không xử lý</option>
            </select>
          </div>

          <div className="border-t border-slate-200"></div>

          {/* FR1: Threshold (Task: Tiền xử lý) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold text-slate-800 flex items-center">
                <Sun size={16} className="mr-2 text-yellow-500" />
                1. Tiền xử lý (Threshold)
              </label>
              <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full font-semibold">
                {settings.thresholdMethod}
              </span>
            </div>
            
            <div className="grid grid-cols-3 gap-2">
              <button 
                onClick={() => setSettings({...settings, thresholdMethod: 'otsu'})}
                className={`text-xs px-3 py-2 rounded-lg border-2 font-semibold transition-all ${
                  settings.thresholdMethod === 'otsu' 
                    ? `${themeColor} text-white border-transparent` 
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
                }`}
              >
                Otsu
              </button>
              <button 
                onClick={() => setSettings({...settings, thresholdMethod: 'adaptive_mean'})}
                className={`text-xs px-3 py-2 rounded-lg border-2 font-semibold transition-all ${
                  settings.thresholdMethod === 'adaptive_mean' 
                    ? `${themeColor} text-white border-transparent` 
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
                }`}
              >
                Adaptive
              </button>
              <button 
                onClick={() => setSettings({...settings, thresholdMethod: 'adaptive_gaussian'})}
                className={`text-xs px-3 py-2 rounded-lg border-2 font-semibold transition-all ${
                  settings.thresholdMethod === 'adaptive_gaussian' 
                    ? `${themeColor} text-white border-transparent` 
                    : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
                }`}
              >
                Gaussian
              </button>
            </div>
          </div>

          <div className="border-t border-slate-200"></div>

          {/* FR2: Opening (Task: Làm sạch nhiễu) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold text-slate-800 flex items-center">
                <Eraser size={16} className="mr-2 text-red-500" />
                2. Làm sạch nhiễu (Opening)
              </label>
              <span className="text-xs text-slate-500 font-semibold">{settings.kernelOpening}×{settings.kernelOpening}</span>
            </div>
            
            <input 
              type="range" 
              min="2" max="5" step="1"
              value={settings.kernelOpening}
              onChange={(e) => setSettings({...settings, kernelOpening: Number(e.target.value)})}
              className={`w-full h-3 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-[#800020]`}
            />
            <p className="text-xs text-slate-500">Loại bỏ các đốm nhỏ và nhiễu trên ảnh</p>
          </div>

          <div className="border-t border-slate-200"></div>

          {/* FR3: Closing (Task: Làm liền nét chữ) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-bold text-slate-800 flex items-center">
                <Link2 size={16} className="mr-2 text-green-500" />
                3. Làm liền nét chữ (Closing)
              </label>
              <span className="text-xs text-slate-500 font-semibold">{settings.kernelClosing}×{settings.kernelClosing}</span>
            </div>
            
            <input 
              type="range" 
              min="2" max="7" step="1"
              value={settings.kernelClosing}
              onChange={(e) => setSettings({...settings, kernelClosing: Number(e.target.value)})}
              className={`w-full h-3 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-[#800020]`}
            />
            <p className="text-xs text-slate-500">Lấp đầy khoảng trống, nối các nét chữ đứt gãy</p>
          </div>
        </div>
      </div>

      {/* Card 2: Actions */}
      <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-4 space-y-3">
        <button 
          onClick={onOCR}
          disabled={isProcessing || !hasProcessedImage}
          className={`w-full py-3.5 px-4 ${themeColor} hover:bg-red-900 text-white rounded-xl font-bold shadow-lg active:scale-95 transition-all flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {isProcessing ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Đang chạy OCR...</span>
            </>
          ) : (
            <>
              <ScanText size={20} />
              <span>Trích xuất văn bản (OCR)</span>
            </>
          )}
        </button>
        
        {isProcessing && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-xs text-blue-900 flex items-center">
              <span className="inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2"></span>
              Tesseract.js đang tải language data và nhận diện văn bản...
            </p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <button 
            onClick={onDownload}
            disabled={!hasProcessedImage}
            className="flex items-center justify-center py-2.5 px-3 border-2 border-slate-300 rounded-xl text-slate-700 hover:bg-slate-50 hover:border-slate-400 font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download size={16} className="mr-2" />
            Tải xuống
          </button>
          <button 
            onClick={onReset}
            className="flex items-center justify-center py-2.5 px-3 border-2 border-red-300 rounded-xl text-red-700 hover:bg-red-50 hover:border-red-400 font-semibold text-sm transition-all"
          >
            <Trash2 size={16} className="mr-2" />
            Xóa hết
          </button>
        </div>
      </div>
      
      {/* Info Box */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl p-4">
        <div className="flex items-start space-x-3">
          <Info size={20} className="text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="space-y-1">
            <p className="text-sm font-bold text-blue-900">Theo đúng Task Requirements</p>
            <p className="text-xs text-blue-800 leading-relaxed">
              <strong>Thứ tự:</strong> Tiền xử lý (Grayscale + Threshold) → Làm sạch nhiễu (Opening) → Làm liền nét chữ (Closing) → Loại bỏ nền (Black-hat/Top-hat) → Tăng cường hiển thị → Đánh giá kết quả
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
