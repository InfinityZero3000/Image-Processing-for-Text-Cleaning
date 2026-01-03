import React from 'react';
import { ImageIcon, Type, Layers, Zap, Info } from 'lucide-react';

const ImageViewer = ({ 
  activeTab, 
  setActiveTab, 
  processedImage, 
  originalImage, 
  isProcessing, 
  extractedText,
  ocrComparison,
  intermediateSteps,
  processingStats
}) => {
  const themeColor = "bg-[#800020]";
  const themeRing = "focus:ring-[#800020]";
  
  return (
    <div className="space-y-4">
      {/* Navigation Tabs */}
      <div className="bg-white p-1 rounded-xl shadow-md inline-flex border border-slate-200">
        <button 
          onClick={() => setActiveTab('process')}
          className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-2 ${
            activeTab === 'process' 
              ? `${themeColor} text-white shadow-md` 
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <ImageIcon size={18} />
          <span>Kết quả</span>
        </button>
        <button 
          onClick={() => setActiveTab('steps')}
          className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-2 ${
            activeTab === 'steps' 
              ? `${themeColor} text-white shadow-md` 
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Layers size={18} />
          <span>Các bước</span>
        </button>
        <button 
          onClick={() => setActiveTab('compare')}
          className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-2 ${
            activeTab === 'compare' 
              ? `${themeColor} text-white shadow-md` 
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Zap size={18} />
          <span>So sánh</span>
        </button>
        <button 
          onClick={() => setActiveTab('ocr')}
          className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-2 ${
            activeTab === 'ocr' 
              ? `${themeColor} text-white shadow-md` 
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Type size={18} />
          <span>OCR</span>
        </button>
        <button 
          onClick={() => setActiveTab('ocr-compare')}
          className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-2 ${
            activeTab === 'ocr-compare' 
              ? `${themeColor} text-white shadow-md` 
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          <Zap size={18} />
          <span>So sánh OCR</span>
        </button>
      </div>

      {/* Main Display Area */}
      <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-slate-200">
        {/* Tab: Kết quả xử lý */}
        {activeTab === 'process' && (
          <div className="relative min-h-[500px] flex items-center justify-center bg-checkered p-8">
            <img 
              src={processedImage || originalImage} 
              alt="Document" 
              className={`max-w-full max-h-[70vh] object-contain rounded-lg shadow-2xl transition-all duration-500 ${
                isProcessing ? 'blur-sm grayscale opacity-60' : ''
              }`} 
            />
            {isProcessing && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 backdrop-blur-md z-10">
                <div className="w-16 h-16 border-4 border-[#800020] border-t-transparent rounded-full animate-spin"></div>
                <p className="mt-6 font-bold text-[#800020] text-lg">Đang xử lý Pipeline V2...</p>
                <p className="text-sm text-slate-500 mt-2">Grayscale → CLAHE → Blur → Adaptive Threshold → Opening → Closing → Dilation</p>
              </div>
            )}
            
            {/* Processing Stats Badge */}
            {processingStats && !isProcessing && (
              <div className="absolute top-4 right-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-xl shadow-lg backdrop-blur-md">
                <div className="flex items-center space-x-2">
                  <Zap size={16} />
                  <span className="text-sm font-semibold">{processingStats.time}ms</span>
                </div>
              </div>
            )}
            
            {/* Status Badge */}
            <div className={`absolute bottom-6 left-6 px-4 py-2 rounded-xl text-sm font-semibold backdrop-blur-md shadow-lg ${
              isProcessing 
                ? 'bg-yellow-500/90 text-white' 
                : 'bg-green-500/90 text-white'
            }`}>
              {isProcessing ? "⏳ Đang xử lý..." : "✓ Hoàn thành"}
            </div>
          </div>
        )}

        {/* Tab: Các bước xử lý */}
        {activeTab === 'steps' && (
          <div className="p-6 space-y-4">
            <div className="flex items-center space-x-2 mb-4">
              <Info size={20} className="text-blue-500" />
              <h3 className="font-bold text-lg">Các bước xử lý (Pipeline V2)</h3>
            </div>
            
            {intermediateSteps ? (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-slate-600 uppercase">Gốc</p>
                  <img src={originalImage} alt="Original" className="w-full rounded-lg border-2 border-slate-200 shadow-sm" />
                </div>
                
                {/* Pipeline CẢI TIẾN */}
                {intermediateSteps['1_grayscale'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-slate-600 uppercase">1. Grayscale</p>
                    <img src={intermediateSteps['1_grayscale']} alt="Grayscale" className="w-full rounded-lg border-2 border-slate-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['2_contrast'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-yellow-600 uppercase">2. Tăng độ tương phản (CLAHE)</p>
                    <img src={intermediateSteps['2_contrast']} alt="Contrast" className="w-full rounded-lg border-2 border-yellow-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['3_denoised'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-cyan-600 uppercase">3. Khử nhiễu (Blur)</p>
                    <img src={intermediateSteps['3_denoised']} alt="Denoised" className="w-full rounded-lg border-2 border-cyan-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['4_threshold'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-blue-600 uppercase">4. Threshold (Adaptive)</p>
                    <img src={intermediateSteps['4_threshold']} alt="Threshold" className="w-full rounded-lg border-2 border-blue-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['5_opening'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-orange-600 uppercase">5. Opening (Sạch nhiễu)</p>
                    <img src={intermediateSteps['5_opening']} alt="Opening" className="w-full rounded-lg border-2 border-orange-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['6_closing'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-green-600 uppercase">6. Closing (Nối nét gãy)</p>
                    <img src={intermediateSteps['6_closing']} alt="Closing" className="w-full rounded-lg border-2 border-green-200 shadow-sm" />
                  </div>
                )}
                
                {intermediateSteps['7_final'] && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-pink-600 uppercase">7. Kết quả cuối (Dày nét)</p>
                    <img src={intermediateSteps['7_final']} alt="Final" className="w-full rounded-lg border-2 border-pink-200 shadow-sm" />
                  </div>
                )}
                
                {processedImage && (
                  <div className="space-y-2">
                    <p className="text-xs font-semibold text-green-600 uppercase">6. Kết quả</p>
                    <img src={processedImage} alt="Final" className="w-full rounded-lg border-2 border-green-500 shadow-md" />
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-slate-400">
                <Layers size={48} className="mx-auto mb-4 opacity-50" />
                <p>Chưa có dữ liệu các bước xử lý</p>
              </div>
            )}
          </div>
        )}

        {/* Tab: So sánh */}
        {activeTab === 'compare' && (
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-slate-600 uppercase">Ảnh gốc</h4>
                <img src={originalImage} alt="Original" className="w-full rounded-lg border-2 border-slate-300 shadow-lg" />
              </div>
              <div className="space-y-2">
                <h4 className="font-bold text-sm text-green-600 uppercase">Ảnh đã xử lý</h4>
                <img src={processedImage || originalImage} alt="Processed" className="w-full rounded-lg border-2 border-green-500 shadow-lg" />
              </div>
            </div>
            
            {/* Statistics & Evaluation Metrics */}
            {processingStats && (
              <div className="mt-6 space-y-4">
                {/* Processing Stats */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg text-center">
                    <p className="text-2xl font-bold text-blue-600">{processingStats.time}ms</p>
                    <p className="text-xs text-slate-600 mt-1">Thời gian xử lý</p>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg text-center">
                    <p className="text-2xl font-bold text-purple-600">{processingStats.width}×{processingStats.height}</p>
                    <p className="text-xs text-slate-600 mt-1">Kích thước</p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg text-center">
                    <p className="text-2xl font-bold text-green-600">{processingStats.steps}</p>
                    <p className="text-xs text-slate-600 mt-1">Số bước</p>
                  </div>
                </div>
                
                {/* Evaluation Metrics (Task requirement) */}
                {processingStats.evaluation && (
                  <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-4 rounded-xl border-2 border-indigo-200">
                    <h4 className="font-bold text-indigo-900 mb-3 flex items-center">
                      <Info size={16} className="mr-2" />
                      Đánh giá chất lượng (Task Requirement)
                    </h4>
                    <div className="grid grid-cols-4 gap-3 text-center">
                      <div className="bg-white p-3 rounded-lg">
                        <p className="text-lg font-bold text-indigo-600">{processingStats.evaluation.psnr}</p>
                        <p className="text-xs text-slate-600">PSNR (dB)</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg">
                        <p className="text-lg font-bold text-purple-600">{processingStats.evaluation.ssim}</p>
                        <p className="text-xs text-slate-600">SSIM</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg">
                        <p className="text-lg font-bold text-pink-600">{processingStats.evaluation['contrast_improvement_%']}%</p>
                        <p className="text-xs text-slate-600">Contrast ↑</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg">
                        <p className="text-lg font-bold text-green-600">{processingStats.evaluation['edge_preservation_%']}%</p>
                        <p className="text-xs text-slate-600">Edge ↓</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab: OCR */}
        {activeTab === 'ocr' && (
          <div className="p-6 min-h-[500px]">
            {extractedText ? (
              <textarea 
                className={`w-full h-[60vh] p-4 text-base leading-relaxed border-2 border-slate-200 rounded-xl focus:outline-none focus:ring-2 ${themeRing} font-mono bg-slate-50`}
                value={extractedText}
                readOnly
                placeholder="Kết quả OCR sẽ hiển thị ở đây..."
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-[60vh] text-slate-400">
                <Type size={64} className="mb-4 opacity-30" />
                <p className="text-lg font-semibold">Chưa có dữ liệu văn bản</p>
                <p className="text-sm mt-2">Nhấn "Trích xuất OCR" để bắt đầu</p>
              </div>
            )}
          </div>
        )}

        {/* Tab: So sánh OCR */}
        {activeTab === 'ocr-compare' && (
          <div className="p-6 min-h-[500px]">
            {ocrComparison ? (
              <div className="space-y-6">
                {/* Verdict Banner */}
                <div className={`p-4 rounded-xl border-2 ${
                  ocrComparison.comparison.verdict === 'IMPROVED' 
                    ? 'bg-green-50 border-green-300' 
                    : ocrComparison.comparison.verdict === 'DEGRADED'
                    ? 'bg-red-50 border-red-300'
                    : 'bg-yellow-50 border-yellow-300'
                }`}>
                  <h3 className={`font-bold text-xl mb-2 ${
                    ocrComparison.comparison.verdict === 'IMPROVED' ? 'text-green-700' :
                    ocrComparison.comparison.verdict === 'DEGRADED' ? 'text-red-700' :
                    'text-yellow-700'
                  }`}>
                    {ocrComparison.comparison.verdict === 'IMPROVED' ? '✓ CẢI THIỆN' :
                     ocrComparison.comparison.verdict === 'DEGRADED' ? '✗ SUY GIẢM' :
                     '= KHÔNG THAY ĐỔI'}
                  </h3>
                  <p className="text-sm text-slate-600">
                    {ocrComparison.comparison.verdict === 'IMPROVED' 
                      ? 'Xử lý ảnh đã cải thiện khả năng nhận dạng OCR'
                      : ocrComparison.comparison.verdict === 'DEGRADED'
                      ? 'Xử lý ảnh đã làm giảm khả năng nhận dạng OCR'
                      : 'Xử lý ảnh không ảnh hưởng đến khả năng nhận dạng OCR'}
                  </p>
                </div>

                {/* Comparison Metrics */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-white p-4 rounded-lg border-2 border-slate-200">
                    <p className="text-xs text-slate-600 mb-1">Độ tin cậy</p>
                    <p className={`text-2xl font-bold ${
                      ocrComparison.comparison.confidence_improvement > 0 ? 'text-green-600' :
                      ocrComparison.comparison.confidence_improvement < 0 ? 'text-red-600' :
                      'text-slate-600'
                    }`}>
                      {ocrComparison.comparison.confidence_improvement > 0 ? '+' : ''}{ocrComparison.comparison.confidence_improvement}%
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {ocrComparison.original.confidence}% → {ocrComparison.processed.confidence}%
                    </p>
                  </div>

                  <div className="bg-white p-4 rounded-lg border-2 border-slate-200">
                    <p className="text-xs text-slate-600 mb-1">Số ký tự</p>
                    <p className={`text-2xl font-bold ${
                      ocrComparison.comparison.characters_diff > 0 ? 'text-green-600' :
                      ocrComparison.comparison.characters_diff < 0 ? 'text-red-600' :
                      'text-slate-600'
                    }`}>
                      {ocrComparison.comparison.characters_diff > 0 ? '+' : ''}{ocrComparison.comparison.characters_diff}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {ocrComparison.original.characters} → {ocrComparison.processed.characters}
                    </p>
                  </div>

                  <div className="bg-white p-4 rounded-lg border-2 border-slate-200">
                    <p className="text-xs text-slate-600 mb-1">Số từ</p>
                    <p className={`text-2xl font-bold ${
                      ocrComparison.comparison.words_diff > 0 ? 'text-green-600' :
                      ocrComparison.comparison.words_diff < 0 ? 'text-red-600' :
                      'text-slate-600'
                    }`}>
                      {ocrComparison.comparison.words_diff > 0 ? '+' : ''}{ocrComparison.comparison.words_diff}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {ocrComparison.original.words} → {ocrComparison.processed.words}
                    </p>
                  </div>
                </div>

                {/* Image and Text Comparison */}
                <div className="space-y-4">
                  <h4 className="font-bold text-slate-800">So sánh hình ảnh</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm font-semibold text-slate-700 flex items-center">
                        <ImageIcon size={14} className="mr-2" />
                        Ảnh gốc (chưa xử lý)
                      </p>
                      <img 
                        src={originalImage} 
                        alt="Original" 
                        className="w-full rounded-lg border-2 border-red-300 shadow-sm"
                      />
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-semibold text-slate-700 flex items-center">
                        <Zap size={14} className="mr-2" />
                        Ảnh đã xử lý
                      </p>
                      <img 
                        src={processedImage} 
                        alt="Processed" 
                        className="w-full rounded-lg border-2 border-green-300 shadow-sm"
                      />
                    </div>
                  </div>
                </div>

                {/* Text Comparison */}
                <div className="space-y-4">
                  <h4 className="font-bold text-slate-800">So sánh văn bản OCR</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <p className="text-sm font-semibold text-slate-700 flex items-center">
                        <ImageIcon size={14} className="mr-2" />
                        Văn bản từ ảnh gốc
                      </p>
                      <textarea 
                        className="w-full h-64 p-3 text-sm border-2 border-red-200 rounded-lg bg-red-50 font-mono"
                        value={ocrComparison.original.text}
                        readOnly
                      />
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-semibold text-slate-700 flex items-center">
                        <Zap size={14} className="mr-2" />
                        Văn bản từ ảnh đã xử lý
                      </p>
                      <textarea 
                        className="w-full h-64 p-3 text-sm border-2 border-green-200 rounded-lg bg-green-50 font-mono"
                        value={ocrComparison.processed.text}
                        readOnly
                      />
                    </div>
                  </div>
                </div>

                {/* Statistics */}
                <div className="bg-slate-50 p-4 rounded-lg text-sm">
                  <p className="text-slate-600">
                    <strong>Thời gian xử lý:</strong> {ocrComparison.time}ms | 
                    <strong className="ml-3">Provider:</strong> {ocrComparison.provider} | 
                    <strong className="ml-3">Độ tương đồng:</strong> {ocrComparison.comparison.similarity_percent}%
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[60vh] text-slate-400">
                <Zap size={64} className="mb-4 opacity-30" />
                <p className="text-lg font-semibold">Chưa có kết quả so sánh</p>
                <p className="text-sm mt-2">Nhấn "So sánh OCR" để bắt đầu</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageViewer;
