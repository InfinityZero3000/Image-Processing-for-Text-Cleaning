#model AI
qubvel-hf/documents-restoration

Input BGR
    ↓
【1】Grayscale
    ↓
【2】Background Feature (Auto: Blackhat/Tophat)
    ↓
【3】Enhance với Feature
    ↓
【4】Binarization (Otsu/Adaptive)
    ↓
【5】Opening (Loại nhiễu)
    ↓
【6】Closing (Nối nét)
    ↓
Output Final

CLAHE (Contrast Limited Adaptive Histogram Equalization) - tăng contrast
