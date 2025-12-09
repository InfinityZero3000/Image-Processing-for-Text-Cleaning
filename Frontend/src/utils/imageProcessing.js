// ===== IMAGE PROCESSING UTILITIES =====
// Pipeline V2 - Fixed Background Removal & CLAHE Masked

// Chuyển sang grayscale
export const applyGrayscale = (imageData) => {
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    data[i] = data[i + 1] = data[i + 2] = gray;
  }
  return imageData;
};

// Tính mean cục bộ (Local Mean)
export const calculateLocalMean = (imageData, kernelSize) => {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const mean = new Array(width * height).fill(0);
  const halfKernel = Math.floor(kernelSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, count = 0;
      
      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const nx = x + kx;
          const ny = y + ky;
          
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const idx = (ny * width + nx) * 4;
            sum += data[idx];
            count++;
          }
        }
      }
      
      mean[y * width + x] = sum / count;
    }
  }
  
  return mean;
};

// Background Removal (Simplified Morphological Transform)
export const applyBackgroundRemoval = (imageData, settings) => {
  const data = imageData.data;
  const kernel = settings.backgroundKernel;
  
  // Tính mean cục bộ
  const mean = calculateLocalMean(imageData, kernel);
  
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    let value = data[i] - mean[idx];
    
    if (settings.backgroundRemoval === 'blackhat') {
      // Blackhat: Loại vết tối (dark stains)
      value = Math.max(0, data[i] - Math.abs(value) * 0.7);
    } else if (settings.backgroundRemoval === 'tophat') {
      // Tophat: Làm sáng nền
      value = Math.min(255, data[i] + Math.abs(value) * 0.5);
    } else {
      // Auto: Kết hợp cả hai
      value = data[i] - Math.abs(value) * 0.3;
    }
    
    data[i] = data[i + 1] = data[i + 2] = Math.max(0, Math.min(255, value));
  }
  
  return imageData;
};

// Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
export const applyContrastEnhancement = (imageData, settings) => {
  if (settings.contrastMethod === 'none') return imageData;
  
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  
  // CLAHE parameters
  const tileSize = 16; // Kích thước tile nhỏ hơn cho local enhancement
  const clipLimit = settings.claheClipLimit || 2.0;
  
  // Divide image into tiles
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);
  
  // Process each tile
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x1 = tx * tileSize;
      const y1 = ty * tileSize;
      const x2 = Math.min(x1 + tileSize, width);
      const y2 = Math.min(y1 + tileSize, height);
      
      // Calculate histogram for this tile
      const histogram = new Array(256).fill(0);
      let pixelCount = 0;
      
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          const idx = (y * width + x) * 4;
          histogram[data[idx]]++;
          pixelCount++;
        }
      }
      
      // Clip histogram
      const clipValue = Math.floor(clipLimit * pixelCount / 256);
      let clippedPixels = 0;
      
      for (let i = 0; i < 256; i++) {
        if (histogram[i] > clipValue) {
          clippedPixels += histogram[i] - clipValue;
          histogram[i] = clipValue;
        }
      }
      
      // Redistribute clipped pixels
      const redistribution = Math.floor(clippedPixels / 256);
      for (let i = 0; i < 256; i++) {
        histogram[i] += redistribution;
      }
      
      // Calculate CDF
      const cdf = new Array(256);
      cdf[0] = histogram[0];
      for (let i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
      }
      
      // Normalize and apply
      const cdfMin = cdf.find(v => v > 0) || 0;
      
      for (let y = y1; y < y2; y++) {
        for (let x = x1; x < x2; x++) {
          const idx = (y * width + x) * 4;
          const oldValue = data[idx];
          const newValue = Math.round(((cdf[oldValue] - cdfMin) / (pixelCount - cdfMin)) * 255);
          data[idx] = data[idx + 1] = data[idx + 2] = Math.min(255, Math.max(0, newValue));
        }
      }
    }
  }
  
  return imageData;
};

// Otsu Threshold Calculation
export const calculateOtsuThreshold = (histogram, total) => {
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }
  
  let sumB = 0, wB = 0, wF = 0;
  let maxVariance = 0, threshold = 0;
  
  for (let i = 0; i < 256; i++) {
    wB += histogram[i];
    if (wB === 0) continue;
    
    wF = total - wB;
    if (wF === 0) break;
    
    sumB += i * histogram[i];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const variance = wB * wF * (mB - mF) * (mB - mF);
    
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = i;
    }
  }
  
  return threshold;
};

// Median Filter - loại bỏ nhiễu salt & pepper
export const applyMedianFilter = (imageData, kernelSize = 3) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const output = new Uint8ClampedArray(data);
  const halfKernel = Math.floor(kernelSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixels = [];
      
      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          const ny = Math.min(Math.max(y + ky, 0), height - 1);
          const idx = (ny * width + nx) * 4;
          pixels.push(data[idx]);
        }
      }
      
      pixels.sort((a, b) => a - b);
      const median = pixels[Math.floor(pixels.length / 2)];
      
      const idx = (y * width + x) * 4;
      output[idx] = output[idx + 1] = output[idx + 2] = median;
      output[idx + 3] = 255;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Gaussian Blur - làm mịn để giảm nhiễu
export const applyGaussianBlur = (imageData, kernelSize = 5, sigma = 1.0) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const output = new Uint8ClampedArray(data);
  const halfKernel = Math.floor(kernelSize / 2);
  
  // Create Gaussian kernel
  const kernel = [];
  let kernelSum = 0;
  for (let y = -halfKernel; y <= halfKernel; y++) {
    kernel[y + halfKernel] = [];
    for (let x = -halfKernel; x <= halfKernel; x++) {
      const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel[y + halfKernel][x + halfKernel] = value;
      kernelSum += value;
    }
  }
  
  // Normalize kernel
  for (let y = 0; y < kernelSize; y++) {
    for (let x = 0; x < kernelSize; x++) {
      kernel[y][x] /= kernelSum;
    }
  }
  
  // Apply convolution
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      
      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          const ny = Math.min(Math.max(y + ky, 0), height - 1);
          const idx = (ny * width + nx) * 4;
          sum += data[idx] * kernel[ky + halfKernel][kx + halfKernel];
        }
      }
      
      const idx = (y * width + x) * 4;
      output[idx] = output[idx + 1] = output[idx + 2] = sum;
      output[idx + 3] = 255;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Bilateral Filter - làm mịn nền NHƯNG giữ cạnh chữ
export const applyBilateralFilter = (imageData, d = 9, sigmaColor = 75, sigmaSpace = 75) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const output = new Uint8ClampedArray(data);
  const halfD = Math.floor(d / 2);
  
  // Precompute Gaussian for space
  const gaussianSpace = [];
  for (let i = -halfD; i <= halfD; i++) {
    gaussianSpace[i + halfD] = [];
    for (let j = -halfD; j <= halfD; j++) {
      const dist = Math.sqrt(i * i + j * j);
      gaussianSpace[i + halfD][j + halfD] = Math.exp(-(dist * dist) / (2 * sigmaSpace * sigmaSpace));
    }
  }
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const centerVal = data[idx];
      
      let sumWeight = 0;
      let sumValue = 0;
      
      for (let ky = -halfD; ky <= halfD; ky++) {
        for (let kx = -halfD; kx <= halfD; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          const ny = Math.min(Math.max(y + ky, 0), height - 1);
          const nidx = (ny * width + nx) * 4;
          const neighborVal = data[nidx];
          
          // Color difference
          const colorDiff = Math.abs(centerVal - neighborVal);
          const gaussianColor = Math.exp(-(colorDiff * colorDiff) / (2 * sigmaColor * sigmaColor));
          
          // Combined weight
          const weight = gaussianSpace[ky + halfD][kx + halfD] * gaussianColor;
          
          sumWeight += weight;
          sumValue += weight * neighborVal;
        }
      }
      
      const result = sumValue / sumWeight;
      output[idx] = output[idx + 1] = output[idx + 2] = result;
      output[idx + 3] = 255;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Sauvola Threshold - TỐT NHẤT cho văn bản scan nhiễu
export const applySauvolaThreshold = (imageData, windowSize = 15, k = 0.5, R = 128) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const output = new Uint8ClampedArray(data);
  const halfWindow = Math.floor(windowSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, sumSq = 0, count = 0;
      
      // Calculate local mean and std dev
      for (let ky = -halfWindow; ky <= halfWindow; ky++) {
        for (let kx = -halfWindow; kx <= halfWindow; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          const ny = Math.min(Math.max(y + ky, 0), height - 1);
          const idx = (ny * width + nx) * 4;
          const val = data[idx];
          sum += val;
          sumSq += val * val;
          count++;
        }
      }
      
      const mean = sum / count;
      const variance = (sumSq / count) - (mean * mean);
      const stdDev = Math.sqrt(Math.max(0, variance));
      
      // Sauvola formula: T = mean * (1 + k * ((stdDev / R) - 1))
      const threshold = mean * (1 + k * ((stdDev / R) - 1));
      
      const idx = (y * width + x) * 4;
      const pixel = data[idx];
      const value = pixel > threshold ? 255 : 0;
      
      output[idx] = output[idx + 1] = output[idx + 2] = value;
      output[idx + 3] = 255;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Niblack Threshold - Tốt cho chữ viết tay
export const applyNiblackThreshold = (imageData, windowSize = 15, k = -0.2) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const output = new Uint8ClampedArray(data);
  const halfWindow = Math.floor(windowSize / 2);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, sumSq = 0, count = 0;
      
      for (let ky = -halfWindow; ky <= halfWindow; ky++) {
        for (let kx = -halfWindow; kx <= halfWindow; kx++) {
          const nx = Math.min(Math.max(x + kx, 0), width - 1);
          const ny = Math.min(Math.max(y + ky, 0), height - 1);
          const idx = (ny * width + nx) * 4;
          const val = data[idx];
          sum += val;
          sumSq += val * val;
          count++;
        }
      }
      
      const mean = sum / count;
      const variance = (sumSq / count) - (mean * mean);
      const stdDev = Math.sqrt(Math.max(0, variance));
      
      // Niblack formula: T = mean + k * stdDev
      const threshold = mean + k * stdDev;
      
      const idx = (y * width + x) * 4;
      const pixel = data[idx];
      const value = pixel > threshold ? 255 : 0;
      
      output[idx] = output[idx + 1] = output[idx + 2] = value;
      output[idx + 3] = 255;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Threshold - Otsu + Adaptive + Sauvola + Niblack
export const applyThreshold = (imageData, method, sauvolaK = 0.5, niblackK = -0.2, windowSize = 25) => {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  
  if (method === 'niblack') {
    // Niblack - TỐT cho chữ viết tay
    return applyNiblackThreshold(imageData, windowSize, niblackK);
    
  } else if (method === 'sauvola') {
    // Sauvola - TỐT cho ảnh scan nhiễu
    return applySauvolaThreshold(imageData, windowSize, sauvolaK, 128);
    
  } else if (method === 'adaptive_mean' || method === 'adaptive_gaussian') {
    // Adaptive threshold
    const blockSize = 11;
    const C = 10;
    const halfBlock = Math.floor(blockSize / 2);
    
    const output = new Uint8ClampedArray(data);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0, count = 0;
        
        for (let ky = -halfBlock; ky <= halfBlock; ky++) {
          for (let kx = -halfBlock; kx <= halfBlock; kx++) {
            const nx = Math.min(Math.max(x + kx, 0), width - 1);
            const ny = Math.min(Math.max(y + ky, 0), height - 1);
            const idx = (ny * width + nx) * 4;
            sum += data[idx];
            count++;
          }
        }
        
        const localMean = sum / count;
        const idx = (y * width + x) * 4;
        const pixel = data[idx];
        
        const value = pixel > (localMean - C) ? 255 : 0;
        output[idx] = output[idx + 1] = output[idx + 2] = value;
      }
    }
    
    for (let i = 0; i < data.length; i++) {
      data[i] = output[i];
    }
    
  } else {
    // Otsu threshold
    const histogram = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
      histogram[data[i]]++;
    }
    const threshold = calculateOtsuThreshold(histogram, data.length / 4);
    
    for (let i = 0; i < data.length; i += 4) {
      const value = data[i] > threshold ? 255 : 0;
      data[i] = data[i + 1] = data[i + 2] = value;
    }
  }
  
  return imageData;
};

// Erosion (Morphological Operation)
export const applyErosion = (imageData, kernelSize) => {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const output = new Uint8ClampedArray(data);
  const halfKernel = Math.floor(kernelSize / 2);
  
  for (let y = halfKernel; y < height - halfKernel; y++) {
    for (let x = halfKernel; x < width - halfKernel; x++) {
      let minVal = 255;
      
      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const idx = ((y + ky) * width + (x + kx)) * 4;
          minVal = Math.min(minVal, data[idx]);
        }
      }
      
      const idx = (y * width + x) * 4;
      output[idx] = output[idx + 1] = output[idx + 2] = minVal;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Dilation (Morphological Operation)
export const applyDilation = (imageData, kernelSize) => {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  const output = new Uint8ClampedArray(data);
  const halfKernel = Math.floor(kernelSize / 2);
  
  for (let y = halfKernel; y < height - halfKernel; y++) {
    for (let x = halfKernel; x < width - halfKernel; x++) {
      let maxVal = 0;
      
      for (let ky = -halfKernel; ky <= halfKernel; ky++) {
        for (let kx = -halfKernel; kx <= halfKernel; kx++) {
          const idx = ((y + ky) * width + (x + kx)) * 4;
          maxVal = Math.max(maxVal, data[idx]);
        }
      }
      
      const idx = (y * width + x) * 4;
      output[idx] = output[idx + 1] = output[idx + 2] = maxVal;
    }
  }
  
  for (let i = 0; i < data.length; i++) {
    data[i] = output[i];
  }
  
  return imageData;
};

// Morphological Opening (Erosion → Dilation)
export const applyMorphologicalOpening = (imageData, kernelSize) => {
  let eroded = applyErosion(imageData, kernelSize);
  return applyDilation(eroded, kernelSize);
};

// Morphological Closing (Dilation → Erosion)
export const applyMorphologicalClosing = (imageData, kernelSize) => {
  let dilated = applyDilation(imageData, kernelSize);
  return applyErosion(dilated, kernelSize);
};
