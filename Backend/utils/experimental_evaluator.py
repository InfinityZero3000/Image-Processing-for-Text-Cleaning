"""
Experimental Evaluation Module
Đánh giá thực nghiệm kết quả xử lý ảnh

Chức năng:
- Đánh giá chất lượng xử lý ảnh trên từng ảnh đơn lẻ
- Đánh giá batch trên tập dữ liệu
- So sánh kết quả trước và sau xử lý
- Xuất báo cáo chi tiết (CSV, JSON, HTML)
- Tính toán các metrics: PSNR, SSIM, MSE, Contrast, Sharpness
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
from datetime import datetime
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentalEvaluator:
    """
    Đánh giá thực nghiệm theo yêu cầu task
    
    Viết chương trình thực nghiệm đánh giá kết quả trước và sau khi xử lý 
    trên tập dữ liệu đã áp dụng
    """
    
    def __init__(self, output_dir='logs/experiments'):
        self.results = []
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Khởi tạo ExperimentalEvaluator với thư mục output: {output_dir}")
    
    def evaluate_single_image(self, original, processed, image_name="", settings=None):
        """
        Đánh giá một ảnh theo yêu cầu task
        
        Returns:
            dict với các metrics và thống kê
        """
        try:
            # Convert to grayscale if needed
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            else:
                original_gray = original.copy()
            
            if len(processed.shape) == 3:
                processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                processed_gray = processed.copy()
            
            # Resize if needed
            if original_gray.shape != processed_gray.shape:
                processed_gray = cv2.resize(processed_gray, 
                    (original_gray.shape[1], original_gray.shape[0]))
            
            # 1. PSNR - Peak Signal to Noise Ratio
            psnr_value = psnr(original_gray, processed_gray, data_range=255)
            
            # 2. SSIM - Structural Similarity Index
            ssim_value = ssim(original_gray, processed_gray, data_range=255)
            
            # 3. MSE - Mean Squared Error
            mse = np.mean((original_gray.astype(float) - processed_gray.astype(float)) ** 2)
            
            # 4. Contrast Improvement
            original_contrast = np.std(original_gray)
            processed_contrast = np.std(processed_gray)
            contrast_improvement = ((processed_contrast - original_contrast) / original_contrast) * 100
            
            # 5. Noise Reduction (standard deviation of difference)
            noise_reduction = np.std(original_gray.astype(float) - processed_gray.astype(float))
            
            # 6. Edge Preservation
            original_edges = cv2.Canny(original_gray, 50, 150)
            processed_edges = cv2.Canny(processed_gray, 50, 150)
            edge_preservation = np.sum(processed_edges > 0) / np.sum(original_edges > 0) * 100
            
            # 7. Text Clarity (black pixel ratio in binary image)
            if processed_gray.max() > 1:  # Binary image (0-255)
                text_ratio = (np.sum(processed_gray == 0) / processed_gray.size) * 100
            else:
                text_ratio = 0
            
            result = {
                'image_name': image_name,
                'timestamp': datetime.now().isoformat(),
                'settings': settings or {},
                'metrics': {
                    'psnr': round(float(psnr_value), 2),
                    'ssim': round(float(ssim_value), 4),
                    'mse': round(float(mse), 2),
                    'contrast_improvement_%': round(float(contrast_improvement), 2),
                    'noise_reduction': round(float(noise_reduction), 2),
                    'edge_preservation_%': round(float(edge_preservation), 2),
                    'text_ratio_%': round(float(text_ratio), 2)
                },
                'statistics': {
                    'original': {
                        'mean': round(float(np.mean(original_gray)), 2),
                        'std': round(float(np.std(original_gray)), 2),
                        'min': int(np.min(original_gray)),
                        'max': int(np.max(original_gray))
                    },
                    'processed': {
                        'mean': round(float(np.mean(processed_gray)), 2),
                        'std': round(float(np.std(processed_gray)), 2),
                        'min': int(np.min(processed_gray)),
                        'max': int(np.max(processed_gray))
                    }
                }
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'image_name': image_name
            }
    
    def evaluate_dataset(self, dataset_path, processor, settings):
        """
        Đánh giá trên tập dữ liệu
        
        Args:
            dataset_path: Đường dẫn thư mục chứa ảnh
            processor: ImageProcessor instance
            settings: Cấu hình xử lý
        """
        results = []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(dataset_path) 
                              if f.lower().endswith(ext)])
        
        for img_file in image_files:
            img_path = os.path.join(dataset_path, img_file)
            
            # Load image
            original = cv2.imread(img_path)
            if original is None:
                continue
            
            # Process image
            result = processor.process_pipeline_v2(original, settings)
            processed = result['final_image']
            
            # Evaluate
            eval_result = self.evaluate_single_image(
                original, processed, 
                image_name=img_file, 
                settings=settings
            )
            
            results.append(eval_result)
        
        return results
    
    def generate_report(self, output_path='evaluation_report.json'):
        """
        Tạo báo cáo đánh giá
        """
        if not self.results:
            return None
        
        # Calculate statistics across all images
        all_metrics = [r['metrics'] for r in self.results if 'metrics' in r]
        
        summary = {
            'total_images': len(self.results),
            'timestamp': datetime.now().isoformat(),
            'average_metrics': {},
            'detailed_results': self.results
        }
        
        # Calculate averages
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                summary['average_metrics'][key] = {
                    'mean': round(np.mean(values), 2),
                    'std': round(np.std(values), 2),
                    'min': round(np.min(values), 2),
                    'max': round(np.max(values), 2)
                }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def generate_comparison_table(self):
        """
        Tạo bảng so sánh DataFrame
        """
        if not self.results:
            return None
        
        data = []
        for r in self.results:
            if 'metrics' in r:
                row = {
                    'Image': r['image_name'],
                    'PSNR (dB)': r['metrics']['psnr'],
                    'SSIM': r['metrics']['ssim'],
                    'MSE': r['metrics']['mse'],
                    'Contrast↑ (%)': r['metrics']['contrast_improvement_%'],
                    'Noise↓': r['metrics']['noise_reduction'],
                    'Edge (%)': r['metrics']['edge_preservation_%'],
                    'Text (%)': r['metrics']['text_ratio_%']
                }
                data.append(row)
        
        return pd.DataFrame(data)
