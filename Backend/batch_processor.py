"""
Batch Image Processing Script
Xử lý hàng loạt ảnh và đánh giá kết quả theo yêu cầu task

Chức năng:
1. Xử lý nhiều ảnh trong thư mục
2. Áp dụng pipeline xử lý ảnh đầy đủ
3. So sánh kết quả trước và sau xử lý
4. Xuất báo cáo đánh giá chi tiết
"""

import cv2
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Tuple

# Import các module xử lý
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.image_processing import ImageProcessor
from utils.experimental_evaluator import ExperimentalEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchImageProcessor:
    """
    Xử lý hàng loạt ảnh theo yêu cầu task:
    
    Viết chương trình thực nghiệm đánh giá kết quả trước và sau khi xử lý 
    trên tập dữ liệu đã áp dụng.
    """
    
    def __init__(self, output_dir='logs/batch_experiments'):
        self.processor = ImageProcessor()
        self.evaluator = ExperimentalEvaluator(output_dir=output_dir)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)
        os.makedirs(f"{output_dir}/comparisons", exist_ok=True)
        logger.info(f"Khởi tạo BatchImageProcessor với output: {output_dir}")
    
    def process_directory(self, input_dir: str, settings: Dict = None) -> Dict:
        """
        Xử lý tất cả ảnh trong thư mục
        
        Args:
            input_dir: Đường dẫn thư mục chứa ảnh đầu vào
            settings: Cấu hình xử lý ảnh
        
        Returns:
            Dict chứa kết quả tổng hợp
        """
        if settings is None:
            settings = self._get_default_settings()
        
        logger.info(f"Bắt đầu xử lý batch từ thư mục: {input_dir}")
        logger.info(f"Cấu hình: {json.dumps(settings, indent=2, ensure_ascii=False)}")
        
        # Tìm tất cả file ảnh
        image_files = self._find_image_files(input_dir)
        logger.info(f"Tìm thấy {len(image_files)} ảnh để xử lý")
        
        if len(image_files) == 0:
            logger.warning("Không tìm thấy ảnh nào trong thư mục")
            return {'error': 'No images found'}
        
        results = []
        successful = 0
        failed = 0
        
        # Xử lý từng ảnh
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"Đang xử lý {idx}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                result = self._process_single_image(image_path, settings)
                results.append(result)
                successful += 1
                logger.info(f"✓ Xử lý thành công - PSNR: {result['metrics']['psnr']}, SSIM: {result['metrics']['ssim']}")
            except Exception as e:
                logger.error(f"✗ Lỗi xử lý {image_path}: {str(e)}")
                failed += 1
        
        # Tạo báo cáo tổng hợp
        summary = self._generate_summary(results, settings)
        summary['total_images'] = len(image_files)
        summary['successful'] = successful
        summary['failed'] = failed
        
        # Lưu kết quả
        self._save_results(results, summary)
        
        logger.info(f"Hoàn thành batch processing: {successful} thành công, {failed} thất bại")
        return summary
    
    def _process_single_image(self, image_path: str, settings: Dict) -> Dict:
        """Xử lý một ảnh đơn lẻ"""
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        # Chuyển từ BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Xử lý ảnh qua pipeline
        result = self.processor.process_pipeline_v2(image_rgb, settings)
        
        # Lưu ảnh đã xử lý
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        output_path = os.path.join(self.output_dir, 'processed', f"{name_without_ext}_processed.png")
        cv2.imwrite(output_path, result['final_image'])
        
        # Tạo ảnh so sánh
        comparison = self._create_comparison_image(
            result['original_gray'], 
            result['final_image'],
            filename
        )
        comparison_path = os.path.join(self.output_dir, 'comparisons', f"{name_without_ext}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        # Đánh giá chi tiết
        evaluation = self.evaluator.evaluate_single_image(
            result['original_gray'],
            result['final_image'],
            image_name=filename,
            settings=settings
        )
        
        return {
            'filename': filename,
            'input_path': image_path,
            'output_path': output_path,
            'comparison_path': comparison_path,
            'metrics': result['metrics'],
            'evaluation': evaluation,
            'processing_summary': result.get('processing_summary', {}),
            'image_size': image.shape
        }
    
    def _create_comparison_image(self, original: np.ndarray, processed: np.ndarray, title: str) -> np.ndarray:
        """Tạo ảnh so sánh trước và sau"""
        # Resize về cùng kích thước nếu cần
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Ghép 2 ảnh ngang
        comparison = np.hstack([original, processed])
        
        # Thêm text
        comparison_with_text = comparison.copy()
        h, w = comparison.shape[:2]
        
        # Vẽ text "Original" và "Processed"
        cv2.putText(comparison_with_text, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 128, 2)
        cv2.putText(comparison_with_text, "Processed", (w//2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 128, 2)
        
        return comparison_with_text
    
    def _find_image_files(self, directory: str) -> List[str]:
        """Tìm tất cả file ảnh trong thư mục"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _get_default_settings(self) -> Dict:
        """Cấu hình mặc định theo yêu cầu task"""
        return {
            'thresholdMethod': 'otsu',  # Otsu threshold
            'kernelOpening': 2,          # Kernel nhỏ 2x2 cho opening
            'kernelClosing': 3,          # Kernel 3x3 cho closing
            'backgroundRemoval': 'auto', # Auto chọn black-hat/top-hat
            'backgroundKernel': 15,      # Kernel lớn cho background removal
            'contrastMethod': 'none'     # Không cần CLAHE mặc định
        }
    
    def _generate_summary(self, results: List[Dict], settings: Dict) -> Dict:
        """Tạo báo cáo tổng hợp"""
        if not results:
            return {}
        
        # Tính toán thống kê
        psnr_values = [r['metrics']['psnr'] for r in results]
        ssim_values = [r['metrics']['ssim'] for r in results]
        mse_values = [r['metrics']['mse'] for r in results]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'settings': settings,
            'statistics': {
                'psnr': {
                    'mean': np.mean(psnr_values),
                    'std': np.std(psnr_values),
                    'min': np.min(psnr_values),
                    'max': np.max(psnr_values)
                },
                'ssim': {
                    'mean': np.mean(ssim_values),
                    'std': np.std(ssim_values),
                    'min': np.min(ssim_values),
                    'max': np.max(ssim_values)
                },
                'mse': {
                    'mean': np.mean(mse_values),
                    'std': np.std(mse_values),
                    'min': np.min(mse_values),
                    'max': np.max(mse_values)
                }
            },
            'detailed_results': results
        }
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict):
        """Lưu kết quả ra file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Lưu JSON
        json_path = os.path.join(self.output_dir, f'results_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu kết quả JSON: {json_path}")
        
        # Lưu CSV
        df_data = []
        for r in results:
            df_data.append({
                'filename': r['filename'],
                'psnr': r['metrics']['psnr'],
                'ssim': r['metrics']['ssim'],
                'mse': r['metrics']['mse'],
                'width': r['image_size'][1],
                'height': r['image_size'][0],
                'output_path': r['output_path']
            })
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(self.output_dir, f'results_{timestamp}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Đã lưu kết quả CSV: {csv_path}")
        
        # Lưu báo cáo HTML
        html_path = os.path.join(self.output_dir, f'report_{timestamp}.html')
        self._generate_html_report(summary, html_path)
        logger.info(f"Đã tạo báo cáo HTML: {html_path}")
    
    def _generate_html_report(self, summary: Dict, output_path: str):
        """Tạo báo cáo HTML"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Báo cáo Xử lý Ảnh - {summary.get('timestamp', '')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3fe; padding: 15px; border-left: 6px solid #2196F3; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
    </style>
</head>
<body>
    <h1>📊 Báo cáo Xử lý Ảnh Văn Bản</h1>
    <p><strong>Thời gian:</strong> {summary.get('timestamp', '')}</p>
    
    <div class="summary">
        <h2>Tổng quan</h2>
        <div class="metric">
            <strong>Tổng số ảnh:</strong> {summary.get('total_images', 0)}
        </div>
        <div class="metric">
            <strong>Thành công:</strong> {summary.get('successful', 0)}
        </div>
        <div class="metric">
            <strong>Thất bại:</strong> {summary.get('failed', 0)}
        </div>
    </div>
    
    <h2>Cấu hình Xử lý</h2>
    <table>
        <tr><th>Tham số</th><th>Giá trị</th></tr>
        <tr><td>Threshold Method</td><td>{summary.get('settings', {}).get('thresholdMethod', '')}</td></tr>
        <tr><td>Opening Kernel</td><td>{summary.get('settings', {}).get('kernelOpening', '')}×{summary.get('settings', {}).get('kernelOpening', '')}</td></tr>
        <tr><td>Closing Kernel</td><td>{summary.get('settings', {}).get('kernelClosing', '')}×{summary.get('settings', {}).get('kernelClosing', '')}</td></tr>
        <tr><td>Background Removal</td><td>{summary.get('settings', {}).get('backgroundRemoval', '')}</td></tr>
        <tr><td>Background Kernel</td><td>{summary.get('settings', {}).get('backgroundKernel', '')}×{summary.get('settings', {}).get('backgroundKernel', '')}</td></tr>
    </table>
    
    <h2>Thống kê Chất lượng</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Trung bình</th>
            <th>Độ lệch chuẩn</th>
            <th>Min</th>
            <th>Max</th>
        </tr>
        <tr>
            <td><strong>PSNR</strong></td>
            <td>{summary.get('statistics', {}).get('psnr', {}).get('mean', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('psnr', {}).get('std', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('psnr', {}).get('min', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('psnr', {}).get('max', 0):.2f}</td>
        </tr>
        <tr>
            <td><strong>SSIM</strong></td>
            <td>{summary.get('statistics', {}).get('ssim', {}).get('mean', 0):.4f}</td>
            <td>{summary.get('statistics', {}).get('ssim', {}).get('std', 0):.4f}</td>
            <td>{summary.get('statistics', {}).get('ssim', {}).get('min', 0):.4f}</td>
            <td>{summary.get('statistics', {}).get('ssim', {}).get('max', 0):.4f}</td>
        </tr>
        <tr>
            <td><strong>MSE</strong></td>
            <td>{summary.get('statistics', {}).get('mse', {}).get('mean', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('mse', {}).get('std', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('mse', {}).get('min', 0):.2f}</td>
            <td>{summary.get('statistics', {}).get('mse', {}).get('max', 0):.2f}</td>
        </tr>
    </table>
    
    <h2>Kết quả Chi tiết</h2>
    <table>
        <tr>
            <th>STT</th>
            <th>Tên file</th>
            <th>PSNR</th>
            <th>SSIM</th>
            <th>MSE</th>
            <th>Kích thước</th>
        </tr>
"""
        
        for idx, result in enumerate(summary.get('detailed_results', []), 1):
            html_content += f"""
        <tr>
            <td>{idx}</td>
            <td>{result['filename']}</td>
            <td>{result['metrics']['psnr']:.2f}</td>
            <td>{result['metrics']['ssim']:.4f}</td>
            <td>{result['metrics']['mse']:.2f}</td>
            <td>{result['image_size'][1]}×{result['image_size'][0]}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <p style="margin-top: 30px; color: #666;">
        <em>Báo cáo được tạo tự động bởi hệ thống xử lý ảnh văn bản</em>
    </p>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Hàm chính để chạy batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Xử lý hàng loạt ảnh văn bản')
    parser.add_argument('input_dir', help='Thư mục chứa ảnh đầu vào')
    parser.add_argument('--output-dir', default='logs/batch_experiments', 
                       help='Thư mục lưu kết quả (mặc định: logs/batch_experiments)')
    parser.add_argument('--threshold', default='otsu', 
                       choices=['otsu', 'adaptive_mean', 'adaptive_gaussian'],
                       help='Phương pháp threshold (mặc định: otsu)')
    parser.add_argument('--bg-removal', default='auto',
                       choices=['auto', 'blackhat', 'tophat', 'none'],
                       help='Phương pháp loại nền (mặc định: auto)')
    
    args = parser.parse_args()
    
    # Tạo settings
    settings = {
        'thresholdMethod': args.threshold,
        'kernelOpening': 2,
        'kernelClosing': 3,
        'backgroundRemoval': args.bg_removal,
        'backgroundKernel': 15,
        'contrastMethod': 'none'
    }
    
    # Khởi tạo và chạy
    processor = BatchImageProcessor(output_dir=args.output_dir)
    summary = processor.process_directory(args.input_dir, settings)
    
    print("\n" + "="*60)
    print("✓ HOÀN THÀNH XỬ LÝ BATCH")
    print("="*60)
    print(f"Tổng số ảnh: {summary.get('total_images', 0)}")
    print(f"Thành công: {summary.get('successful', 0)}")
    print(f"Thất bại: {summary.get('failed', 0)}")
    print(f"\nKết quả được lưu tại: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
