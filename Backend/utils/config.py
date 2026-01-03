"""
Configuration Module
Default settings and presets for Pipeline V4.0 PREMIUM
"""


class Config:
    """Configuration manager"""
    
    @staticmethod
    def get_default_config():
        """Get default Pipeline PREMIUM configuration"""
        return {
            # Pipeline selection
            'pipeline': 'premium',  # 'premium', 'handwriting', 'robust', 'v3'
            
            # Step 1: Border/Artifact Removal (NÂNG CẤP)
            'enableBorderRemoval': True,
            'borderThreshold': 50,
            'borderMargin': 0.15,
            'borderAggressive': True,  # Bật chế độ aggressive để loại artifact tốt hơn
            
            # Step 2: Illumination Correction
            'illuminationMethod': 'adaptive',  # 'adaptive', 'divide', 'subtract', 'morphological', 'none'
            'illuminationBlur': 51,
            
            # Step 3: Background Estimation
            'backgroundMethod': 'multiscale',  # 'multiscale', 'morphological', 'none'
            'bgKernelSmall': 15,
            'bgKernelMedium': 31,
            'bgKernelLarge': 61,
            
            # Step 4: Contrast Enhancement
            'contrastMethod': 'clahe_adaptive',  # 'clahe', 'clahe_adaptive', 'unsharp', 'gamma', 'none'
            'claheClipLimit': 3.0,
            'claheGrid': 8,
            
            # Step 5: Text Detection
            'textDetectionMethod': 'gradient',  # 'gradient', 'mser', 'edge', 'none'
            
            # Step 6: Binarization
            'thresholdMethod': 'sauvola_adaptive',  # 'sauvola', 'sauvola_adaptive', 'niblack', 'otsu', 'adaptive_gaussian', 'hybrid'
            'sauvolaWindow': 31,
            'sauvolaK': 0.2,
            
            # Step 7: Noise Removal
            'noiseRemovalMethod': 'adaptive',  # 'adaptive', 'median', 'morphological', 'none'
            'openingKernel': 2,
            
            # Step 8: Stroke Enhancement
            'strokeMethod': 'adaptive',  # 'none', 'light', 'adaptive', 'strong'
            'strokeKernelH': 3,
            'strokeKernelV': 2,
            'closingKernel': 3,
            
            # Step 9: Component Filtering
            'minComponentArea': 30,
        }
    
    @staticmethod
    def get_presets():
        """Get predefined configuration presets"""
        return {
            'premium_default': {
                'name': 'Premium - Mặc định',
                'description': 'Pipeline Premium với cài đặt tối ưu cho hầu hết tài liệu',
                'config': Config.get_default_config()
            },
            
            'premium_scan': {
                'name': 'Premium - Tài liệu Scan',
                'description': 'Tối ưu cho ảnh scan có viền đen, bóng, ánh sáng không đều',
                'config': {
                    'pipeline': 'premium',
                    'enableBorderRemoval': True,
                    'borderThreshold': 40,
                    'illuminationMethod': 'adaptive',
                    'backgroundMethod': 'multiscale',
                    'contrastMethod': 'clahe_adaptive',
                    'claheClipLimit': 3.5,
                    'textDetectionMethod': 'gradient',
                    'thresholdMethod': 'sauvola_adaptive',
                    'noiseRemovalMethod': 'adaptive',
                    'strokeMethod': 'light',
                    'minComponentArea': 25,
                }
            },
            
            'premium_photo': {
                'name': 'Premium - Ảnh chụp',
                'description': 'Xử lý ảnh chụp tài liệu bằng điện thoại (có bóng, nghiêng)',
                'config': {
                    'pipeline': 'premium',
                    'enableBorderRemoval': True,
                    'illuminationMethod': 'multiscale',
                    'illuminationBlur': 71,
                    'backgroundMethod': 'multiscale',
                    'bgKernelLarge': 81,
                    'contrastMethod': 'clahe_adaptive',
                    'claheClipLimit': 4.0,
                    'thresholdMethod': 'hybrid',
                    'noiseRemovalMethod': 'adaptive',
                    'strokeMethod': 'adaptive',
                    'minComponentArea': 40,
                }
            },
            
            'premium_handwriting': {
                'name': 'Premium - Chữ viết tay',
                'description': 'Tối ưu cho chữ viết tay, bảo toàn nét chữ mảnh',
                'config': {
                    'pipeline': 'premium',
                    'enableBorderRemoval': True,
                    'illuminationMethod': 'adaptive',
                    'backgroundMethod': 'multiscale',
                    'contrastMethod': 'clahe_adaptive',
                    'claheClipLimit': 3.0,
                    'textDetectionMethod': 'gradient',
                    'thresholdMethod': 'sauvola',
                    'sauvolaWindow': 25,
                    'sauvolaK': 0.15,
                    'noiseRemovalMethod': 'adaptive',
                    'openingKernel': 1,  # Rất nhỏ để không phá nét
                    'strokeMethod': 'strong',
                    'closingKernel': 4,
                    'minComponentArea': 20,
                }
            },
            
            'premium_old_document': {
                'name': 'Premium - Tài liệu cũ',
                'description': 'Xử lý tài liệu cũ, giấy ố vàng, chữ mờ',
                'config': {
                    'pipeline': 'premium',
                    'enableBorderRemoval': True,
                    'borderThreshold': 60,
                    'illuminationMethod': 'morphological',
                    'morphKernel': 61,
                    'backgroundMethod': 'multiscale',
                    'bgKernelLarge': 71,
                    'contrastMethod': 'clahe_adaptive',
                    'claheClipLimit': 4.5,
                    'thresholdMethod': 'sauvola_adaptive',
                    'sauvolaK': 0.3,
                    'noiseRemovalMethod': 'morphological',
                    'strokeMethod': 'strong',
                    'closingKernel': 5,
                    'minComponentArea': 35,
                }
            },
            
            'premium_clean': {
                'name': 'Premium - Tài liệu sạch',
                'description': 'Cho tài liệu chất lượng tốt, ít nhiễu',
                'config': {
                    'pipeline': 'premium',
                    'enableBorderRemoval': False,
                    'illuminationMethod': 'none',
                    'backgroundMethod': 'none',
                    'contrastMethod': 'none',
                    'textDetectionMethod': 'none',
                    'thresholdMethod': 'otsu',
                    'noiseRemovalMethod': 'adaptive',
                    'openingKernel': 2,
                    'strokeMethod': 'light',
                    'minComponentArea': 20,
                }
            },
            
            # Legacy presets (backward compatible)
            'default': {
                'name': 'Pipeline V2 - Default (Legacy)',
                'description': 'Cấu hình cũ cho tài liệu scan thông thường',
                'config': {
                    'pipeline': 'v3',
                    'backgroundRemoval': 'auto',
                    'backgroundKernel': 15,
                    'contrastMethod': 'clahe_masked',
                    'claheClipLimit': 2.0,
                    'thresholdMethod': 'otsu',
                    'kernelOpening': 2,
                    'kernelClosing': 2,
                }
            },
            
            'handwriting': {
                'name': 'Chữ viết tay (Legacy)',
                'description': 'Pipeline cũ cho chữ viết tay',
                'config': {
                    'pipeline': 'handwriting',
                    'strokeRepairLevel': 'strong',
                    'bgOpenKernel': 61,
                    'bgBlurKernel': 51,
                    'claheClipLimit': 3.5,
                    'claheGrid': 8,
                    'sauvolaWindowSize': 51,
                    'sauvolaK': 0.25,
                    'kernelOpening': 2,
                    'minComponentArea': 50,
                }
            },
            
            'robust': {
                'name': 'Robust Document (Legacy)',
                'description': 'Pipeline mạnh cho tài liệu phức tạp',
                'config': {
                    'pipeline': 'robust',
                    'denoiseStrength': 10,
                    'backgroundBlur': 35,
                    'backgroundKernel': 25,
                    'backgroundRemoval': 'auto',
                    'claheClipLimit': 2.5,
                    'claheGrid': 8,
                    'thresholdMethod': 'sauvola',
                    'removeLines': True,
                    'kernelOpening': 2,
                    'minComponentArea': 40,
                }
            }
        }
