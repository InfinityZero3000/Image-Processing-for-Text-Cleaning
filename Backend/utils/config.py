"""
Configuration Module
Default settings and presets
"""


class Config:
    """Configuration manager"""
    
    @staticmethod
    def get_default_config():
        """Get default Pipeline V2 configuration"""
        return {
            # FR6: Background Removal
            'backgroundRemoval': 'auto',  # 'auto', 'blackhat', 'tophat', 'none'
            'backgroundKernel': 15,  # 15x15 kernel (V2 fixed)
            
            # FR7: Contrast Enhancement
            'contrastMethod': 'clahe_masked',  # 'clahe', 'clahe_masked', 'histogram_eq', 'none'
            'claheClipLimit': 2.0,
            'claheTileGrid': 8,
            
            # FR3: Threshold
            'thresholdMethod': 'otsu',  # 'otsu', 'adaptive_mean', 'adaptive_gaussian'
            
            # FR4: Opening (Noise Removal) - Thực hiện SAU closing
            'kernelOpening': 2,  # 2x2 - kernel nhỏ để không phá vỡ chữ
            
            # FR5: Closing (Connect Strokes) - Thực hiện TRƯỚC opening
            'kernelClosing': 2,  # 2x2 - kernel nhỏ để kết nối các nét gần nhau
        }
    
    @staticmethod
    def get_presets():
        """Get predefined configuration presets"""
        return {
            'default': {
                'name': 'Pipeline V2 - Default',
                'description': 'Cấu hình chuẩn cho tài liệu scan thông thường',
                'config': Config.get_default_config()
            },
            
            'heavy_stains': {
                'name': 'Vết bẩn nặng',
                'description': 'Xử lý ảnh có nhiều vết bẩn, vết coffee',
                'config': {
                    'backgroundRemoval': 'blackhat',
                    'backgroundKernel': 21,  # Kernel lớn hơn
                    'contrastMethod': 'clahe_masked',
                    'claheClipLimit': 3.0,  # Tăng clip limit
                    'thresholdMethod': 'adaptive_gaussian',
                    'kernelOpening': 3,  # Tăng opening
                    'kernelClosing': 3
                }
            },
            
            'broken_strokes': {
                'name': 'Nét chữ đứt gãy (MẠNH)',
                'description': 'Ảnh có nét chữ bị đứt nhiều, cần nối liền - TĂNG CƯỜNG',
                'config': {
                    'backgroundRemoval': 'none',  # Tắt để không làm mất chữ
                    'backgroundKernel': 15,
                    'contrastMethod': 'none',
                    'claheClipLimit': 2.5,
                    'thresholdMethod': 'adaptive_gaussian',  # Tốt hơn cho chữ viết tay
                    'kernelOpening': 1,  # Rất nhỏ - chỉ loại điểm lẻ
                    'kernelClosing': 4  # Tăng mạnh để nối chữ
                }
            },
            
            'handwriting': {
                'name': 'Chữ viết tay (TĂNG CƯỜNG)',
                'description': 'Tối ưu cho ảnh chữ viết tay - LÀM LIỀN NÉT MẠNH',
                'config': {
                    'backgroundRemoval': 'none',  
                    'backgroundKernel': 15,
                    'contrastMethod': 'none',
                    'claheClipLimit': 2.0,
                    'thresholdMethod': 'adaptive_gaussian',
                    'kernelOpening': 1,  # Gần như không opening
                    'kernelClosing': 5  # Closing mạnh nhất
                }
            },
            
            'faded_text': {
                'name': 'Text mờ nhạt',
                'description': 'Ảnh có chữ mờ, cần tăng độ tương phản mạnh',
                'config': {
                    'backgroundRemoval': 'tophat',
                    'backgroundKernel': 15,
                    'contrastMethod': 'clahe',
                    'claheClipLimit': 3.5,  # Clip limit cao
                    'thresholdMethod': 'otsu',
                    'kernelOpening': 2,
                    'kernelClosing': 3
                }
            },
            
            'low_noise': {
                'name': 'Ảnh sạch ít nhiễu',
                'description': 'Ảnh scan chất lượng tốt, chỉ cần xử lý nhẹ',
                'config': {
                    'backgroundRemoval': 'none',
                    'backgroundKernel': 15,
                    'contrastMethod': 'histogram_eq',
                    'claheClipLimit': 2.0,
                    'thresholdMethod': 'otsu',
                    'kernelOpening': 2,
                    'kernelClosing': 2  # Opening/Closing nhỏ
                }
            }
        }
