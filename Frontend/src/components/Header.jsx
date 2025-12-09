import React from 'react';
import { ScanText, RefreshCw } from 'lucide-react';

const Header = ({ onReset }) => {
  const themeColor = "bg-[#800020]";
  
  return (
    <header className={`${themeColor} text-white shadow-lg sticky top-0 z-50`}>
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-white/10 p-2 rounded-lg backdrop-blur-sm">
            <ScanText className="w-7 h-7" />
          </div>
          <div>
            <h1 className="text-xl font-bold leading-tight">DocCleaner AI</h1>
            <p className="text-xs text-red-200">Pipeline V2 - Morphological Image Processing</p>
          </div>
        </div>
        <button 
          onClick={onReset}
          className="flex items-center space-x-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all text-sm font-medium backdrop-blur-sm"
        >
          <RefreshCw size={16} />
          <span>Làm mới</span>
        </button>
      </div>
    </header>
  );
};

export default Header;
