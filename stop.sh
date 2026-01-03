#!/bin/bash

# Script để dừng tất cả services
# Author: DocCleaner AI Team

echo "🛑 Đang dừng DocCleaner AI services..."

# Tìm và kill tất cả process liên quan
pkill -f "vite"
pkill -f "api/app.py"
pkill -f "npm run dev"

# Tìm và kill process đang dùng port 3000 và 5001
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null

echo "✅ Đã dừng tất cả services!"
