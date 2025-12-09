#!/bin/bash

# Script để chạy Frontend và Backend đồng thời với logging
# Author: DocCleaner AI Team
# Date: 2025-11-29

# Màu sắc cho terminal
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Tạo thư mục logs nếu chưa có
mkdir -p logs

# Tạo timestamp cho log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FRONTEND_LOG="logs/frontend_${TIMESTAMP}.log"
BACKEND_LOG="logs/backend_${TIMESTAMP}.log"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   DocCleaner AI - Starting Services   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function để dọn dẹp khi thoát
cleanup() {
    echo ""
    echo -e "${YELLOW}Đang dừng các services...${NC}"
    kill $FRONTEND_PID $BACKEND_PID $TAIL_PID 2>/dev/null
    wait $FRONTEND_PID $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}Đã dừng tất cả services.${NC}"
    exit 0
}

# Bắt signal Ctrl+C
trap cleanup SIGINT SIGTERM

# Kiểm tra và cài đặt Frontend dependencies
echo -e "${BLUE}[1/4] Kiểm tra Frontend dependencies...${NC}"
if [ ! -d "Frontend/node_modules" ]; then
    echo -e "${YELLOW}Đang cài đặt Frontend dependencies...${NC}"
    cd Frontend && npm install && cd ..
    echo -e "${GREEN}✓ Frontend dependencies đã cài đặt${NC}"
else
    echo -e "${GREEN}✓ Frontend dependencies đã có${NC}"
fi

# Kiểm tra và cài đặt Backend dependencies
echo -e "${BLUE}[2/4] Kiểm tra Backend dependencies...${NC}"
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Đang cài đặt Backend dependencies...${NC}"
    echo -e "${YELLOW}Lưu ý: Sử dụng Python 3.11 (python3 alias)${NC}"
    python3 -m pip install pytesseract opencv-python numpy pillow flask flask-cors scikit-image scipy pandas gunicorn python-dotenv
    echo -e "${GREEN}✓ Backend dependencies đã cài đặt${NC}"
else
    echo -e "${GREEN}✓ Backend dependencies đã có${NC}"
fi

echo ""
echo -e "${BLUE}[3/4] Khởi động Backend Server...${NC}"
echo -e "      Log file: ${BACKEND_LOG}"
echo -e "      Port: 5001 (tránh conflict với AirPlay port 5000)"
cd Backend
python3 app.py > "../${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
cd ..

# Đợi Backend khởi động
echo -e "${YELLOW}Đợi Backend khởi động...${NC}"
for i in {1..10}; do
    sleep 1
    if lsof -i:5001 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend Server đang chạy (PID: $BACKEND_PID)${NC}"
        echo -e "      URL: http://localhost:5001"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}✗ Backend Server không khởi động được sau 10 giây${NC}"
        echo -e "${RED}  Xem log tại: ${BACKEND_LOG}${NC}"
        if [ -s "${BACKEND_LOG}" ]; then
            echo ""
            echo -e "${RED}=== LOG OUTPUT ===${NC}"
            tail -50 "${BACKEND_LOG}"
        else
            echo -e "${RED}Log file trống. Có thể process die ngay.${NC}"
        fi
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# Double check process còn sống
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Backend process đã die${NC}"
    echo -e "${RED}  Xem log tại: ${BACKEND_LOG}${NC}"
    cat "${BACKEND_LOG}"
    exit 1
fi

echo ""
echo -e "${BLUE}[4/4] Khởi động Frontend Server...${NC}"
echo -e "      Log file: ${FRONTEND_LOG}"
cd Frontend
npm run dev > "../${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!
cd ..

# Đợi Frontend khởi động
echo -e "${YELLOW}Đợi Frontend khởi động...${NC}"
for i in {1..15}; do
    sleep 1
    if lsof -i:3000 >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend Server đang chạy (PID: $FRONTEND_PID)${NC}"
        echo -e "      URL: http://localhost:3000"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${RED}✗ Frontend Server không khởi động được sau 15 giây${NC}"
        echo -e "${RED}  Xem log tại: ${FRONTEND_LOG}${NC}"
        if [ -s "${FRONTEND_LOG}" ]; then
            echo ""
            echo -e "${RED}=== LOG OUTPUT ===${NC}"
            tail -50 "${FRONTEND_LOG}"
        else
            echo -e "${RED}Log file trống. Có thể process die ngay.${NC}"
        fi
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# Double check process còn sống
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Frontend process đã die${NC}"
    echo -e "${RED}  Xem log tại: ${FRONTEND_LOG}${NC}"
    cat "${FRONTEND_LOG}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   ✅ Tất cả services đã khởi động!    ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📱 Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}🔧 Backend:${NC}  http://localhost:5001"
echo -e "${BLUE}📊 API Docs:${NC} http://localhost:5001/api/config"
echo ""
echo -e "${BLUE}📝 Log files:${NC}"
echo -e "  • Frontend: ${FRONTEND_LOG}"
echo -e "  • Backend:  ${BACKEND_LOG}"
echo ""
echo -e "${YELLOW} Tips:${NC}"
echo -e "  • Nhấn ${YELLOW}Ctrl+C${NC} để dừng tất cả services"
echo -e "  • Xem logs: ${YELLOW}tail -f ${FRONTEND_LOG} ${BACKEND_LOG}${NC}"
echo -e "  • Test Backend: ${YELLOW}curl http://localhost:5001/${NC}"
echo ""
echo -e "${BLUE}📊 Theo dõi logs realtime:${NC}"
echo ""

# Theo dõi logs theo thời gian thực
tail -f "${FRONTEND_LOG}" "${BACKEND_LOG}" &
TAIL_PID=$!

# Đợi cho đến khi người dùng nhấn Ctrl+C
wait $FRONTEND_PID $BACKEND_PID

# Cleanup khi process kết thúc
cleanup
