@echo off
REM Script để chạy Frontend và Backend đồng thời trên Windows
REM Author: DocCleaner AI Team
REM Date: 2025-12-09

echo ========================================
echo    DocCleaner AI - Starting Services   
echo ========================================
echo.

REM Tạo thư mục logs nếu chưa có
if not exist logs mkdir logs

REM Tạo timestamp cho log files
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set FRONTEND_LOG=logs\frontend_%TIMESTAMP%.log
set BACKEND_LOG=logs\backend_%TIMESTAMP%.log

REM Kiểm tra và cài đặt Frontend dependencies
echo [1/4] Kiểm tra Frontend dependencies...
if not exist "Frontend\node_modules" (
    echo Đang cài đặt Frontend dependencies...
    cd Frontend
    call npm install
    cd ..
    echo [OK] Frontend dependencies đã cài đặt
) else (
    echo [OK] Frontend dependencies đã có
)
echo.

REM Kiểm tra và cài đặt Backend dependencies
echo [2/4] Kiểm tra Backend dependencies...
if not exist "Backend\venv" (
    echo Đang tạo Python virtual environment...
    cd Backend
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Đang cài đặt Backend dependencies...
    pip install -r requirements.txt
    deactivate
    cd ..
    echo [OK] Backend dependencies đã cài đặt
) else (
    echo [OK] Backend virtual environment đã có
)
echo.

REM Khởi động Backend
echo [3/4] Đang khởi động Backend (Port 5001)...
start "DocCleaner Backend" /MIN cmd /c "cd Backend && venv\Scripts\activate.bat && python app.py > ..\%BACKEND_LOG% 2>&1"
echo [OK] Backend đang chạy trên http://localhost:5001
echo Backend logs: %BACKEND_LOG%
echo.

REM Đợi Backend khởi động
echo Đang đợi Backend khởi động...
timeout /t 3 /nobreak > nul

REM Khởi động Frontend
echo [4/4] Đang khởi động Frontend (Port 3000)...
start "DocCleaner Frontend" /MIN cmd /c "cd Frontend && npm run dev > ..\%FRONTEND_LOG% 2>&1"
echo [OK] Frontend đang chạy trên http://localhost:3000
echo Frontend logs: %FRONTEND_LOG%
echo.

echo ========================================
echo       Services đã được khởi động!      
echo ========================================
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5001
echo.
echo Để xem logs:
echo   - Frontend: type %FRONTEND_LOG%
echo   - Backend:  type %BACKEND_LOG%
echo.
echo Để dừng services, chạy: stop.bat
echo Hoặc đóng các cửa sổ console Backend và Frontend
echo.
pause
