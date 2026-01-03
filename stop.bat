@echo off
REM Script để dừng tất cả services trên Windows
REM Author: DocCleaner AI Team
REM Date: 2025-12-09

echo.
echo ========================================
echo   Đang dừng DocCleaner AI services...
echo ========================================
echo.

REM Tìm và kill process Node.js (Frontend)
echo Đang dừng Frontend...
taskkill /F /IM node.exe /T 2>nul
if %errorlevel% equ 0 (
    echo [OK] Frontend đã dừng
) else (
    echo [INFO] Không tìm thấy Frontend process
)
echo.

REM Tìm và kill process Python (Backend)
echo Đang dừng Backend...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5001" ^| find "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
    if %errorlevel% equ 0 (
        echo [OK] Backend đã dừng (PID: %%a)
    )
)
echo.

REM Tìm và kill process trên port 3000 (Frontend)
echo Kiểm tra port 3000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":3000" ^| find "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
    if %errorlevel% equ 0 (
        echo [OK] Đã giải phóng port 3000 (PID: %%a)
    )
)
echo.

REM Đóng các cửa sổ console có tiêu đề DocCleaner
taskkill /FI "WindowTitle eq DocCleaner*" /F 2>nul

echo ========================================
echo     Đã dừng tất cả services!
echo ========================================
echo.
pause
