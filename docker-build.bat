@echo off
REM Docker Helper Script for Employee Attrition Project
REM Gunakan: .\docker-build.bat [username_anda]

setlocal enabledelayedexpansion

if "%1"=="" (
    echo.
    echo ❌ ERROR: Username Docker Hub belum diisi!
    echo.
    echo Penggunaan: docker-build.bat [username_anda]
    echo Contoh: docker-build.bat john123
    echo.
    exit /b 1
)

set DOCKER_USERNAME=%1
set IMAGE_NAME=%DOCKER_USERNAME%/employee-attrition:latest

echo.
echo 🔨 Building Docker image: %IMAGE_NAME%
echo.

docker build -t %IMAGE_NAME% .

if %errorlevel% equ 0 (
    echo.
    echo ✅ Build berhasil!
    echo.
    echo 📤 Push ke Docker Hub? (Optional)
    echo Command: docker push %IMAGE_NAME%
    echo.
    echo 🚀 Test lokal?
    echo Command: docker run -p 5000:5000 -p 8501:8501 %IMAGE_NAME%
    echo.
) else (
    echo.
    echo ❌ Build gagal! Cek error di atas.
    echo.
    exit /b 1
)

endlocal
