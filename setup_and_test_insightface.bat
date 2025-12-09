@echo off
REM Batch script to setup and test InsightFace in voca-insight environment

echo ============================================================
echo VOCA-Lens InsightFace Setup and Test
echo ============================================================

REM Change to project directory
cd /d D:\VOCA-Lens

REM Activate voca-insight environment
echo.
echo Activating voca-insight environment...
call "C:\ProgramData\miniconda3\Scripts\activate.bat" voca-insight

if errorlevel 1 (
    echo ERROR: Failed to activate voca-insight environment
    pause
    exit /b 1
)

REM Install required packages
echo.
echo Installing required packages...
echo This may take 5-10 minutes on first run...
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: torch installation had issues, continuing...
)

pip install insightface
if errorlevel 1 (
    echo ERROR: Failed to install insightface
    pause
    exit /b 1
)

pip install onnxruntime opencv-python scikit-learn matplotlib
if errorlevel 1 (
    echo WARNING: Some packages had installation issues, continuing...
)

REM Run test script
echo.
echo ============================================================
echo Running InsightFace setup verification...
echo ============================================================
python test_insightface_setup.py

if errorlevel 1 (
    echo.
    echo ERROR: Setup verification failed
    pause
    exit /b 1
)

REM If verification passed, run the full encoder test
echo.
echo ============================================================
echo Running full encoder test on test2 video...
echo ============================================================
python face_encoder_insightface.py

echo.
echo ============================================================
echo All tests complete!
echo ============================================================
pause

