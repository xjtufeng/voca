@echo off
REM Direct installation without conda activate
REM Uses the Python executable directly from voca-insight environment

echo ============================================================
echo VOCA-Lens InsightFace Setup and Test (Direct Method)
echo ============================================================

cd /d D:\VOCA-Lens

REM Define paths (with quotes for spaces)
set "CONDA_ENV_PATH=C:\Users\Xiaoyang FENG\.conda\envs\voca-insight"
set "PYTHON_EXE=%CONDA_ENV_PATH%\python.exe"
set "PIP_EXE=%CONDA_ENV_PATH%\Scripts\pip.exe"

REM Check if environment exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at: %PYTHON_EXE%
    echo Please check if voca-insight environment exists.
    pause
    exit /b 1
)

echo Found Python at: %PYTHON_EXE%
echo.

REM Check Python version
echo Checking Python version...
"%PYTHON_EXE%" --version
echo.

REM Install packages using pip directly
echo Installing torch and torchvision...
"%PIP_EXE%" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.

echo Installing insightface...
"%PIP_EXE%" install insightface
echo.

echo Installing other dependencies...
"%PIP_EXE%" install onnxruntime opencv-python scikit-learn matplotlib
echo.

REM Run test scripts
echo ============================================================
echo Running InsightFace setup verification...
echo ============================================================
"%PYTHON_EXE%" test_insightface_setup.py

if errorlevel 1 (
    echo.
    echo ERROR: Setup verification failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Running full encoder test on test2 video...
echo ============================================================
"%PYTHON_EXE%" face_encoder_insightface.py

echo.
echo ============================================================
echo All tests complete!
echo Check output file: test2_visual_embeddings_insightface.npz
echo ============================================================
pause

