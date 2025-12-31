@echo off
REM Batch script to run all components (Windows)

echo ========================================
echo Agentic Compliance Auditor - Start All
echo ========================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Check if Ollama is running
echo.
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama is not running!
    echo Please start Ollama in a separate terminal with: ollama serve
    echo.
    pause
)

REM Start API server in background
echo.
echo Starting API server...
start "Compliance API" cmd /k "cd src\api && python main.py"

REM Wait a few seconds for API to start
timeout /t 5 /nobreak >nul

REM Start Streamlit UI
echo.
echo Starting Streamlit UI...
start "Compliance UI" cmd /k "cd ui && streamlit run streamlit_app.py"

echo.
echo ========================================
echo All components started!
echo API: http://localhost:8000
echo UI: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press Ctrl+C in each window to stop
echo.
pause
