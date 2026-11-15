@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    py -3.11 -m venv venv
    if errorlevel 1 exit /b %errorlevel%
)

"venv\Scripts\python.exe" -c "import streamlit, cv2, mediapipe, ultralytics, google.genai, dotenv, PIL, numpy"
if errorlevel 1 (
    echo Installing project dependencies into venv...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 exit /b %errorlevel%
)

"venv\Scripts\python.exe" -m streamlit run app.py
