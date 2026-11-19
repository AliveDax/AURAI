@echo off
rem Double-click this file to start the art recommender. It opens in your browser by itself.
rem To stop it, close this black window.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo The venv folder is missing, so the app cannot start. See README.md, "Setup".
    pause
    exit /b 1
)

echo Starting the art recommender... your browser will open in a few seconds.
echo The first search takes 10-30 seconds while CLIP loads onto the GPU.
echo To stop the app, close this window.
echo.
"venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.headless false --server.showEmailPrompt false --browser.gatherUsageStats false

rem Only reached if the app stops or crashes: keep the window open so any error can be read
pause
