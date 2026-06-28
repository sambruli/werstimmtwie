@echo off
cd /d "%~dp0"

where python >nul 2>&1 || (echo Python nicht gefunden. Bitte installieren. & pause & exit /b 1)

if not exist ".venv" (
    echo Erstelle virtuelle Umgebung...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installiere Abhaengigkeiten...
pip install -q -r requirements.txt

echo Starte App...
streamlit run app.py --server.headless false

pause
