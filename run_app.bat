@echo off
cd /d "%~dp0"
echo Starting FraudGuard AI Streamlit Application...
".\.venv\Scripts\python.exe" -m streamlit run app.py
pause
