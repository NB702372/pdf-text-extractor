@echo off
cd /d "%~dp0"
echo Installing required packages (first run only)...
pip install -q -r requirements.txt
echo.
echo Starting PDF text extraction app...
streamlit run app.py
