@echo off
REM One-time setup helper for Windows 11 + Python 3.12.
REM Usage: double-click, or run `setup_windows.bat` from a terminal.

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" not found. Install Python 3.12 x64 from
    echo https://www.python.org/downloads/ and re-run this script.
    pause
    exit /b 1
)

py -3.12 -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
python download_models.py

echo.
echo Setup complete. Set your MySQL password, then run the app with:
echo   .venv\Scripts\activate
echo   set DB_PASSWORD=your_mysql_password
echo   python app.py
pause
