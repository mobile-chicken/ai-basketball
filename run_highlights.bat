@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist "input" mkdir "input"
if not exist "output" mkdir "output"
if not exist ".venv\Scripts\python.exe" goto NO_PYTHON
echo Basketball highlight detection is starting.
echo Input folder: %CD%\input
echo Output folder: %CD%\output
echo.
pause
".venv\Scripts\python.exe" -u "run_batch.py" --input "input" --output "output" --rim "auto" --before 4 --after 2
echo.
echo Process finished. ErrorLevel=%ERRORLEVEL%
pause
exit /b
:NO_PYTHON
echo Project Python was not found:
echo %CD%\.venv\Scripts\python.exe
pause
exit /b 1
