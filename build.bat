@echo off
rem Build a portable Windows folder: dist\stabplayer\ (onedir, windowed, no console).
rem Usage: create a venv, pip install -r requirements-build.txt, activate it, then run this script from the repo root.
rem Ship the WHOLE dist\stabplayer folder (including _internal), not just the exe.
python -m PyInstaller --noconfirm --windowed --name stabplayer --distpath dist --workpath build --specpath build --collect-all moderngl --collect-all glcontext stabplayer.py
if errorlevel 1 exit /b 1
rem PyInstaller wipes the output folder on every build, so the user-facing files are copied in afterwards.
copy /y release\README.txt dist\stabplayer\ >nul
copy /y release\selftest.bat dist\stabplayer\ >nul
