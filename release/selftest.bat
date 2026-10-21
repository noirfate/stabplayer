@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a video file onto selftest.bat to run the self-test.
  pause
  exit /b 1
)
echo Running self-test. It plays about 0.5 s of audio through the default sound card...
start "" /wait stabplayer.exe --selftest "%~1"
echo.
echo Done. Send stabplayer.log from this folder to the developer.
pause
