@echo off
cd /d "%~dp0"
set TF_ENABLE_ONEDNN_OPTS=0
set MPLCONFIGDIR=%~dp0.matplotlib_cache
".venv312\Scripts\python.exe" -m eyetrax.app.gaze_typing_suite --filter kalman
pause
