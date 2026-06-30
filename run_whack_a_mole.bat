@echo off
cd /d "%~dp0"
set TF_ENABLE_ONEDNN_OPTS=0
set MPLCONFIGDIR=%~dp0.matplotlib_cache
".venv312\Scripts\python.exe" -m eyetrax.app.whack_a_mole --filter kalman --timed 60 --dwell 0.35 --mole-ttl 6.0 --mole-radius 120 --music "%~dp0assets\audio\Sun_Drenched_Garden.mp3"
pause
