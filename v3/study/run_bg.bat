@echo off
rem Runs "python <args>" in the study folder, logging to <first arg>.log; started detached via launch.ps1
cd /d C:\Users\maik\pacing_v3\study
set LOG=%~n1.log
C:\Users\maik\pacing_v2\venv\Scripts\python.exe %* > %LOG% 2>&1
echo EXIT %ERRORLEVEL% >> %LOG%
