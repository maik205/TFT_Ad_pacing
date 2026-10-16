@echo off
setlocal EnableDelayedExpansion
rem Neural forecasts on the CUDA GPU host (PROTOCOL 10.11-10.12). Logs go to %USERPROFILE%\pacing_v2\logs.
rem A run that crashes produces no forecast file and is re-run (max 3 attempts); every attempt is logged.
set PATH=C:\Windows\system32;C:\Windows
cd /d %USERPROFILE%\pacing_v2\code
set PYTHONPATH=src
set PY=%USERPROFILE%\pacing_v2\venv\Scripts\python.exe
set L=%USERPROFILE%\pacing_v2\logs
if not exist %L% mkdir %L%
for %%s in (0 1 2) do (
  for %%m in (TFT NHITS) do (
    call :run %%m %%s
    if errorlevel 1 exit /b 1
  )
)
echo !date! !time! ALL DONE>> %L%\progress.log
exit /b 0

:run
set tries=0
:retry
set /a tries+=1
echo !date! !time! start %1 seed %2 attempt !tries!>> %L%\progress.log
%PY% scripts\02_forecast.py %1 --seed %2 --accelerator gpu > %L%\02_%1_seed%2_attempt!tries!.log 2>&1
if errorlevel 1 (
  echo !date! !time! FAILED %1 seed %2 attempt !tries!>> %L%\progress.log
  if !tries! lss 3 goto retry
  exit /b 1
)
echo !date! !time! done  %1 seed %2 attempt !tries!>> %L%\progress.log
exit /b 0
