@echo off
rem Full CTR pipeline on the PC. Usage: run_all.bat  (from v4_ctr; PY points at the ctr conda env)
setlocal
if "%PY%"=="" set PY=C:\Users\maik\miniconda3\envs\ctr\python.exe
set W=%~dp0worlds
set D=%~dp0data
set R=%~dp0runs
if not exist "%W%" mkdir "%W%"
if not exist "%D%" mkdir "%D%"
if not exist "%R%" mkdir "%R%"
pushd %~dp0synthetic
%PY% -m pytest tests -q -p no:cacheprovider --basetemp="%~dp0.pytest_tmp" || exit /b 1
for %%s in (0 1 2) do (
  if not exist "%W%\default_s%%s\observed\synth_meta.json" %PY% -m synth.generate --out "%W%\default_s%%s" --seed %%s || exit /b 1
  if not exist "%W%\gamma0_s%%s\observed\synth_meta.json" %PY% -m synth.generate --out "%W%\gamma0_s%%s" --seed %%s --set gamma=0 || exit /b 1
)
popd
pushd %~dp0
%PY% -m pytest tests -q -p no:cacheprovider --basetemp="%~dp0.pytest_tmp" || exit /b 1
for %%w in (default_s0 default_s1 default_s2 gamma0_s0 gamma0_s1 gamma0_s2) do (
  if not exist "%D%\%%w.npz" %PY% -m ctr.dataset "%W%\%%w" "%D%\%%w.npz" || exit /b 1
)
for %%m in (logreg cadet) do (
  %PY% -m ctr.train --model %%m --train "%D%\default_s0.npz" --eval "%D%\default_s1.npz" "%D%\default_s2.npz" --out "%R%\%%m_default" || exit /b 1
  %PY% -m ctr.train --model %%m --train "%D%\gamma0_s0.npz" --eval "%D%\gamma0_s1.npz" "%D%\gamma0_s2.npz" --out "%R%\%%m_gamma0" || exit /b 1
)
%PY% -m evaluation.ctr_truth "%W%" "%R%\logreg_default" "%R%\cadet_default" "%R%\logreg_gamma0" "%R%\cadet_gamma0" || exit /b 1
%PY% -m ctr.export --run "%R%\cadet_default" --data "%D%\default_s0.npz" --out "%~dp0export\cadet" || exit /b 1
%PY% -m ctr.export --run "%R%\logreg_default" --data "%D%\default_s0.npz" --out "%~dp0export\logreg" || exit /b 1
%PY% -m ctr.bench "%~dp0export\cadet\ctr-cadet.onnx" --threads 1 > "%~dp0export\cadet\bench_cpu1.json" || exit /b 1
%PY% -m ctr.bench "%~dp0export\logreg\ctr-logreg.onnx" --threads 1 > "%~dp0export\logreg\bench_cpu1.json" || exit /b 1
echo PIPELINE_DONE
popd
