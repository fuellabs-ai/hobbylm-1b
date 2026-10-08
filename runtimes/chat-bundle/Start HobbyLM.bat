@echo off
rem HobbyLM-1B chat: double-click to start. No installation, no admin rights, no downloads.
rem Uses the model in .\model\ and the runtime in .\runtime\ (both found relative to this file).
rem If HobbyLM from this folder is already running, opens it instead of starting a second copy.
rem Otherwise picks a free port between 8080 and 8099 on 127.0.0.1 (this computer only); never stops other programs.
rem Opens the chat page in your browser when the server is ready. Close this window to stop HobbyLM.
rem (Written without parenthesised blocks so that folders with spaces or ( ) work.)
setlocal
title HobbyLM-1B chat  -  close this window to stop
set "HERE=%~dp0"
set "MODEL=%HERE%model\hobbylm-1b-sft-3450-Q4_K_M.gguf"
set "SERVER=%HERE%runtime\bin\llama-server.exe"
set "OPENER=%HERE%support\open-browser-when-ready.ps1"
set "FINDER=%HERE%support\find-running-hobbylm.ps1"
if not exist "%SERVER%" goto :noserver
if not exist "%MODEL%" goto :nomodel
rem Same-folder launch guard: if HobbyLM from THIS folder is already running, reuse it instead of starting another copy.
rem Identity = a local server whose /props reports exactly this folder's model file (support\find-running-hobbylm.ps1).
call :findrunning
if defined FOUND goto :alreadyrunning
set "PORT="
for /L %%P in (8080,1,8099) do call :tryport %%P
if not defined PORT goto :noport
set "LOG=%TEMP%\HobbyLM-server-%PORT%.log"
echo.
echo   HobbyLM-1B chat is starting ...
echo.
echo   Your browser will open automatically when it is ready (usually 5-20 seconds).
echo   If it does not, open this address yourself:  http://127.0.0.1:%PORT%/
echo.
echo   Keep this window open while you chat.
echo   To STOP HobbyLM: close this window.
echo.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%OPENER%" -Port %PORT%
"%SERVER%" -m "%MODEL%" --jinja -c 4096 --host 127.0.0.1 --port %PORT% --offline --log-colors off --log-file "%LOG%" >nul 2>&1
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :stopped
echo.
echo   HobbyLM stopped unexpectedly (code %RC%).
echo   Details are in: "%LOG%"
echo.
pause
exit /b %RC%

:stopped
echo   HobbyLM has stopped.
exit /b 0

:findrunning
set "FOUND="
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%FINDER%" -Model "%MODEL%" >nul 2>&1
set "FC=%ERRORLEVEL%"
if %FC% GEQ 8080 if %FC% LEQ 8099 set "FOUND=%FC%"
exit /b 0

:alreadyrunning
title HobbyLM-1B  -  already running, opening the browser
echo.
echo   HobbyLM is already running from this folder (in its other window).
echo   Opening it in your browser:  http://127.0.0.1:%FOUND%/
echo.
echo   No second copy was started. To stop HobbyLM, close the OTHER HobbyLM window.
echo   This window closes by itself in a few seconds.
start "" "http://127.0.0.1:%FOUND%/"
timeout /t 8 /nobreak >nul
exit /b 0

:tryport
if defined PORT exit /b 0
netstat -ano -p tcp | findstr /R /C:":%1 .*LISTENING" >nul && exit /b 0
set "PORT=%1"
exit /b 0

:noserver
echo.
echo   ERROR: the HobbyLM runtime was not found:
echo   "%SERVER%"
echo   Extract the whole ZIP file first (right-click it, then "Extract All..."), then run
echo   "Start HobbyLM.bat" from the extracted folder. Do not run it from inside the ZIP.
echo.
pause
exit /b 2

:nomodel
echo.
echo   ERROR: the model file was not found:
echo   "%MODEL%"
echo   Extract the whole ZIP file again; the "model" folder must stay next to this file.
echo.
pause
exit /b 2

:noport
echo.
echo   ERROR: all ports from 8080 to 8099 are in use by other programs.
echo   HobbyLM did not stop or change any of them. Close some programs and try again.
echo.
pause
exit /b 3
