@echo off
rem Start THIS package's patched Ollama server, isolated from any stock Ollama installation:
rem   - runs this folder's ollama.exe (never an "ollama" found on PATH)
rem   - listens on 127.0.0.1:11435 (stock Ollama uses 11434)
rem   - keeps its models and settings in this package's hobbylm-data\ folder
rem   - all settings apply only inside this window; nothing system-wide is changed
rem Usage: start-hobbylm-ollama-server.bat
rem Optional (set in the same window first): HOBBYLM_OLLAMA_PORT (default 11435), HOBBYLM_OLLAMA_DATA (default hobbylm-data)
rem Exit codes: 0 server stopped normally, 3 port already in use, 4 invalid port, 5 data folder not writable,
rem             other = ollama's own exit code.
setlocal
set "PORT=%HOBBYLM_OLLAMA_PORT%"
if "%PORT%"=="" set "PORT=11435"
set "DATA=%HOBBYLM_OLLAMA_DATA%"
if "%DATA%"=="" set "DATA=%~dp0hobbylm-data"
echo %PORT%| findstr /R /X "[0-9][0-9]*" >nul || goto :badport
rem Refuse to start if anything already listens on this port. Nothing is stopped or killed to free it.
netstat -ano -p tcp | findstr /R /C:":%PORT% .*LISTENING" >nul && goto :portbusy
if not exist "%DATA%\models" mkdir "%DATA%\models" 2>nul
if not exist "%DATA%\home" mkdir "%DATA%\home" 2>nul
if not exist "%DATA%\models" goto :nodata
set "OLLAMA_HOST=127.0.0.1:%PORT%"
set "OLLAMA_MODELS=%DATA%\models"
rem Ollama keeps its key and history under the user profile; point it at the package data folder, for this window only.
set "USERPROFILE=%DATA%\home"
echo HobbyLM patched Ollama server: http://127.0.0.1:%PORT%  (this computer only)
echo Models and settings: "%DATA%"
echo Leave this window open. In a second window use hobbylm-ollama.bat. Press Ctrl+C here to stop.
"%~dp0ollama.exe" serve
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo ollama serve exited with code %RC%.
exit /b %RC%

:badport
echo ERROR: HOBBYLM_OLLAMA_PORT must be a number, got "%PORT%".
exit /b 4
:portbusy
echo ERROR: port %PORT% is already in use by another program (possibly another Ollama server).
echo Nothing was changed. Use another port in BOTH windows, for example:
echo   set HOBBYLM_OLLAMA_PORT=11436
exit /b 3
:nodata
echo ERROR: cannot create the data folder "%DATA%".
echo Move the package to a folder you can write to, or set HOBBYLM_OLLAMA_DATA to one.
exit /b 5
