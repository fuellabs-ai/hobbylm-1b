@echo off
rem Run an ollama command against THIS package's patched server only (started by start-hobbylm-ollama-server.bat).
rem It always uses this folder's ollama.exe and 127.0.0.1:11435, so it can never reach a stock Ollama server.
rem Examples:
rem   hobbylm-ollama.bat create hobbylm-1b-sft-q4 -f "C:\My Models\Modelfile.hobbylm-1b-sft-q4"
rem   hobbylm-ollama.bat run hobbylm-1b-sft-q4
rem   hobbylm-ollama.bat list
rem Optional (same values as the server window): HOBBYLM_OLLAMA_PORT, HOBBYLM_OLLAMA_DATA
rem Exit codes: 1 no command given, 3 patched server not running, 4 invalid port, other = ollama's own exit code.
setlocal
if "%~1"=="" goto :usage
set "PORT=%HOBBYLM_OLLAMA_PORT%"
if "%PORT%"=="" set "PORT=11435"
set "DATA=%HOBBYLM_OLLAMA_DATA%"
if "%DATA%"=="" set "DATA=%~dp0hobbylm-data"
echo %PORT%| findstr /R /X "[0-9][0-9]*" >nul || goto :badport
netstat -ano -p tcp | findstr /R /C:"127.0.0.1:%PORT% .*LISTENING" >nul || goto :noserver
set "OLLAMA_HOST=127.0.0.1:%PORT%"
set "USERPROFILE=%DATA%\home"
"%~dp0ollama.exe" %*
exit /b %ERRORLEVEL%

:usage
echo Usage: %~nx0 ^<ollama command^>   e.g.  %~nx0 run hobbylm-1b-sft-q4
exit /b 1
:badport
echo ERROR: HOBBYLM_OLLAMA_PORT must be a number, got "%PORT%".
exit /b 4
:noserver
echo ERROR: the HobbyLM patched Ollama server is not running on 127.0.0.1:%PORT%.
echo Start it first with start-hobbylm-ollama-server.bat (in another window), then try again.
exit /b 3
