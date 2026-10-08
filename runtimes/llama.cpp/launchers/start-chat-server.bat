@echo off
rem Start the HobbyLM chat server (browser page + OpenAI-style API) on this computer only (127.0.0.1).
rem Usage: start-chat-server.bat "C:\path\to\hobbylm-1b-sft-3450-Q4_K_M.gguf" [port]
rem Exit codes: 0 server stopped normally, 1 usage error, 2 model file not found, 3 port already in use,
rem             4 invalid port, other = llama-server's own exit code.
rem Written without parenthesised blocks so that paths containing spaces or ( ) are handled safely.
setlocal
set "MODEL=%~1"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"
if "%MODEL%"=="" goto :usage
if not exist "%MODEL%" goto :nomodel
echo %PORT%| findstr /R /X "[0-9][0-9]*" >nul || goto :badport
rem Refuse to start if anything already listens on this port. Nothing is stopped or killed to free it.
netstat -ano -p tcp | findstr /R /C:":%PORT% .*LISTENING" >nul && goto :portbusy
echo Starting HobbyLM chat server on http://127.0.0.1:%PORT%/  (this computer only)
echo Open that address in your browser once this window says it is listening. Press Ctrl+C here to stop.
"%~dp0bin\llama-server.exe" -m "%MODEL%" --jinja -c 4096 --host 127.0.0.1 --port %PORT%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo llama-server exited with code %RC%.
exit /b %RC%

:usage
echo Usage: %~nx0 "C:\path\to\hobbylm-1b-sft-3450-Q4_K_M.gguf" [port]
exit /b 1
:nomodel
echo ERROR: model file not found: "%MODEL%"
echo Check the path. Put it in double quotes if it contains spaces.
exit /b 2
:badport
echo ERROR: port must be a number, got "%PORT%".
exit /b 4
:portbusy
echo ERROR: port %PORT% is already in use by another program.
echo Nothing was changed. Start on another port, for example:
echo   %~nx0 "%MODEL%" 8081
exit /b 3
