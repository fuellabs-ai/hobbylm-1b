@echo off
rem Continue a text with the HobbyLM-1B BASE model (not a chat model): greedy, 40 new tokens, 1024-token context.
rem Usage: run-base-completion.bat "C:\path\to\hobbylm-1b-base-F32.gguf" "text to continue"
rem Exit codes: 0 ok, 1 usage error, 2 model file not found, other = llama-completion's own exit code.
rem Written without parenthesised blocks so that paths containing spaces or ( ) are handled safely.
setlocal
set "MODEL=%~1"
set "TEXT=%~2"
if "%MODEL%"=="" goto :usage
if "%TEXT%"=="" goto :usage
if not exist "%MODEL%" goto :nomodel
"%~dp0bin\llama-completion.exe" -m "%MODEL%" -p "%TEXT%" -n 40 --temp 0 --top-k 1 -no-cnv -c 1024 --no-display-prompt
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo llama-completion exited with code %RC%.
exit /b %RC%

:usage
echo Usage: %~nx0 "C:\path\to\hobbylm-1b-base-F32.gguf" "text to continue"
exit /b 1
:nomodel
echo ERROR: model file not found: "%MODEL%"
echo Check the path. Put it in double quotes if it contains spaces.
exit /b 2
