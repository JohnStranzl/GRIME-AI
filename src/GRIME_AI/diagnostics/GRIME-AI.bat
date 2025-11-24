@echo off
set PYTHONVERBOSE=1
REM Run the executable, redirecting both stdout and stderr to debug_log.txt
GRIME-AI.exe > GRIME_AI_debug_log.txt 2>&1
pause
