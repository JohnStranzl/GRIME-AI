@echo off

"/opt/anaconda1anaconda2anaconda3"\python.exe "/opt/anaconda1anaconda2anaconda3"\download-yolo11-weights.py


IF "%~1"=="1" (
    ECHO.
    ECHO This window will close automatically in a few seconds...
    TIMEOUT /T 10
    EXIT
)
