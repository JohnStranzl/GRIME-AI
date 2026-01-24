@echo off

"/opt/anaconda1anaconda2anaconda3"\python.exe "/opt/anaconda1anaconda2anaconda3"\download-sam2-checkpoints.py


IF "%~1"=="1" (
    ECHO This window will close automatically in a few seconds...
    TIMEOUT /T 10
    EXIT
)
