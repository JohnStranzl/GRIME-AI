@echo off

(
echo.
echo **************************************************************
echo * NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE:         *
echo * Checkpoint files for SAM2 and YOLO11 models *must* be      *
echo * downloaded before running GRIME-AI. Please activate the    *
echo * environment and run the commands                           *
echo * 'download-sam2-checkpoints' followed by                    *
echo * 'download-yolo-models' prior to using GRIME-AI. These      *
echo * commands will fetch the needed files and save them to the  *
echo * proper locations.                                          *
echo **************************************************************
echo.
) > %PREFIX%\.messages.txt
