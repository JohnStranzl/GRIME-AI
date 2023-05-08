echo on

pyi-makespec --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI\venv\Lib\site-packages" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI" main.py

pyinstaller main.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm
REM pyinstaller setup.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm

REM Copy the splash screen graphic to the distribution folder
copy "SplashScreen Images\Splash_007FE.jpg" dist\main

REM Change the directory to the distribution folder and then rename the executable
cd dist\main
ren main.exe GRIMe-AI(FE).exe

REM RUN THE PROGRAM TO ENSURE IT EXECUTES
GRIMe-AI_DownloadManager

REM cd ..\..\

echo off