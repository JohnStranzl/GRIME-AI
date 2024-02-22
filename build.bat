echo on

REM pyi-makespec --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI\venv\Lib\site-packages" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI" main.py
REM "C:\Users\Astrid Haugen\AppData\Local\Programs\Python\Python311\Scripts\pyi-makespec" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI\venv\Lib\site-packages" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI" main.py

REM pyinstaller -F --debug=all --noupx --noconfirm --hidden-import=hook-sklearnneighborstypedef.py main.py
REM pyinstaller --onefile --noconfirm --hidden-import=hook-sklearnneighborstypedef.py --log-level ERROR main.py
REM "C:\Users\Astrid Haugen\AppData\Local\Programs\Python\Python311\Scripts\pyinstaller" --onefile main.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm
REM pyinstaller setup.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm

pyinstaller --clean --noconfirm --onedir --hidden-import playsound --contents-directory "." main.py
REM Copy the splash screen graphic to the distribution folder
copy "SplashScreen Images\Splash_007.jpg" dist\main

REM Change the directory to the distribution folder and then rename the executable
cd dist\main
ren main.exe GRIMe-AI.exe

REM RUN THE PROGRAM TO ENSURE IT EXECUTES
GRIMe-AI

REM cd ..\..\

echo off