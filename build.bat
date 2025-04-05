echo on

REM pyi-makespec --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI\venv\Lib\site-packages" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI" main.py
REM "C:\Users\Astrid Haugen\AppData\Local\Programs\Python\Python311\Scripts\pyi-makespec" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI\venv\Lib\site-packages" --path="C:\Users\Astrid Haugen\PycharmProjects\neonAI" main.py

REM pyinstaller -F --debug=all --noupx --noconfirm --hidden-import=hook-sklearnneighborstypedef.py main.py
REM pyinstaller --onefile --noconfirm --hidden-import=hook-sklearnneighborstypedef.py --log-level ERROR main.py
REM "C:\Users\Astrid Haugen\AppData\Local\Programs\Python\Python311\Scripts\pyinstaller" --onefile main.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm
REM pyinstaller setup.py --hidden-import=hook-sklearnneighborstypedef.py --noconfirm

REM pyinstaller --clean --noconfirm --onedir --hidden-import playsound --contents-directory "." --add-data ".\venv\Lib\site-packages\ultralytics\cfg\default.yaml; .\ultralytics\cfg" main.py



REM Generate the .spec file with hidden imports
REM pyi-makespec --onefile --noconfirm --hidden-import=playsound --hidden-import=imageio --hidden-import=imageio_ffmpeg main.py

REM pyinstaller --clean --noconfirm main.spec
@echo on

REM Build the executable with PyInstaller
pyinstaller --clean --noconfirm --onedir ^
--hidden-import=playsound ^
--hidden-import=imageio ^
--hidden-import=imageio_ffmpeg ^
--hidden-import=openpyxl ^
--hidden-import=scikit-image ^
--contents-directory "." main.py

REM Ensure the distribution folder exists
if not exist "dist\main" mkdir "dist\main"

REM
REM Copy the splash screen graphic to the distribution folder
copy "SplashScreen Images\Splash_007.jpg" dist\main
copy "SplashScreen Images\GRIME-AI Logo.jpg" dist\main
copy "SplashScreen Images\GRIME-AI Logo.jpg" dist\main

REM Copy additional required directories and files
if not exist "dist\main\sam2" mkdir "dist\main\sam2"
xcopy "venv\Lib\site-packages\sam2" "dist\main\sam2" /s /e || echo Error copying sam2 folder

if not exist "dist\main\ultralytics" mkdir "dist\main\ultralytics"
if not exist "dist\main\ultralytics\cfg" mkdir "dist\main\ultralytics\cfg"
copy "venv\Lib\site-packages\ultralytics\cfg\default.yaml" "dist\main\ultralytics\cfg" || echo Error copying ultralytics configuration

REM Change the directory to the distribution folder and then rename the executable
cd dist\main
ren main.exe GRIME-AI.exe || echo Error renaming executable

REM Run the executable to validate its functionality
GRIME-AI || echo Error running the program

REM Return to the root directory
cd ..\..\
@echo off
