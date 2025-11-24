#
# Programmatically detect the version of the Chrome web browser installed on the PC.
# Compatible with Windows, Mac, Linux.
# Written in Python.
# Uses native OS detection. Does not require Selenium nor the Chrome web driver.
#

import os
import re
from sys import platform

# ======================================================================================================================
#
# ======================================================================================================================
def extract_version_registry(output):
    try:
        google_version = ''
        for letter in output[output.rindex('DisplayVersion    REG_SZ') + 24:]:
            if letter != '\n':
                google_version += letter
            else:
                break
        return(google_version.strip())
    except TypeError:
        return

# ======================================================================================================================
#
# ======================================================================================================================
def extract_version_folder():
    # Check if the Chrome folder exists in the x32 or x64 Program Files folders.
    for i in range(2):
        path = 'C:\\Program Files' + (' (x86)' if i else '') +'\\Google\\Chrome\\Application'
        if os.path.isdir(path):
            paths = [f.path for f in os.scandir(path) if f.is_dir()]
            for path in paths:
                filename = os.path.basename(path)
                pattern = '\d+\.\d+\.\d+\.\d+'
                match = re.search(pattern, filename)
                if match and match.group():
                    # Found a Chrome version.
                    return match.group(0)

    return None

# ======================================================================================================================
#
# ======================================================================================================================
def get_chrome_version():
    version = None
    install_path = None

    try:
        if platform == "linux" or platform == "linux2":
            # linux
            install_path = "/usr/bin/google-chrome"
        elif platform == "darwin":
            # OS X
            install_path = "/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
        elif platform == "win32":
            # Windows...
            try:
                # Try registry key.
                stream = os.popen('reg query "HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Google Chrome"')
                output = stream.read()
                version = extract_version_registry(output)
            except Exception as ex:
                # Try folder path.
                version = extract_version_folder()
    except Exception as ex:
        print(ex)

    version = os.popen(f"{install_path} --version").read().strip('Google Chrome ').strip() if install_path else version

    return version

# ======================================================================================================================
#
# ======================================================================================================================
def loadChromeDriver():

    #JES
    #return

    # ----------------------------------------------------------------------------------------------------
    # GET THE VERSION OF CHROME INSTALLED ON THE COMPUTER AND THE VERSION OF THE CHROME DRIVER INSTALLED WITH GRIME-AI
    # ----------------------------------------------------------------------------------------------------
    strChromeVersion = get_chrome_version()
    print(strChromeVersion)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')  # Last I checked this was necessary.

    # strChromeExe = os.path.join('C:/Program Files (x86)/GRIME-AI/chromedriver/', strChromeVersion, 'chromedriver.exe')
    # Old driver = webdriver.Chrome(strChromeExe, options=options)  # Optional argument, if not specified will search path.

    options = webdriver.ChromeOptions()

    #webdriver.Chrome.close()

    try:
        try:
            strChromeDriverPath = os.path.join('C:/Program Files (x86)/GRIME-AI/chromedriver')
            driver = webdriver.Chrome(service=Service(ChromeDriverManager(path=strChromeDriverPath).install()), options=options)
        except Exception:
            strChromeExe = os.path.join('C:/Program Files (x86)/GRIME-AI/chromedriver/chromedriver.exe')
            print(strChromeExe)
            # driver = webdriver.Chrome(strChromeExe, options=options)  # Optional argument, if not specified will search path.
            driver = webdriver.Chrome(strChromeExe)  # Optional argument, if not specified will search path.

        strChromeDriverVersion = driver.capabilities['browserVersion']
    except Exception:
        pass

    """
    if strChromeVersion != strChromeDriverVersion:
        msgBox = GRIME_AI_QMessageBox('Chrome Driver Error!', 'Chrome Version: ' + strChromeVersion + '\nChrome Driver Version: ' + strChromeDriverVersion + '\n\nYou can use the software for analyzing data but you cannot download images from the Internet until the Chrome Driver and Chrome Browser versions match.')
        response = msgBox.displayMsgBox()
    else:
        msgBox = GRIME_AI_QMessageBox('Chrome Version', 'Chrome Version: ' + strChromeVersion + '\nChrome Driver Version: ' + strChromeDriverVersion)
        response = msgBox.displayMsgBox()
    """

# ======================================================================================================================
#
# ======================================================================================================================
# def download_chromedriver():
#     def get_latestversion(version):
#         url = 'https://chromedriver.storage.googleapis.com/LATEST_RELEASE_' + str(version)
#         response = requests.get(url)
#         version_number = response.text
#         return version_number
#
#     def download(download_url, driver_binaryname, target_name):
#         # download the zip file using the url built above
#         latest_driver_zip = wget.download(download_url, out='./temp/chromedriver.zip')
#
#         # extract the zip file
#         with zipfile.ZipFile(latest_driver_zip, 'r') as zip_ref:
#             zip_ref.extractall(path='./temp/')  # you can specify the destination folder path here
#         # delete the zip file downloaded above
#         os.remove(latest_driver_zip)
#         os.rename(driver_binaryname, target_name)
#         os.chmod(target_name, 755)
#
#     if os.name == 'nt':
#         replies = os.popen(r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version').read()
#         replies = replies.split('\n')
#         for reply in replies:
#             if 'version' in reply:
#                 reply = reply.rstrip()
#                 reply = reply.lstrip()
#                 tokens = re.split(r"\s+", reply)
#                 fullversion = tokens[len(tokens) - 1]
#                 tokens = fullversion.split('.')
#                 version = tokens[0]
#                 break
#         target_name = './bin/chromedriver-win-' + version + '.exe'
#         found = os.path.exists(target_name)
#         if not found:
#             version_number = get_latestversion(version)
#             # build the donwload url
#             download_url = "https://chromedriver.storage.googleapis.com/" + version_number + "/chromedriver_win32.zip"
#             download(download_url, './temp/chromedriver.exe', target_name)
#
#     elif os.name == 'posix':
#         reply = os.popen(r'chromium --version').read()
#
#         if reply != '':
#             reply = reply.rstrip()
#             reply = reply.lstrip()
#             tokens = re.split(r"\s+", reply)
#             fullversion = tokens[1]
#             tokens = fullversion.split('.')
#             version = tokens[0]
#         else:
#             reply = os.popen(r'google-chrome --version').read()
#             reply = reply.rstrip()
#             reply = reply.lstrip()
#             tokens = re.split(r"\s+", reply)
#             fullversion = tokens[2]
#             tokens = fullversion.split('.')
#             version = tokens[0]
#
#         target_name = './bin/chromedriver-linux-' + version
#         print('new chrome driver at ' + target_name)
#         found = os.path.exists(target_name)
#         if not found:
#             version_number = get_latestversion(version)
#             download_url = "https://chromedriver.storage.googleapis.com/" + version_number + "/chromedriver_linux64.zip"
#             download(download_url, './temp/chromedriver', target_name)

