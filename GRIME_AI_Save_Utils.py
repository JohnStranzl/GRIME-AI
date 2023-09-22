import os
from configparser import ConfigParser

class GRIME_AI_Save_Utils:
    def __init__(self):
        self.className = "GRIME_AI_Save_Utils"

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def getSettingsFolder(self):
        configFilePath = os.path.expanduser('~')
        configFilePath = os.path.join(configFilePath, 'Documents')
        configFilePath = os.path.join(configFilePath, 'GRIMe-AI')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)
        configFilePath = os.path.join(configFilePath, 'Settings')
        if not os.path.exists(configFilePath):
            os.mkdir(configFilePath)

        return configFilePath

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def saveSettings(self):
        configFilePath = self.getSettingsFolder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'ROI'
        config.add_section(szSection)
        config.set(szSection, 'Sky', 'TBD')
        config.set(szSection, 'Grass1', 'TBD')
        config.set(szSection, 'Grass2', 'TBD')
        config.set(szSection, 'Trees1', 'TBD')
        config.set(szSection, 'Trees2', 'TBD')

        config.write(f)
        f.close()

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def USGS_SaveFolderPath(self, USGS_FolderPath):
        configFilePath = self.getSettingsFolder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'USGS'
        config.add_section(szSection)
        config.set(szSection, 'SaveFilePath', USGS_FolderPath)

        config.write(f)
        f.close()

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def NEON_SaveFolderPath(self, NEON_FolderPath):
        configFilePath = self.getSettingsFolder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'NEON'
        config.add_section(szSection)
        config.set(szSection, 'SaveFilePath', NEON_FolderPath)

        config.write(f)
        f.close()

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def USGS_getSaveFolderPath(self):
        configFilePath = self.getSettingsFolder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'r')

        config.read(configFile)

        try:
            szSection = 'USGS'
            USGS_FolderPath = config.get(szSection, 'SaveFilePath')
        except:
            USGS_FolderPath = ""

        f.close()

        return USGS_FolderPath

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    def NEON_getSaveFolderPath(self):
        configFilePath = self.getSettingsFolder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIMe-AI.cfg')
        f = open(configFile, 'r')

        config.read(configFile)

        try:
            szSection = 'NEON'
            NEON_FolderPath = config.get(szSection, 'SaveFilePath')
        except:
            NEON_FolderPath = ""

        f.close()

        return NEON_FolderPath