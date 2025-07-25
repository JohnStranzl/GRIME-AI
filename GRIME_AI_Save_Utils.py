import sys
import json
import os

from configparser import ConfigParser
import configparser


# ======================================================================================================================
# ======================================================================================================================
#
# ======================================================================================================================
# ======================================================================================================================
class GRIME_AI_Save_Utils:
    def __init__(self):
        self.className = "GRIME_AI_Save_Utils"

        self.configFile = self.get_settings_folder()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_models_folder(self):
        models_file_path = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'Models')

        if not os.path.exists(models_file_path):
            os.makedirs(models_file_path)

        return models_file_path


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_settings_folder(self):
        configFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'GRIME-AI', 'Settings')

        if not os.path.exists(configFilePath):
            os.makedirs(configFilePath)

        return configFilePath


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def read_config_file(self):
        configFilePath = self.get_settings_folder()
        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')

        config_object = configparser.ConfigParser()
        with open(configFile, "r") as file_object:
            config_object.read_file(file_object)
            #JES print(config_object.get("Local_Images_Folder"))


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def saveSettings(self):
        config = ConfigParser()

        configFilePath = self.get_settings_folder()
        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')

        with open(configFile, 'w+') as f:
            config.read(configFile)

        szSection = 'ROI'
        config.add_section(szSection)
        config.set(szSection, 'Sky', 'TBD')
        config.set(szSection, 'Grass1', 'TBD')
        config.set(szSection, 'Grass2', 'TBD')
        config.set(szSection, 'Trees1', 'TBD')
        config.set(szSection, 'Trees2', 'TBD')

        szSection = 'FilePaths'
        config.add_section(szSection)

        config.write(f)
        f.close()

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def save_images_folder_path(self, images_folder_path):
        configFilePath = self.get_settings_folder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'Local_Images_Folder'
        config.add_section(szSection)
        config.set(szSection, 'ImagesFilePath', images_folder_path)

        config.write(f)
        f.close()
    '''

    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def USGS_SaveFolderPath(self, USGS_FolderPath):
        configFilePath = self.get_settings_folder()

        config = ConfigParser()

        configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
        f = open(configFile, 'w+')

        config.read(configFile)

        szSection = 'USGS'
        config.add_section(szSection)
        config.set(szSection, 'SaveFilePath', USGS_FolderPath)

        config.write(f)
        f.close()
    '''


    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def USGS_getSaveFolderPath(self):
    
        try:
            configFilePath = self.get_settings_folder()

            config = ConfigParser()

            configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
            f = open(configFile, 'r')

            config.read(configFile)

            szSection = 'USGS'
            USGS_FolderPath = config.get(szSection, 'SaveFilePath')

            f.close()
        except Exception:
            USGS_FolderPath = ""

        return USGS_FolderPath
    '''


    # ------------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------------
    '''
    def NEON_getSaveFolderPath(self):

        try:
            configFilePath = self.get_settings_folder()

            config = ConfigParser()

            configFile = os.path.join(configFilePath, 'GRIME-AI.cfg')
            f = open(configFile, 'r')

            config.read(configFile)

            szSection = 'NEON'
            NEON_FolderPath = config.get(szSection, 'SaveFilePath')

            f.close()
        except Exception:
            NEON_FolderPath = ""

        return NEON_FolderPath
    '''

# ======================================================================================================================
# ======================================================================================================================
#
# ======================================================================================================================
# ======================================================================================================================
class JsonEditor():
    def __init__(self):
        super().__init__()

    '''
    def save_to_json(self):
        data = {f'entry_{i}': edit_line.text() for i, edit_line in enumerate(self.edit_lines)}
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print('Data saved to data.json')
    '''

    def update_json_entry(self, entry_key, new_value):
        try:
            json_file = os.path.join(GRIME_AI_Save_Utils().get_settings_folder(), 'GRIME-AI.json')
            json_file = os.path.normpath(json_file)

            with open(json_file, 'r') as file:
                data = json.load(file)

            if entry_key in data:
                data[entry_key] = new_value
                with open(json_file, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f'Entry {entry_key} updated to {new_value}')
            else:
                print(f'Adding missing entry {entry_key} to GRIME-AI.json')
                self.add_key_value_to_json(entry_key, new_value)

        except FileNotFoundError:
            print('GRIME-AI.json file not found')
        except Exception:
            print(f'Adding missing entry {entry_key} to GRIME-AI.json')
            self.add_key_value_to_json(entry_key, new_value)


    def add_key_value_to_json(self, key, value):
        json_file = os.path.join(GRIME_AI_Save_Utils().get_settings_folder(), 'GRIME-AI.json')

        try:
            # Load the existing data from the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)
        except Exception:
            # If the file doesn't exist, create an empty dictionary
            data = {}

        # Check if the key is already in the data
        if key not in data:
            # Add the key-value pair to the data
            data[key] = value

            # Write the updated data back to the JSON file
            with open(json_file, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Added key '{key}' with value '{value}' to the JSON file.")
        else:
            print(f"Key '{key}' already exists in the JSON file.")


    def getValue(self, key):
        json_file = os.path.join(GRIME_AI_Save_Utils().get_settings_folder(), 'GRIME-AI.json')

        try:
            # Load the existing data from the JSON file
            with open(json_file, 'r') as file:
                data = json.load(file)

            return data[key]
        except Exception:
            # If the file doesn't exist, create an empty dictionary
            data = {}

            return None