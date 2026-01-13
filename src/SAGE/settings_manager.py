# SAGE/settings_manager.py
import json
import os
from pathlib import Path


class SettingsManager:
    """Manage application settings persistence"""

    def __init__(self, settings_file="sage_settings.json"):
        self.settings_file = Path(settings_file)
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load settings from JSON file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading settings: {e}")
                return {}
        return {}

    def _save_settings(self):
        """Save settings to JSON file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get a setting value"""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self._save_settings()

    def get_folder_path(self):
        """Get the last used folder path"""
        return self.get("folder_path", "")

    def set_folder_path(self, path):
        """Set the folder path"""
        self.set("folder_path", path)