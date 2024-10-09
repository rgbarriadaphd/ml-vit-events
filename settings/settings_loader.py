"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: settings_loader.py

Description: In charge of load settings defined in JSON configuration file
"""

import json


class Settings:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def get_global_settings(self):
        return self.config

