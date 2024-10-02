"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: loader.py

Description: dataset loader
"""
import pandas as pd

from dataset.transformations import DataFrameTransform
from settings.settings_loader import Settings


class EventsFeaturesDataLoader:

    def __init__(self, settings: Settings):
            self.settings = settings.get_global_settings()

    def get(self):
        df_base = pd.read_csv(self.settings["dataset"]["path"])

        transforms = self.settings["dataset"]["transform"]

        dataframe_transform = DataFrameTransform(transforms)
        df_transform = dataframe_transform.apply_transform(df_base)

        return df_transform
