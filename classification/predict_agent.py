"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: predict_agent.py

Description: "Enter description here"
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from settings.settings_loader import Settings

class PredictAgent:
    def __init__(self, settings: Settings, dataframe: pd.DataFrame):
        self.settings = settings.get_global_settings()
        self.dataframe = dataframe

        # Obtain unique patients and classes to stratify split
        self.patient_classes = self.dataframe.groupby('pid')['event'].first()



    def _get_fold_split(self):
        while True:
            # Split data and check patients in train and test are different always
            train_patients, test_patients = train_test_split(self.patient_classes.index,
                                                             stratify=self.patient_classes,
                                                             shuffle=True,
                                                             test_size=self.settings['parameters']['test_size'])
            assert not set(list(train_patients)).intersection(list(test_patients)), \
                f"train and test split wrong: sharing patients!"

            # Get train/test splits assuring patients are in different groups
            train_df = self.dataframe[self.dataframe['pid'].isin(train_patients)]
            test_df = self.dataframe[self.dataframe['pid'].isin(test_patients)]

            if self._all_class_represented(test_df):
                break

        return train_df, test_df

    def _all_class_represented(self, test_df):
        class_represented = True
        for setting in self.settings["settings"]:
            df_setting = self._filter_data_by_setting(test_df, setting)
            n_control = len(df_setting[df_setting["event"] == 0])
            n_event = len(df_setting[df_setting["event"] == 1])
            if n_control == 0 or n_event == 0:
                print(f'Error: One of de classes is empty: C{n_control}, E{n_event}')
                class_represented = False
                break
        return class_represented

    def _filter_data_by_setting(self, df_to_filter, setting):
        if setting == "NO RD":
            return df_to_filter[df_to_filter["rd"] == 0]
        elif setting == "RD":
            return df_to_filter[df_to_filter["rd"] == 1]
        else:
            return df_to_filter

    def run(self):
        for fold_id in range(1, self.settings["parameters"]["n_folds"] + 1):
            print(f"Executing fold: {fold_id}")

            train_df, test_df = self._get_fold_split()


if __name__ == '__main__':
    pass
