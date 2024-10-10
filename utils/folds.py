"""
# Author = ruben
# Date: 10/10/24
# Project: ml-vit-events
# File: folds.py

Description: Creation of k static folds
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.filters import filter_data_by_setting


def all_class_represented(test_df: pd.DataFrame, config: dict) -> bool:
    """
    Check if all classes are represented in a given test dataframe.

    Parameters
    ----------
    test_df : pd.DataFrame
        The dataframe to check
    config : dict
        A configuration dictionary, which should contain a 'settings' key with a list of settings
        to split the data by.

    Returns
    -------
    class_represented : bool
        True if all classes are represented in the test set, False otherwise
    """
    class_represented = True
    for setting in config["settings"]:
        df_setting = filter_data_by_setting(test_df, setting)
        n_control = len(df_setting[df_setting["event"] == 0])
        n_event = len(df_setting[df_setting["event"] == 1])
        if n_control == 0 or n_event == 0:
            class_represented = False
            break
    return class_represented


def get_fold_split(df: pd.DataFrame, patient_classes: pd.DataFrame, config: dict) -> (pd.DataFrame, pd.DataFrame):
    """
    Get a single train/test split from a given dataframe df, respecting patient strata.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which to split the data
    patient_classes : pd.DataFrame
        A series containing the event labels for each patient
    config : dict
        A configuration dictionary, which should contain a 'parameters' key with a 'test_size' value.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        The train and test dataframes, split according to the given parameters.

    Notes
    -----
    This function will keep trying to split the data until each class is represented at least once in the test set.
    """
    while True:
        train_patients, test_patients = train_test_split(patient_classes.index,
                                                         stratify=patient_classes,
                                                         shuffle=True,
                                                         test_size=config['parameters']['test_size'])
        assert not set(list(train_patients)).intersection(list(test_patients)), \
            f"train and test split wrong: sharing patients!"

        # Get train/test splits assuring patients are in different groups
        train_df = df[df['pid'].isin(train_patients)]
        test_df = df[df['pid'].isin(test_patients)]

        if all_class_represented(test_df, config):
            break

    return train_df, test_df


def make_folds(df: pd.DataFrame, config: dict, n_folds: int) -> None:
    """
    Creates n_folds number of train/test splits from a given dataframe df. The splits are stratified,
    meaning that the same proportion of positive and negative samples will be present in each split.

    Parameters:
    df (pd.DataFrame): The dataframe from which to create the folds
    config (dict): the configuration dictionary, which should contain the path to save the folds
    n_folds (int): the number of folds to create
    """
    patient_classes = df.groupby('pid')['event'].first()
    base_folder = config["dataset"]["folds"]
    hashes = []
    for fold_id in range(1, n_folds + 1):
        train_df, test_df = get_fold_split(df, patient_classes, config)

        # Check uniqueness of train and test, no patients are in both train and test
        unique_hash_train = pd.util.hash_pandas_object(train_df, index=True).sum()
        unique_hash_test = pd.util.hash_pandas_object(test_df, index=True).sum()
        assert unique_hash_test not in hashes
        assert unique_hash_train not in hashes
        hashes.append(unique_hash_train)
        hashes.append(unique_hash_test)

        # save csv
        train_df.to_csv(f'{base_folder}/train_{fold_id}.csv', index=False)
        test_df.to_csv(f'{base_folder}/test_{fold_id}.csv', index=False)
