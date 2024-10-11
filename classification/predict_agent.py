"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: predict_agent.py

Description: Prediction Agent in charge of running classification
"""
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from settings.settings_loader import Settings
from classification.classifiers import SVM, KNN, LogisticRegression, DecisionTree, GaussianProcess, RandomForest, \
    AdaBoost, \
    QuadraticClassifier, NaiveBayes, MLP
from utils.df_formatting import format_results_table
from utils.filters import filter_data_by_setting
from utils.latex import get_latex_format


class PredictBaseAgent(ABC):
    classifier_mapping = {
        "logistic_regression": LogisticRegression,
        "svm": SVM,
        "knn": KNN,
        "gaussian_process": GaussianProcess,
        "decision_tree": DecisionTree,
        "random_forest": RandomForest,
        "ada_boost": AdaBoost,
        "quadratic_classifier": QuadraticClassifier,
        "naive_bayes": NaiveBayes,
        "mlp": MLP
    }

    def __init__(self, config_path: str) -> None:
        """Constructor for PredictImageFeaturesAgent."""
        self.__config = Settings(config_path).get_global_settings()
        self.__results_df = pd.DataFrame(columns=[
            'fold', 'classifier', 'setting', 'method', 'accuracy',
            'recall', 'precision', 'f1', 'confusion_matrix'
        ])
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + self.__config['experiment']
        self.__output_dir = os.path.join('output', experiment_id)
        os.makedirs(self.__output_dir, exist_ok=True)

    def get_config(self):
        """Provides config settings"""
        return self.__config

    def _get_class_numbers(self, df: pd.DataFrame) -> (int, int, int):
        """Computes the number of total samples, control samples and event samples
        given a dataframe."""
        control = len(df[df["event"] == 0])
        event = len(df[df["event"] == 1])
        total = control + event
        return total, control, event

    def _get_model(self, clf_id: str):
        """Returns an instance of a classifier given its id.

        Returns
        -------
        clf : object
            Classifier instance.

        Notes
        -----
        The classifier's parameters are fetched from the configuration
        dictionary. The 'run' parameter is ignored.

        """
        clf_class = self.classifier_mapping[clf_id]
        clf_params = self.__config['classifiers'][clf_id]
        clf_args = {k: v for k, v in clf_params.items() if k != 'run'}
        return clf_class(**clf_args)

    def _save_results(self) -> None:
        """Saves the results of the experiments to files."""
        self.__results_df[['accuracy', 'recall', 'precision', 'f1']] = self.__results_df[
            ['accuracy', 'recall', 'precision', 'f1']].round(2)

        self.__results_df.to_csv(os.path.join(self.__output_dir, 'results.csv'), index=False)

        # Save config
        with open(os.path.join(self.__output_dir, 'config.json'), 'w') as json_file:
            json.dump(self.__config, json_file)

        # Save formatted results
        df_results_table = format_results_table(self.__results_df)
        df_results_table.to_csv(os.path.join(self.__output_dir, 'results_table.csv'), index=False)

        # Save LaTeX formatted results per setting
        for setting_result in self.__config["settings"]:
            latex_file_path = os.path.join(self.__output_dir, f'results_table_{setting_result}.tex')
            latex_output = get_latex_format(df_results_table, setting_result, self.__config)
            with open(latex_file_path, 'w') as latex_file:
                latex_file.write(latex_output)

    @abstractmethod
    def _retrieve_train_test_data(self, fid: int) -> (pd.DataFrame, pd.DataFrame):
        """Retrieves the train and test data for a given fold id."""
        pass

    @abstractmethod
    def _filter_usable_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Filters usable data for a given fold depending on clinical or images features data"""
        pass

    def _run_fold(self, fid: int) -> None:
        """
        Runs the experiments for a given fold.

        Notes
        -----
        The data is loaded from the input folder. The classifiers are loaded
        from the configuration dictionary. The settings to be experimented are
        loaded from the configuration dictionary as well. The results are
        stored in the results dataframe.
        """
        train_df, test_df = self._retrieve_train_test_data(fid)
        train_total, train_control, train_event = self._get_class_numbers(train_df)

        x_train, y_train = self._filter_usable_data(train_df)

        for clf_name in self.__config["classifiers"].keys():
            if not self.__config["classifiers"][clf_name]['run']:
                continue

            clf = self._get_model(clf_name)
            clf.fit(x_train, y_train)

            for setting in self.__config["settings"]:
                df_test_filtered = filter_data_by_setting(test_df, setting)
                test_total, test_control, test_event = self._get_class_numbers(df_test_filtered)

                x_test, y_test = self._filter_usable_data(df_test_filtered)

                self.__results_df = clf.evaluate(x_test, y_test, fid,
                                                 (train_total, train_control, train_event),
                                                 (test_total, test_control, test_event),
                                                 clf_name, setting,
                                                 self.__results_df)

    def run(self):
        folds = self.__config["parameters"]["n_folds"]
        with tqdm(total=folds, desc=f'{self.__class__.__name__}', unit='img') as pbar:
            for fold_id in range(1, folds + 1):
                self._run_fold(fold_id)
                pbar.update(1)
        self._save_results()


class PredictMultiModalAgent(PredictBaseAgent):
    pass


class PredictClinicalDataAgent(PredictBaseAgent):
    def __init__(self, config_path: str):
        # Get protected config from base class
        super().__init__(config_path)
        self.__config = super().get_config()

    def _clean_for_clinical_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe with both ViT features and clinical data, keep only
        the clinical data, remove duplicate rows (keep just one row per patient)
        and delete columns not relevant in the clinical case.

        Also, remove columns with missing values, cast categorical variables to
        integers and scale numerical variables.
        """
        # Remove ViT features and keep just clinical data
        df_base_clinical = df[[col for col in df.columns if not col.startswith('x')]]

        # The dataset contains several images for each patient, but the clinical data is the same.
        # Keep just one row per patient and delete image column (id) which is not relevant in the clinical case.
        df_base_clinical = df_base_clinical.drop_duplicates(subset='pid', keep='first')
        df_base_clinical = df_base_clinical.drop(columns='id')

        # We know some variables are not relevant since does not provide differentiation information. Remove them
        df_base_clinical = df_base_clinical.drop(self.__config["dataset"]["clinical"]["non_relevant"], axis=1)

        # Remove columns with missing values
        df_base_clinical = df_base_clinical.dropna()

        # For numerical sake, cast categorical variables to integers
        df_cleaned = df_base_clinical.copy()
        df_cleaned[self.__config["dataset"]["clinical"]["relevant"]] = \
            df_cleaned[self.__config["dataset"]["clinical"]["relevant"]].astype(int)

        # Scale numerical variable. Age is the only case in this dataset.
        scaler = MinMaxScaler()
        df_cleaned[self.__config["dataset"]["clinical"]["numerical"]] = scaler.fit_transform(
            df_cleaned[self.__config["dataset"]["clinical"]["numerical"]])

        return df_cleaned

    def _filter_usable_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Filter usable clinical data variables from the given dataframe."""
        x = df[self.__config["dataset"]["clinical"]["trainable"]]
        y = df[self.__config["dataset"]["target"]]
        return x, y

    def _retrieve_train_test_data(self, fid: int) -> (pd.DataFrame, pd.DataFrame):
        """Retrieves the train and test data for a given fold id."""

        # Load train and test data
        train_df = pd.read_csv(os.path.join(self.__config["dataset"]["folds"], f'train_{fid}.csv'))
        test_df = pd.read_csv(os.path.join(self.__config["dataset"]["folds"], f'test_{fid}.csv'))

        train_df = self._clean_for_clinical_process(train_df)
        test_df = self._clean_for_clinical_process(test_df)

        return train_df, test_df


class PredictImageFeaturesAgent(PredictBaseAgent):
    def __init__(self, config_path: str):
        # Get protected config from base class
        super().__init__(config_path)
        self.__config = super().get_config()

    def _filter_usable_data(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Filter usable features from the data from the given dataframe."""
        x = df.filter(like='x')
        y = df[self.__config["dataset"]["target"]]
        return x, y

    def _retrieve_train_test_data(self, fid: int) -> (pd.DataFrame, pd.DataFrame):
        """Retrieves the train and test data for a given fold id."""
        # Get protected config from base class
        config = super().get_config()

        # Load train and test data
        train_df = pd.read_csv(os.path.join(config["dataset"]["folds"], f'train_{fid}.csv'))
        test_df = pd.read_csv(os.path.join(config["dataset"]["folds"], f'test_{fid}.csv'))

        return train_df, test_df


class PredictAgentFactory:
    @staticmethod
    def get_agent(mode, config):
        """Factory method to create a PredictAgent instance based on the given mode."""
        if mode == "features":
            return PredictImageFeaturesAgent(config)
        elif mode == "clinical":
            return PredictClinicalDataAgent(config)
        elif mode == "multimodal":
            return PredictMultiModalAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {mode}")
