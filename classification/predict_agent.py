"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: predict_agent.py

Description: Prediction Agent in charge of running classification
"""
import json
import os
import pandas as pd

from settings.settings_loader import Settings
from classification.classifiers import SVM, KNN, LogisticRegression, DecisionTree, GaussianProcess, RandomForest, \
    AdaBoost, \
    QuadraticClassifier, NaiveBayes, MLP
from utils.df_formatting import format_results_table
from utils.filters import filter_data_by_setting
from utils.latex import get_latex_format


class PredictImageFeaturesAgent:
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
        """
        Constructor for PredictImageFeaturesAgent.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.

        Attributes
        ----------
        __config : dict
            Configuration dictionary.
        __results_df : pandas.DataFrame
            Empty dataframe to store results.
        __output_dir : str
            Path to the output directory, where results will be saved.

        """
        self.__config = Settings(config_path).get_global_settings()
        self.__results_df = pd.DataFrame(columns=[
            'fold', 'classifier', 'setting', 'method', 'accuracy',
            'recall', 'precision', 'f1', 'confusion_matrix'
        ])
        self.__output_dir = os.path.join('output', 'comparison')
        os.makedirs(self.__output_dir, exist_ok=True)

    def _get_class_numbers(self, df: pd.DataFrame) -> (int, int, int):
        """
        Computes the number of total samples, control samples and event samples
        given a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data.

        Returns
        -------
        tuple
            A tuple containing the total number of samples, the number of control
            samples and the number of event samples.

        """
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
        train_df = pd.read_csv(os.path.join(self.__config["dataset"]["folds"], f'train_{fid}.csv'))
        test_df = pd.read_csv(os.path.join(self.__config["dataset"]["folds"], f'test_{fid}.csv'))
        train_total, train_control, train_event = self._get_class_numbers(train_df)

        x_train = train_df.filter(like='x')
        y_train = train_df['event']

        for clf_name in self.__config["classifiers"].keys():
            if not self.__config["classifiers"][clf_name]['run']:
                continue

            clf = self._get_model(clf_name)
            clf.fit(x_train, y_train)

            for setting in self.__config["settings"]:
                df_test_filtered = filter_data_by_setting(test_df, setting)
                test_total, test_control, test_event = self._get_class_numbers(df_test_filtered)

                x_test = df_test_filtered.filter(like='x')
                y_test = df_test_filtered['event']

                self.__results_df = clf.evaluate(x_test, y_test, fid,
                                                 (train_total, train_control, train_event),
                                                 (test_total, test_control, test_event),
                                                 clf_name, setting,
                                                 self.__results_df)

    def run(self):
        folds = self.__config["parameters"]["n_folds"]
        for fold_id in range(1, folds + 1):
            self._run_fold(fold_id)
        self._save_results()
