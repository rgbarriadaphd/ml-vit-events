"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: predict_agent.py

Description: "Enter description here"
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from settings.settings_loader import Settings

from classification.classifiers import  SVM, KNN, LogisticRegression, DecisionTree, GaussianProcess, RandomForest, AdaBoost, \
    QuadraticClassifier, NaiveBayes, MLP


classifier_mapping = {
    "logistic_regression": LogisticRegression,
    "svm": SVM,
    "knn": KNN,
    "gaussian_process":GaussianProcess,
    "decision_tree":DecisionTree,
    "random_forest":RandomForest,
    "ada_boost":AdaBoost,
    "quadratic_classifier": QuadraticClassifier,
    "naive_bayes":NaiveBayes,
    "mlp": MLP
}


def confidence_interval(data, confidence=0.9):
    mean = np.mean(data)
    n = len(data)
    std_err = stats.sem(data)  # error estándar
    t_critical = stats.t.ppf((1 + confidence) / 2., n - 1)  # valor crítico de t

    # Cálculo del intervalo de confianza
    ci_lower = mean - t_critical * std_err
    ci_upper = mean + t_critical * std_err
    return (ci_lower, ci_upper)


# Función para formatear los resultados
def format_results(mean, std, ci_lower, ci_upper):
    return f"{mean:.2f} ± {std:.2f} | ({ci_lower:.2f}, {ci_upper:.2f})"


class PredictAgent:
    def __init__(self, settings: Settings, dataframe: pd.DataFrame):
        self.settings = settings.get_global_settings()
        self.dataframe = dataframe
        self.run_classifiers = {}

        self.results_df = pd.DataFrame(columns=[
            'fold', 'classifier', 'setting', 'accuracy',
            'recall', 'precision', 'f1', 'confusion_matrix'
        ])

        # Obtain unique patients and classes to stratify split
        self.patient_classes = self.dataframe.groupby('pid')['event'].first()


    def get_results_table(self):
        self.results_df[['train_control', 'train_event', 'test_control', 'test_event']] = self.results_df[
            ['train_control', 'train_event', 'test_control', 'test_event']
        ].astype(int)

        grouped_df = self.results_df.groupby(['classifier', 'setting']).agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            accuracy_ci_lower=('accuracy', lambda x: confidence_interval(x)[0]),
            accuracy_ci_upper=('accuracy', lambda x: confidence_interval(x)[1]),
            recall_mean=('recall', 'mean'),
            recall_std=('recall', 'std'),
            recall_ci_lower=('recall', lambda x: confidence_interval(x)[0]),
            recall_ci_upper=('recall', lambda x: confidence_interval(x)[1]),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            precision_ci_lower=('precision', lambda x: confidence_interval(x)[0]),
            precision_ci_upper=('precision', lambda x: confidence_interval(x)[1]),
            f1_mean=('f1', 'mean'),
            f1_std=('f1', 'std'),
            f1_ci_lower=('f1', lambda x: confidence_interval(x)[0]),
            f1_ci_upper=('f1', lambda x: confidence_interval(x)[1]),
            train_control=('train_control', 'first'),
            train_event=('train_event', 'first'),
            test_control=('test_control', 'first'),
            test_event=('test_event', 'first')
        ).reset_index()

        # Aplicar el formato a cada métrica
        grouped_df['accuracy'] = grouped_df.apply(
            lambda row: format_results(row['accuracy_mean'], row['accuracy_std'], row['accuracy_ci_lower'],
                                       row['accuracy_ci_upper']), axis=1
        )
        grouped_df['recall'] = grouped_df.apply(
            lambda row: format_results(row['recall_mean'], row['recall_std'], row['recall_ci_lower'],
                                       row['recall_ci_upper']), axis=1
        )
        grouped_df['precision'] = grouped_df.apply(
            lambda row: format_results(row['precision_mean'], row['precision_std'], row['precision_ci_lower'],
                                       row['precision_ci_upper']), axis=1
        )
        grouped_df['f1'] = grouped_df.apply(
            lambda row: format_results(row['f1_mean'], row['f1_std'], row['f1_ci_lower'], row['f1_ci_upper']), axis=1
        )

        # Eliminar las columnas intermedias que ya no son necesarias
        grouped_df = grouped_df.drop(columns=[
            'accuracy_mean', 'accuracy_std', 'accuracy_ci_lower', 'accuracy_ci_upper',
            'recall_mean', 'recall_std', 'recall_ci_lower', 'recall_ci_upper',
            'precision_mean', 'precision_std', 'precision_ci_lower', 'precision_ci_upper',
            'f1_mean', 'f1_std', 'f1_ci_lower', 'f1_ci_upper'
        ])

        # Mostrar el DataFrame formateado
        return grouped_df


    def save_results(self):
        experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + self.settings['experiment'].format(balanced=self.settings['parameters']['balance_classes'])
        output_dir = os.path.join('output', experiment_id)
        os.makedirs(output_dir, exist_ok=True)

        self.results_df[['accuracy', 'recall', 'precision', 'f1']] = self.results_df[
            ['accuracy', 'recall', 'precision', 'f1']].round(2)

        csv_file_path = os.path.join(output_dir, 'results.csv')
        self.results_df.to_csv(csv_file_path, index=False)

        config_file_path = os.path.join(output_dir, 'config.json')
        with open(config_file_path, 'w') as json_file:
            json.dump(self.settings, json_file)

        # df table
        results_tablecsv_file_path = os.path.join(output_dir, 'results_table.csv')
        df_results_table = self.get_results_table()
        df_results_table.to_csv(results_tablecsv_file_path, index=False)


        # Latex results
        latex_file_path = os.path.join(output_dir, 'results_table.tex')
        latex_output = df_results_table.style.to_latex()
        with open(latex_file_path, 'w') as f:
            f.write(latex_output)




    def get_fold_split(self):
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

            if self.all_class_represented(test_df):
                break



        return train_df, test_df

    def all_class_represented(self, test_df):
        class_represented = True
        for setting in self.settings["settings"]:
            df_setting = self.filter_data_by_setting(test_df, setting)
            n_control = len(df_setting[df_setting["event"] == 0])
            n_event = len(df_setting[df_setting["event"] == 1])
            if n_control == 0 or n_event == 0:
                print(f'Error: One of de classes is empty: C{n_control}, E{n_event}')
                class_represented = False
                break
        return class_represented

    def filter_data_by_setting(self, df_to_filter, setting):
        if setting == "NO RD":
            return df_to_filter[df_to_filter["rd"] == 0]
        elif setting == "RD":
            return df_to_filter[df_to_filter["rd"] == 1]
        else:
            return df_to_filter

    def get_class_numbers(self, df):
        control = len(df[df["event"] == 0])
        event = len(df[df["event"] == 1])
        total = control + event
        return total, control, event

    def init_classifiers(self):
        for clf_name, clf_params in self.settings['classifiers'].items():
            if clf_params['run']:
                clf_class = classifier_mapping[clf_name]
                clf_args = {k: v for k, v in clf_params.items() if k != 'run'}
                self.run_classifiers[clf_name] = clf_class(**clf_args)

    def run(self):

        self.init_classifiers()
        for fold_id in range(1, self.settings["parameters"]["n_folds"] + 1):
            print(f"............. Executing fold: {fold_id} .............")
            train_df, test_df = self.get_fold_split()
            train_total, train_control, train_event = self.get_class_numbers(train_df)
            x_train = train_df.filter(like='x')
            y_train = train_df['event']

            for clf_name, clf in self.run_classifiers.items():
                # Check for balance classes
                clf_params = clf.get_params()
                balance_configured = self.settings['parameters']['balance_classes']
                class_balanced = False

                if balance_configured:
                    if 'class_weight' not in clf_params:
                        print(f"classifier {clf_name} not configured for class balancing")
                        continue
                    else:
                        print(f"Incorporating class balance to classifier {clf_name}")
                        class_balanced = True
                        clf_params['class_weight'] = 'balanced'
                        clf.set_params(**clf_params)
                clf.fit(x_train, y_train)

                for setting in self.settings["settings"]:
                    # Filter test set by setting
                    df_test_filtered = self.filter_data_by_setting(test_df, setting)
                    test_total, test_control, test_event = self.get_class_numbers(df_test_filtered)

                    x_test = df_test_filtered.filter(like='x')
                    y_test = df_test_filtered['event']
                    self.results_df = clf.evaluate(x_test, y_test, fold_id,
                                                   (train_total, train_control, train_event),
                                                   (test_total, test_control, test_event),
                                                   clf_name, setting, class_balanced,
                                                   self.results_df)
        self.save_results()



