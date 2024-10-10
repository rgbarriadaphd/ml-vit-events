"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: classifiers.py

Description: Defines ML classifiers
"""
import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.svm import SVC as SkSVC
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.gaussian_process import GaussianProcessClassifier as SkGaussianProcess
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQuadraticDA
from sklearn.naive_bayes import GaussianNB as SkNaiveBayes
from sklearn.neural_network import MLPClassifier as SkMLP


def check_metrics(metrics: dict, total_test: int):
    """Manually Check the computed metrics against the true values."""
    tn, fp, fn, tp = np.array(metrics['confusion_matrix']).ravel()
    assert total_test == (tp + tn + fp + fn), f'Error: wrong metrics size and test set size!'
    assert math.isclose(metrics['accuracy'] / 100.0, (tp + tn) / (tp + tn + fp + fn), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing accuracy!'
    assert math.isclose(metrics['precision'] / 100.0, tp / (tp + fp), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing precision!'
    assert math.isclose(metrics['recall'] / 100.0, tp / (tp + fn), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing recall!'


class MLClassifier(ABC):
    def __init__(self, **kwargs):
        self.model = self._create_model(**kwargs)

    @abstractmethod
    def _create_model(self, **kwargs):
        pass

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **params):
        return self.model.set_params(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, fold: int,
                 train_counts: tuple[int], test_counts: tuple[int],
                 clf_name: str, setting: str,
                 results_df: pd.DataFrame) -> pd.DataFrame:
        # make model predictions
        predictions = self.predict(X)
        metrics = {
            'fold': fold,
            'classifier': clf_name,
            'setting': setting,
            'train_n': train_counts[0],
            'train_control': train_counts[1],
            'train_event': train_counts[2],
            'test_n': test_counts[0],
            'test_control': test_counts[1],
            'test_event': test_counts[2],
            'accuracy': accuracy_score(y, predictions) * 100.0,
            'recall': recall_score(y, predictions, zero_division=0) * 100.0,
            'precision': precision_score(y, predictions, zero_division=0) * 100.0,
            'f1': f1_score(y, predictions, zero_division=0) * 100.0,
            'confusion_matrix': confusion_matrix(y, predictions)
        }

        # Check the metrics coherency
        check_metrics(metrics, test_counts[0])

        # Include metrics to results dataframe
        results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)

        # Convert bool columns to bool type (avoid warnings)
        results_df = results_df.astype({col: 'bool' for col in results_df.select_dtypes(include='object').columns if
                                        results_df[col].dropna().isin([True, False]).all()})

        return results_df


class LogisticRegression(MLClassifier):
    def _create_model(self, **kwargs):
        return SkLogisticRegression(**kwargs)


class SVM(MLClassifier):
    def _create_model(self, **kwargs):
        return SkSVC(**kwargs)


class KNN(MLClassifier):
    def _create_model(self, **kwargs):
        return SkKNN(**kwargs)


class GaussianProcess(MLClassifier):
    def _create_model(self, **kwargs):
        kernel = 1 ** 2 * RBF(length_scale=2)
        optimizer = kwargs.pop('optimizer', 'fmin_l_bfgs_b')
        return SkGaussianProcess(kernel=kernel, optimizer=optimizer)


class DecisionTree(MLClassifier):
    def _create_model(self, **kwargs):
        return SkDecisionTree(**kwargs)


class RandomForest(MLClassifier):
    def _create_model(self, **kwargs):
        return SkRandomForest(**kwargs)


class AdaBoost(MLClassifier):
    def _create_model(self, **kwargs):
        return SkAdaBoost(**kwargs)


class QuadraticClassifier(MLClassifier):
    def _create_model(self, **kwargs):
        return SkQuadraticDA(**kwargs)


class NaiveBayes(MLClassifier):
    def _create_model(self, **kwargs):
        return SkNaiveBayes(**kwargs)


class MLP(MLClassifier):
    def _create_model(self, **kwargs):
        return SkMLP(**kwargs)
