"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: classifiers.py

Description: Defines ML classifiers
"""
from abc import ABC, abstractmethod

import numpy as np
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


class MLClassifier(ABC):
    def __init__(self, **kwargs):
        self.model = self._create_model(**kwargs)

    @abstractmethod
    def _create_model(self, **kwargs):
        pass

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, predictions) * 100.0,
            'recall': recall_score(y, predictions, zero_division=0) * 100.0,
            'precision': precision_score(y, predictions, zero_division=0) * 100.0,
            'f1': f1_score(y, predictions, zero_division=0) * 100.0,
            'confusion_matrix': confusion_matrix(y, predictions)
        }
        return metrics


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
