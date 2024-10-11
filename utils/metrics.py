"""
# Author = ruben
# Date: 10/10/24
# Project: ml-vit-events
# File: metrics.py

Description: Module to implement metrics for model evaluation
"""
import math

import numpy as np
from scipy import stats


def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a sample.

    Parameters
    ----------
    data : array_like
        Sample of data to calculate confidence interval from.
    confidence : float, optional
        Confidence level for the interval. Defaults to 0.95.

    Returns
    -------
    ci_lower, ci_upper : tuple
        Lower and upper bounds of confidence interval.

    Notes
    -----
    This function calculates the confidence interval for a sample using
    Student's t-distribution. The confidence level is set to 0.95 by default,
    but can be changed to any value between 0 and 1.
    """
    mean = np.mean(data)
    n = len(data)
    std_err = stats.sem(data)  # standard error
    t_critical = stats.t.ppf((1 + confidence) / 2., n - 1)  # critical value of t for 95% confidence

    ci_lower = mean - t_critical * std_err
    ci_upper = mean + t_critical * std_err
    return ci_lower, ci_upper


def get_metrics(metric_list, confidence=0.95):
    """
    Calculate the mean, standard deviation and confidence interval for a sample.

    Parameters
    ----------
    metric_list : array_like
        Sample of metrics to calculate the mean, standard deviation and confidence interval from.
    confidence : float, optional
        Confidence level for the interval. Defaults to 0.95.

    Returns
    -------
    mean, std, (ci_lower, ci_upper) : tuple
        Mean, standard deviation and confidence interval of the sample.

    Notes
    -----
    This function calculates the mean, standard deviation and confidence interval
    of a sample using the Student's t-distribution or the normal distribution,
    depending on whether the sample is normally distributed or not.
    If the sample size is less than 30 or the p-value of the Shapiro-Wilk test
    is less than 0.05, the Student's t-distribution is used. Otherwise,
    the normal distribution is used.
    """
    n = len(metric_list)
    mean = np.mean(metric_list)
    std = np.std(metric_list, ddof=1)
    std_err = std / np.sqrt(n)

    # Verify if the distribution is normal
    if n > 30 and stats.shapiro(metric_list).pvalue > 0.05:
        z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
        ci = z_value * std_err
    else:
        t_value = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        ci = t_value * std_err

    return mean, std, (mean - ci, mean + ci)


def check_metrics(metrics: dict, total_test: int):
    """
    Manually Check the computed metrics against the true values.

    This function checks if the computed metrics (accuracy, precision, recall) from a
    classification model match the expected values based on the confusion matrix.

    It ensures that:

        - The total test size matches the sum of true positives (tp), true negatives (tn), false positives (fp),
            and false negatives (fn).
        - The computed accuracy, precision, and recall values are close to the expected values calculated from
            the confusion matrix, within a small tolerance.
        - The function raises an AssertionError if any of these checks fail, indicating an error in computing
            the metrics.
    """
    tn, fp, fn, tp = np.array(metrics['confusion_matrix']).ravel()
    assert total_test == (tp + tn + fp + fn), f'Error: wrong metrics size and test set size!'
    assert math.isclose(metrics['accuracy'] / 100.0, (tp + tn) / (tp + tn + fp + fn), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing accuracy!'
    assert math.isclose(metrics['precision'] / 100.0, tp / (tp + fp), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing precision!'
    assert math.isclose(metrics['recall'] / 100.0, tp / (tp + fn), rel_tol=1e-9, abs_tol=1e-11), \
        f'Error computing recall!'