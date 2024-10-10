"""
# Author = ruben
# Date: 10/10/24
# Project: ml-vit-events
# File: latex.py

Description: Utilities to convert to LaTeX specific formats
"""

classifiers_nice_name = {
    "logistic_regression": "LR",
    "knn": "KNN",
    "svm": "SVM",
    "gaussian_process": "GP",
    "decision_tree": "DT",
    "random_forest": "RF",
    "ada_boost": "AB",
    "quadratic_classifier": "QC",
    "naive_bayes": "NB",
    "mlp": "MLP"
}


def get_line(clf_name,  df):
    """
    Return a LaTeX line for given classifier name and data frame.

    Parameters
    ----------
    clf_name : str
        The name of the classifier.
    df : pd.DataFrame
        The data frame containing the results.

    Returns
    -------
    line : str
        A LaTeX line that can be used in a table.
    """
    accuracy = df['accuracy'].iloc[0].replace('±', '\pm')
    recall = df['recall'].iloc[0].replace('±', '\pm')
    precision = df['precision'].iloc[0].replace('±', '\pm')

    line = f"\\textbf{{{classifiers_nice_name[clf_name]}}} &"
    line += f"${accuracy}$ &  ${recall}$ & ${precision}$ \\\\"

    return line


def get_latex_format(df, setting, config):
    """
    Return a LaTeX table for given data frame, setting and configuration.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the results.
    setting : str
        The setting to use for filtering the data frame.
    config : dict
        The configuration containing the classifiers and methods to use.

    Returns
    -------
    table : str
        A LaTeX table that can be used in a document.
    """
    total = ''
    for clf in config["classifiers"].keys():
        if not config["classifiers"][clf]['run']:
            continue
        lines = ''

        filtered_df = df[(df['setting'] == setting) &
                         (df['classifier'] == clf)]

        if filtered_df.empty:
            continue

        lines += get_line(clf, filtered_df)
        lines += '\n'

        lines += '\hline'
        total += lines
    return total
