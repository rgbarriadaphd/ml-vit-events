"""
# Author = ruben
# Date: 10/10/24
# Project: ml-vit-events
# File: df_formatting.py

Description: Utils to format dataframes
"""
import pandas as pd

from utils.metrics import confidence_interval


def format_metrics(mean: float, std: float, ci_lower: float, ci_upper: float) -> str:
    """Format a metric in a readable way.: mean ± std | (ci_lower, ci_upper)"""
    return f"{mean:.2f} ± {std:.2f} | ({ci_lower:.2f}, {ci_upper:.2f})"


def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a dataframe with results of a prediction task by computing the mean, standard deviation
    and confidence interval of the accuracy, recall, precision and f1 score.

    Parameters
    ----------
    results_df : pandas.DataFrame
        the dataframe containing the results of the prediction task

    Returns
    -------
    formatted_df : pandas.DataFrame
        the formatted dataframe
    """
    results_df[['train_control', 'train_event', 'test_control', 'test_event']] = results_df[
        ['train_control', 'train_event', 'test_control', 'test_event']
    ].astype(int)

    grouped_df = results_df.groupby(['classifier', 'setting']).agg(
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

    # apply formatting to each metric
    grouped_df['accuracy'] = grouped_df.apply(
        lambda row: format_metrics(row['accuracy_mean'], row['accuracy_std'], row['accuracy_ci_lower'],
                                   row['accuracy_ci_upper']), axis=1
    )
    grouped_df['recall'] = grouped_df.apply(
        lambda row: format_metrics(row['recall_mean'], row['recall_std'], row['recall_ci_lower'],
                                   row['recall_ci_upper']), axis=1
    )
    grouped_df['precision'] = grouped_df.apply(
        lambda row: format_metrics(row['precision_mean'], row['precision_std'], row['precision_ci_lower'],
                                   row['precision_ci_upper']), axis=1
    )
    grouped_df['f1'] = grouped_df.apply(
        lambda row: format_metrics(row['f1_mean'], row['f1_std'], row['f1_ci_lower'], row['f1_ci_upper']), axis=1
    )

    # delete intermediate columns
    grouped_df = grouped_df.drop(columns=[
        'accuracy_mean', 'accuracy_std', 'accuracy_ci_lower', 'accuracy_ci_upper',
        'recall_mean', 'recall_std', 'recall_ci_lower', 'recall_ci_upper',
        'precision_mean', 'precision_std', 'precision_ci_lower', 'precision_ci_upper',
        'f1_mean', 'f1_std', 'f1_ci_lower', 'f1_ci_upper'
    ])

    return grouped_df
