"""
# Author = ruben
# Date: 2/10/24
# Project: ml-vit-events
# File: transformations.py

Description: module to implement datasetset transformations
"""
"""
# Author = ruben
# Date: 25/9/24
# Project: DLPOC
# File: df_transform.py

Description: "Enter description here"
"""
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BaseTransform(ABC):
    def apply(self, df):
        # Get columns not starting with 'x'
        preserve_columns = df.loc[:, ~df.columns.str.startswith('x')]
        # Get columns starting with 'x'
        df_features = df.loc[:, df.columns.str.startswith('x')]

        df_local_transformed = self.local_transformation(df_features)

        return pd.concat([preserve_columns, df_local_transformed], axis=1)


    @abstractmethod
    def local_transformation(self, df):
        pass


class Standardization(BaseTransform):
    def local_transformation(self, df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df),
                            columns=df.columns)


class Normalization(BaseTransform):
    def local_transformation(self, df):
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df),
                            columns=df.columns)

class Outliers(BaseTransform):
    def __init__(self, threshold: float = 10):
        print(f"Outliers threshold: {threshold}")
        self.threshold = threshold

    def local_transformation(self, df):
        # Calculate Z-score for each numeric column
        z_scores = np.abs((df - df.mean()) / df.std())
        # Remove rows where any value is above the outlier threshold
        outliers = df[~((z_scores < self.threshold).all(axis=1))]
        print(f"Removed {len(outliers)} rows because of outliers")
        return df[(z_scores < self.threshold).all(axis=1)]

class Collinear(BaseTransform):
    def local_transformation(self, df):
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        collinear_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
        print(f"Removed {len(collinear_columns)} columns because of collinearity")
        return df.drop(collinear_columns, axis=1)

class Relevant(BaseTransform):
    def __init__(self, n_columns: int = 1024):
        self.n_columns = n_columns
    def local_transformation(self, df):
        relevant_features = df.columns[:self.n_columns]
        print(f"Selected {len(relevant_features)} relevant columns")
        return df[relevant_features]


class DataFrameTransform:
    def __init__(self, arguments):
        self._arguments = arguments
        self._transforms = {
            'standardization': Standardization(),
            'normalization': Normalization(),
            'outliers': Outliers(threshold=3.5),
            'collinear': Collinear(),
            'relevant': Relevant(n_columns=512)
        }

    def apply_transform(self, df):
        for transform, is_active in self._arguments.items():
            if is_active:
                print(f"Applying {transform}...")
                df = self._transforms[transform].apply(df)
        # Delete rows with nan values in columns with name starting in 'x'
        nan_rows = df[df.columns[df.columns.str.startswith('x')]].isna().any(axis=1)
        df = df[~nan_rows]
        print(f"Removed {sum(nan_rows)} rows because of nan values in columns with name starting in 'x'")
        return df




if __name__ == '__main__':
    df_base = pd.read_csv('input_csv/vit_features_clinical.csv')

    arguments = {
        'standardization': True,
        'normalization': True,
        'outliers': True,
        'collinear': True,
        'relevant': True
    }

    dataframe_transform = DataFrameTransform(arguments)
    df_transformed = dataframe_transform.apply_transform(df_base)


