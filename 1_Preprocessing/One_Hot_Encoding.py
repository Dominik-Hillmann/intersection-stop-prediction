'''
Makes the original dataset suitable for models by applying one-hot encoding to categorical features and
transforming cyclical features such that they are understood as cyclical by the model.
Example: The 'Hour' feature ranges from 1 to 24. If passed to a model as is, it will understand e.g. 
the values 1 and 24 as the values furthest apart. Yet, in the human understanding of hours, 1 AM follows 12 PM.
These columns will be transformed such that 12 PM and 1 AM are values that are placed close together.
'''

import os
import sys
import csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def one_hottify_column(col):
    """Converts a pandas series to an one-hot encoded dataframe for each unique value in the column.
    
    Arguments:
        col {pandas.Series} -- A pandas series of categorical values.

    Returns:
        re_frame {pandas.DataFrame} -- Dataframe with each unique value as column.
    """

    re_frame = pd.DataFrame(index = col.index)

    for unique_value in col.unique():
        re_frame.insert(
            len(re_frame.columns),
            col.name + '_' + unique_value,
            (col == unique_value).map({
                True: 1,
                False: 0
            }).values
        )

    return re_frame


def column_to_cyclical(col):
    """Converts a Pandas series into two featues reflecting the column's cyclical nature.
    An example would be the hour feature with values ranging from 1 to 24. If seen linearly, the values 1 and 24 are furthest apart whereas they should be very close since 1 AM follows 12 PM.
    More information: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    
    Arguments:
        col {pandas.Series} -- A series with a cyclical feature.
    
    Returns:
        {pandas.DataFrame} -- A dataframe containing the transformations reflect the feature' cyclical nature.
    """

    re_frame = pd.DataFrame(index = col.index)
    max_val = np.max(col.values)
    min_val = np.min(col.values)
    min_not_zero = min_val == 1
    
    re_frame.insert(
        len(re_frame.columns),
        col.name + '_sin',
        np.sin((col.values - (1.0 if min_not_zero else 0.0)) * (2.0 * np.pi / max_val))
    )

    re_frame.insert(
        len(re_frame.columns),
        col.name + '_cos',
        np.cos((col.values - (1.0 if min_not_zero else 0.0)) * (2.0 * np.pi / max_val))
    )

    return re_frame


def min_max_scale_col(col):
    """Converts values to values fitting between 0.0 and 1.0.
    
    Arguments:
        col {pandas.Series} -- A series.
    
    Returns:
        {pandas.DataFrame} -- A dataframe containing a single column with the normalized values.
    """

    re_frame = pd.DataFrame(index = col.index)
    x = col.values.reshape((len(col.values), 1))
    scaled_x = MinMaxScaler().fit_transform(x).flatten()
    
    re_frame.insert(len(re_frame.columns), col.name + '_scaled', scaled_x)
    return re_frame


def main():
    file_name = sys.argv[1]

    train_X = pd.read_csv(os.path.join('..', 'Data', file_name)).set_index('RowId')
    train_X.drop([
        'EntryStreetName', 
        'ExitStreetName', 
        'Path', 
        'IntersectionId'
    ], inplace = True, axis = 1) 

    categorical_cols = [
        'EntryHeading', 
        'ExitHeading',
        'City'
    ]

    cyclical_cols = [
        'Hour',
        'Month'
    ]

    train_X['idx'] = train_X.index
    merged = train_X
    for categorical_col in categorical_cols:
        merged = pd.merge(
            left = merged,
            right = one_hottify_column(merged[categorical_col]),
            left_index = True,
            right_index = True
        )
        merged.drop(categorical_col, inplace = True, axis = 1)

    for cyclical_col in cyclical_cols:
        merged = pd.merge(
            left = merged,
            right = column_to_cyclical(merged[cyclical_col]),
            left_index = True,
            right_index = True
        )
        merged.drop(cyclical_col, inplace = True, axis = 1)

    merged.to_csv(
        os.path.join('..', 'Data', file_name.strip('.csv') + '_Transformed.csv'),
        header = True,
        index = True,
        quoting = csv.QUOTE_ALL
    )


if __name__ == '__main__':
    main()
