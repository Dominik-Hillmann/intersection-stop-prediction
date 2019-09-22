'''
Makes the original dataset train.csv suitable for models by applying one-hot encoding to categorical features.
'''

import os
import sys
import csv

import pandas as pd
import numpy as np

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


def main():
    train_X = pd.read_csv(os.path.join('..', 'Data', 'train.csv')).set_index('RowId')
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

    merged.to_csv(
        os.path.join('..', 'Data', 'train_one_hot.csv'),
        header = True,
        index = True,
        quoting = csv.QUOTE_ALL
    )

if __name__ == '__main__':
    main()
