'''
Makes the original dataset train.csv suitable for models by applying one-hot encoding to categorical features.
'''

import os
import sys

import pandas as pd
import numpy as np

def one_hottify_column(col):
    """Converts a pandas series to an one-hot encoded dataframe for each unique value in the column.
    
    Arguments:
        col {pandas.Series} -- A pandas series of categorical values,
    """

    re_frame = pd.DataFrame(index = index)

    for unique_value in col.unique():
        re_frame.insert(
            len(re_frame.columns),
            unique_value,
            (col == unique_value).map({
                True: 1,
                False: 0
            }).values
        )

    return re_frame



train_X = pd.read_csv(os.path.join('..', 'data', 'train.csv')).set_index('RowId')
# print(train_X.head(6))
# print(train_X['City']
print(one_hottify_column(train_X['City']))
