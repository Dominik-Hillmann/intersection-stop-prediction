'''
Creates a decision tree regressor based on the train_one_hot.csv data.
'''
# system libraries
import os
import csv
import imp

import pandas as pd
import numpy as np

# sk-learm model imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# config
SEED = 6942
YCOLS = [
    'TotalTimeStopped_p20', 
    'TotalTimeStopped_p50', 
    'TotalTimeStopped_p80', 
    'DistanceToFirstStop_p20', 
    'DistanceToFirstStop_p50', 
    'DistanceToFirstStop_p80'
]
XCOLS = [
    'Latitude',
    'Longitude',
    'Hour',
    'Weekend',
    'Month',
    'EntryHeading_NW', 'EntryHeading_SE', 'EntryHeading_NE', 'EntryHeading_SW', 'EntryHeading_E',
    'EntryHeading_W', 'EntryHeading_S', 'EntryHeading_N', 'ExitHeading_NW', 'ExitHeading_SE', 
    'ExitHeading_NE', 'ExitHeading_SW', 'ExitHeading_E', 'ExitHeading_W', 'ExitHeading_N', 'ExitHeading_S', 
    'City_Atlanta', 'City_Boston', 'City_Chicago', 'City_Philadelphia'
]


train_csv = pd.read_csv(
    os.path.join('..', '..', 'Data', 'train_one_hot.csv'),
    header = 0,
    quoting = csv.QUOTE_ALL
)

frame_y = train_csv[YCOLS]
frame_X = train_csv[XCOLS]

train_X, val_X, train_y, val_y = train_test_split(frame_X, frame_y, test_size = 0.2, random_state = SEED)

tree_reg = DecisionTreeRegressor(random_state = SEED)
tree_reg.fit(train_X, train_y['TotalTimeStopped_p20'])

