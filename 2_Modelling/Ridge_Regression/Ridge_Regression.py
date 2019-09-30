'''
Creates a Ridge regression model based on the train_one_hot.csv data.
'''
# system libraries
import os
import sys
import csv
import json
import imp
from pprint import pprint

import pandas as pd
import numpy as np

# sk-learm model imports
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# self-written
sys.path.insert(1, os.path.join('..', '..', 'utilities'))
from util import RegressionMetricsRecorder, update_progress_bar, write_intersection_pred

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

def main():
    train_csv = pd.read_csv(
        os.path.join('..', '..', 'Data', 'train_one_hot.csv'),
        header = 0,
        quoting = csv.QUOTE_ALL
    ).sample(frac = 1.0, replace = True) # shuffle

    frame_y = train_csv[YCOLS]
    frame_X = train_csv[XCOLS]

    train_X, val_X, train_y, val_y = train_test_split(
        frame_X, 
        frame_y, 
        test_size = 0.33, 
        random_state = SEED
    )

    #################################
    # BEGIN VALIDATION SET APPROACH #
    #################################

    pred_frame = pd.DataFrame()

    recorders = [RegressionMetricsRecorder() for _ in YCOLS]
    for y_col_num in range(len(YCOLS)):
        update_progress_bar(y_col_num, len(YCOLS))
        y_col = YCOLS[y_col_num]

        lin_reg = Ridge(random_state = SEED)

        lin_reg.fit(train_X, train_y[y_col])
        pred_y = lin_reg.predict(val_X)

        pred_frame[y_col] = pred_y

        recorders[y_col_num].add_metric(
            val_y[y_col].values,
            pred_y
        )

    # Write metrics to jsons
    for i in range(len(recorders)):
        print(YCOLS[i])
        pprint(recorders[i].mean_dict())
        print()

        try:
            with open(os.path.join('Evaluation', YCOLS[i] + '.json'), 'w') as file:
                json.dump(recorders[i].mean_dict(), file)
        except Exception as e:
            print('Could not write metrics. ' + str(e))
            continue
    
    ###############################
    # END VALIDATION SET APPROACH #
    #  START PREDICTION TEST SET  #
    ###############################

    # train on complete train data first

    test_csv = pd.read_csv(
        os.path.join('..', '..', 'Data', 'test_one_hot.csv'),
        header = 0,
        quoting = csv.QUOTE_ALL
    )

    test_X = test_csv[XCOLS]

    pred_frame_test = pd.DataFrame()
    for y_col_num in range(len(YCOLS)):
        update_progress_bar(y_col_num, len(YCOLS))
        y_col = YCOLS[y_col_num]

        full_lin_reg = Ridge(random_state = SEED)
        full_lin_reg.fit(frame_X, frame_y[y_col])
        pred_y = full_lin_reg.predict(test_X)

        pred_frame_test[y_col] = pred_y

    
    print(pred_frame_test.shape)
    print(pred_frame_test.head(5))
    write_intersection_pred(
        pred_frame_test,
        os.path.join('Unmodified_Linear_Regression.csv')
    )    

if __name__ == '__main__':
    main()