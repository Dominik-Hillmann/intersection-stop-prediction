'''
Creates a linear regression model based on the train_one_hot.csv data.
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
import matplotlib.pyplot as plt

# sk-learm model imports, Keras imports
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# self-written
sys.path.insert(1, os.path.join('..', '..', 'utilities'))
from util import RegressionMetricsRecorder, update_progress_bar, write_intersection_pred

sys.path.insert(1, os.path.join('..', '..', '1_Preprocessing'))
from One_Hot_Encoding import min_max_scale_col


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
    'Hour_sin', 'Hour_cos',
    'Weekend',
    'Month_sin', 'Hour_cos',
    'EntryHeading_NW', 'EntryHeading_SE', 'EntryHeading_NE', 'EntryHeading_SW', 'EntryHeading_E',
    'EntryHeading_W', 'EntryHeading_S', 'EntryHeading_N', 'ExitHeading_NW', 'ExitHeading_SE', 
    'ExitHeading_NE', 'ExitHeading_SW', 'ExitHeading_E', 'ExitHeading_W', 'ExitHeading_N', 'ExitHeading_S', 
    'City_Atlanta', 'City_Boston', 'City_Chicago', 'City_Philadelphia'
]


def get_model(train_data_shape):
    model = models.Sequential()
    model.add(layers.Dense(
        512, activation = 'relu', 
        input_shape = (train_data_shape[1], )
    ))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(6))
    
    model.compile(
        optimizer = 'rmsprop',
        loss = 'mse',
        metrics = ['mae', 'mse']
    )

    return model


def main():
    train_csv = pd.read_csv(
        os.path.join('..', '..', 'Data', 'train_Transformed.csv'),
        header = 0,
        quoting = csv.QUOTE_ALL
    ).sample(frac = 1.0, replace = True) # shuffle

    frame_y = train_csv[YCOLS]
    frame_X = train_csv[XCOLS]

    train_X, val_X, train_y, val_y = train_test_split(
        frame_X, 
        frame_y, 
        test_size = 0.2, 
        random_state = SEED
    )

    print(train_X.shape, train_y.shape)
    print(val_X.shape, val_y.shape)

    # Rescaling
    for to_be_scaled_col in ['Latitude', 'Longitude']:
        train_scaled = min_max_scale_col(train_X[to_be_scaled_col])
        train_X[to_be_scaled_col] = train_scaled.values.flatten()
        
        val_scaled = min_max_scale_col(val_X[to_be_scaled_col])
        val_X[to_be_scaled_col] = val_scaled.values.flatten()

    ##################
    # Model training # 
    ##################

    model = get_model(train_X.shape)
    num_epochs = 190
    history = model.fit(
        train_X.values,
        train_y.values,
        epochs = num_epochs,
        batch_size = 512,
        validation_data = (val_X.values, val_y.values)
    )  

    plt.plot(range(1, num_epochs + 1), history.history['mse'], 'r', label = 'Train MSE')
    plt.plot(range(1, num_epochs + 1), history.history['val_mse'], 'y', label = 'Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend(loc = 'upper left')
    plt.savefig('mse.png')

    ##########################
    # Write evaluation JSONs #
    ##########################

    pred_y = model.predict(val_X.values)
    print(pred_y.shape, val_y.shape, pred_y[:, 1].shape)

    for col_num in range(val_y.shape[1]):
        target_var_name = YCOLS[col_num]
        metrics_recorder = RegressionMetricsRecorder()
        print(col_num)
        metrics_recorder.add_metric(
            val_y.values[:, col_num], 
            pred_y[:, col_num]
        )

        try:
            with open(os.path.join('Evaluation', target_var_name + '.json'), 'w') as file:
                json.dump(metrics_recorder.mean_dict(), file)
        except Exception as e:
            print('Could not write metrics. ' + str(e))
            continue

    #############################
    # Predictions for test data #
    #############################
    
    test_csv = pd.read_csv(
        os.path.join('..', '..', 'Data', 'test_Transformed.csv'),
        header = 0,
        quoting = csv.QUOTE_ALL
    )
    
    test_X = test_csv[XCOLS]
    for to_be_scaled_col in ['Latitude', 'Longitude']:
        test_scaled = min_max_scale_col(test_X[to_be_scaled_col])
        test_X[to_be_scaled_col] = test_scaled.values.flatten()

    pred_test_y = model.predict(test_X)
    pred_test_y = pd.DataFrame(data = pred_test_y, columns = YCOLS)
    write_intersection_pred(pred_test_y, 'Simple_Neural_Network.csv')

if __name__ == '__main__':
    main()