import sys
import csv
import time

import numpy as np
import pandas as pd

class MetricsRecorder:
    UNEQUAL_LEN_ARRAYS = 'Both array need to have the same lengths to perform the calculation. Please check their dimensions.'

    def __init__(self):
        self.dicts_list = []
    
    
    def add_metric(self, y_true, y_pred):
        self.dicts_list.append(self.evaluate(y_true, y_pred))
        return self
        
        
    def mean_dict(self):
        # To be overridden in child classes.
        pass
    

    def evaluate(self, y_true, y_pred):
        # To be overridden in child classes.
        pass



class RegressionMetricsRecorder(MetricsRecorder):
    METRICS_NAMES = [
        'mse',
        'rmse',
        'mae',
        'r_squ'
    ]

    def __init__(self):
        super(RegressionMetricsRecorder, self).__init__()


    def mean_dict(self):
        collect_dict = {name: [] for name in RegressionMetricsRecorder.METRICS_NAMES}
        for metric in RegressionMetricsRecorder.METRICS_NAMES:
            for collected_dict in self.dicts_list:
                collect_dict[metric].append(collected_dict[metric])

        mean_dict = {}
        for metric in RegressionMetricsRecorder.METRICS_NAMES:
            mean_dict[metric] = float(np.mean(collect_dict[metric]))

        return mean_dict



    def evaluate(self, y_true, y_pred, n_predictors = 1.0):
        new_metrics_dict = {name: None for name in RegressionMetricsRecorder.METRICS_NAMES}
        new_metrics_dict['mse'] = RegressionMetricsRecorder.mse(y_true, y_pred)
        new_metrics_dict['rmse'] = RegressionMetricsRecorder.rmse(y_true, y_pred)
        new_metrics_dict['mae'] = RegressionMetricsRecorder.mae(y_true, y_pred)
        new_metrics_dict['r_squ'] = RegressionMetricsRecorder.r_squ(y_true, y_pred)

        return new_metrics_dict


    @classmethod 
    def mse(cls, y_true, y_pred):
        """Calculates the mean squared error MSE between true values and predictions. Due to the squaring of the deviations, large errors are weighted more.
        
        Arguments:
            y_true {numpy.array} -- Array of true values.
            y_pred {numpy.array} -- Array of predicted values.

        Except:
            {ValueError} -- If arrays of unequal lengths passed.

        Returns:
            mse {float} -- The mean squared error.
        """
        
        if len(y_true) != len(y_pred):
            raise ValueError(cls.UNEQUAL_LEN_ARRAYS)
        
        n = float(len(y_true))
        mse = np.sum(np.square(y_true - y_pred)) / n
        return float(mse)


    @classmethod
    def rmse(cls, y_true, y_pred):
        """Calculates the root mean squared error between the true values and their predictions.
       Arguments:
            y_true {numpy.array} -- Array of true values.
            y_pred {numpy.array} -- Array of predicted values.

        Except:
            {ValueError} -- If arrays of unequal lengths passed.

        Returns:
            rmse {float} -- The root mean squared error.
        """
        rmse = np.sqrt(cls.mse(y_true, y_pred))
        return float(rmse)


    @classmethod
    def mae(cls, y_true, y_pred):
        """Calculates the mean absolute error between the true values and their predictions. This is a linear score meaning it weights all deviations as their true deviation (large deviations do not cause extra penalty).
       
       Arguments:
            y_true {numpy.array} -- Array of true values.
            y_pred {numpy.array} -- Array of predicted values.

        Except:
            {ValueError} -- If arrays of unequal lengths passed.

        Returns:
            rmse {float} -- The mean absolute error.
        """

        if len(y_true) != len(y_pred):
            raise ValueError(cls.UNEQUAL_LEN_ARRAYS)
        
        n = float(len(y_true))
        mae = np.sum(np.abs(y_true - y_pred)) / n
        return float(mae)


    @classmethod
    def r_squ(cls, y_true, y_pred):
        """Calculates R squared between true values and their predictions. R squared is a value between zero and one and describes  the amount of explained variation by the model.
       
       Arguments:
            y_true {numpy.array} -- Array of true values.
            y_pred {numpy.array} -- Array of predicted values.

        Except:
            {ValueError} -- If arrays of unequal lengths passed.

        Returns:
            r_squ {float} -- R squared between zero and one.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(cls.UNEQUAL_LEN_ARRAYS)
        
        y_true_variance = float(np.var(y_true))        
        mse = cls.mse(y_true, y_pred)

        r_squ = 1.0 - (mse / y_true_variance)
        return float(r_squ)



class TwoClassificationMetricsRecorder(MetricsRecorder):
    SHALLOW_METRICS = [
        'true_positive', 
        'true_negative', 
        'false_positive', 
        'false_negative', 
        'precision', 
        'recall', 
        'misclassification_rate'
    ]

    def __init__(self):
        super(TwoClassificationMetricsRecorder, self).__init__()


    def evaluate(self, y_true, y_pred):
        pass

    def mean_dict(self):
        if len(self.dicts_list) == 0:
            return None
        
        new_metrics_dict = {
            "true_positive_CV": [],
            "true_negative_CV": [],
            "false_positive_CV": [],
            "false_negative_CV": [],
            "recall_CV": [],
            "misclassification_rate_CV": [],
            "F0.5_CV": [],
            "F1.0_CV": [],
            "F1.5_CV": []
        }
        
        for metric in TwoClassificationMetricsRecorder.SHALLOW_METRICS:
            new_metrics_dict[metric + '_CV'] = float(np.mean([m[metric] for m in self.dicts_list]))
    
        new_metrics_dict['F0.5_CV'] = float(np.mean([m['F']['scores'][0] for m in self.dicts_list]))
        new_metrics_dict['F1.0_CV'] = float(np.mean([m['F']['scores'][1] for m in self.dicts_list]))
        new_metrics_dict['F1.5_CV'] = float(np.mean([m['F']['scores'][2] for m in self.dicts_list]))
        
        return new_metrics_dict


    @staticmethod
    def f_beta_score(p, r, beta): 
        return (1 + (beta ** 2)) * (p * r) / (((beta ** 2) * p) + r)


def write_intersection_pred(preds_frame, path):
    n_rows, n_cols = preds_frame.shape
    write_shaped = preds_frame.values.reshape((n_rows * n_cols, ))

    written_index = []
    for row in range(n_rows):
        for y_col_num in range(n_cols): # should be six predictions
            written_index.append(str(row) + '_' + str(y_col_num))
    
    pd.DataFrame({
        'TargetId': written_index,
        'Target': write_shaped.tolist()
    }).to_csv(
        path,
        header = True,
        index = False,
        quoting = csv.QUOTE_NONE
    )

#############
# Cosmetics #
#############

def update_progress_bar(progress, total, bar_len = 40, bar_char = '#', status_message = 'Processing'):
    if len(bar_char) != 1:
        raise ValueError('Bar symbol needs to be exactly one character long.')

    num_progessed = round(bar_len * progress / total)
    bar = '{pre_status}: [{filled}{unfilled}] {post_status}\r'.format(
        filled = bar_char * num_progessed,
        unfilled = ' ' * (bar_len - num_progessed),
        pre_status = status_message,
        post_status = str(round(100 * progress / total)) + r'%'
    )
    sys.stdout.write(bar)
    sys.stdout.flush()
