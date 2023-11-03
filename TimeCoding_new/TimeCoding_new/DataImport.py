import numpy as np
from definitions import *

def DataImport(training = False):
    with open(file_path_left, 'rb') as f_left:
        data_left_spike_time = np.load(f_left)
    with open(file_path_right, 'rb') as f_right:
        data_right_spike_time = np.load(f_right)
    with open(file_path_central, 'rb') as f_central:
        data_central_spike_time = np.load(f_central)

    if training:
        left_spike_time = data_left_spike_time[: n_train // 3]
        right_spike_time = data_right_spike_time[: n_train // 3]
        central_spike_time = data_central_spike_time[: n_train // 3]
    else:
        left_spike_time = data_left_spike_time[- n_test // 3 :]
        right_spike_time = data_right_spike_time[- n_test // 3 :]
        central_spike_time = data_central_spike_time[- n_test // 3 :]
    
    return (left_spike_time, right_spike_time, central_spike_time)
