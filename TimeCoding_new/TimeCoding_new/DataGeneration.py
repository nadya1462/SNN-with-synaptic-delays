import brian2 as b2
import numpy as np
from definitions import *

# Quadratic intensity
# M = 256
# left_intensity = [((k/9)**2)*M for k in range(10)]
# right_intensity = [((k/9)**2)*M for k in range(9,-1,-1)]

# Exponential intensity
# left_intensity = np.array([256, 128, 64, 32, 16, 8, 4, 2, 1, 0]) 
# right_intensity = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])

# Lenear intensity
left_intensity = np.arange(ni - 1, -1, -1) * max_intensity/(ni - 1)
right_intensity = np.arange(ni) * max_intensity/(ni - 1)
central_intensity = np.array([0,2,4,6,8,8,6,4,2,0]) * max_intensity/(ni - 1)

# Normalization of intensities
left_intensity_norm = (left_intensity - min_intensity) / (max_intensity - min_intensity) 
right_intensity_norm = (right_intensity - min_intensity) / (max_intensity - min_intensity) 
central_intensity_norm = (central_intensity - min_intensity) / (max_intensity - min_intensity) 

data_left_spike_time = np.empty(shape = [0, ni])
data_right_spike_time = np.empty(shape = [0, ni])
data_central_spike_time = np.empty(shape = [0, ni])
for _ in range(data_size // 2):
    noise = np.random.randn(ni) * noise_sigma
    data_new_left = np.clip(- (max_spike_time - min_spike_time) * left_intensity_norm + max_spike_time + noise, min_spike_time, max_spike_time)
    data_new_right = np.clip(- (max_spike_time - min_spike_time) * right_intensity_norm + max_spike_time + noise, min_spike_time, max_spike_time)
    data_new_central = np.clip(- (max_spike_time - min_spike_time) * central_intensity_norm + max_spike_time + noise, min_spike_time, max_spike_time)
    data_left_spike_time = np.vstack([data_left_spike_time, data_new_left])
    data_right_spike_time = np.vstack([data_right_spike_time, data_new_right])
    data_central_spike_time = np.vstack([data_central_spike_time, data_new_central])

file_path_left = 'data_left_spike_time.npy'
file_path_right = 'data_right_spike_time.npy'
file_path_central = 'data_central_spike_time.npy'
np.save(file_path_left, data_left_spike_time)
np.save(file_path_right, data_right_spike_time)
np.save(file_path_central, data_central_spike_time)