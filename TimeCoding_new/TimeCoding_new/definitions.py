import brian2 as b2

min_intensity = 0
max_intensity = 100
data_size = 100
noise_sigma = 0.1 # standard deviation
min_spike_time = 1 # ms
max_spike_time = 10 # ms    # tau_n << max_spike_time / ni  !!! (train = max_spike_time)

time_run = 150
time_relax = 60

n_train = 10
n_test = 30

train_period = 30 * b2.ms  # we will have 7 periods for one image (350 ms) = 7 output spikes
nn = 3
ni = 10

file_path_left = 'data_left_spike_time.npy'
file_path_right = 'data_right_spike_time.npy'
file_path_central = 'data_central_spike_time.npy'

file_path_shifts = 'axons_shifts.npy'
file_path_initial_shifts = 'axons_initial_shifts.npy'


