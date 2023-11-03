import brian2 as b2
import numpy as np
from brian2.monitors.spikemonitor import SpikeMonitor
from Model import SNN
from definitions import *
from DataImport import *

def test(snn, times_sample):
    test_mon = SpikeMonitor(snn.group_n)
    snn.net.add(test_mon)
    snn.input_group.set_spikes(list(range(ni)), times_sample * b2.ms, train_period)
    snn.net.run(time_run * b2.ms)
    snn.input_group.set_spikes(list(range(ni)), [30000]*ni * b2.ms, 0 * b2.ms)
    snn.net.run(time_relax * b2.ms)
    snn.net.remove(test_mon)
    return test_mon.count

def CreateClasses(snn):
    (train_left_spike_time, train_right_spike_time, train_central_spike_time) = DataImport(True)

    spike_num_left = np.zeros([nn])
    spike_num_right = np.zeros([nn])
    spike_num_central = np.zeros([nn])
    for i in range(4):
        ind = 0 # or i
        count = test(snn, train_left_spike_time[ind])
        spike_num_left += np.asarray(count)
        count = test(snn, train_right_spike_time[ind])
        spike_num_right += np.asarray(count)
        count = test(snn, train_central_spike_time[ind])
        spike_num_central += np.asarray(count)

    classes = {}
    classes[spike_num_left.argmax()] = "left"
    classes[spike_num_right.argmax()] = "right"
    classes[spike_num_central.argmax()] = "central"

    print(classes)
    print(spike_num_left, spike_num_right, spike_num_central)

    return classes