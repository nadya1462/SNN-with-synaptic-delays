import os
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from Model import SNN
from definitions import *
from DataImport import *

def show_plots(e_state_mon, s_state_mon, i_spike_mon, e_spike_mon, ind_input, init_shifts, final_shifts):

    # plot the evolution of shifts
    plt.figure()
    NS = min(len(ind_input), 4) 
    for n in range(nn):
        for s in range(NS):
            sp_cell = NS * 100 + nn * 10 + s * nn + n + 1 # subplot coordinate
            if (n==0 and s==0): sp0 = plt.subplot(sp_cell) # make equal scales
            else: plt.subplot(sp_cell, sharex=sp0, sharey=sp0)
            plt.plot(s_state_mon.t/b2.ms, (s_state_mon.shift[ind_input[s] * nn + n] + s_state_mon.shift_correction[ind_input[s] * nn + n]) / b2.ms , label=f'delay of n={n+1} s={ind_input[s]+1}')
            for t in i_spike_mon.spike_trains()[ind_input[s]]:
                plt.axvline(t/b2.ms, ls='--', c='C1', lw=1) # orange == input
            for t in e_spike_mon.spike_trains()[n]:
                plt.axvline(t/b2.ms, ls='--', c='C2', lw=1) # green == main
            plt.legend()
            plt.xlabel("$t$, ms")
            plt.ylabel("$delay$, ms")

    # plot the evolution of membrane potential 
    plt.figure()
    for n in range(nn):
        if (n==0): sp1 = plt.subplot(nn*100 + 10*1 + n + 1) # make equal scales
        else: plt.subplot(nn*100 + 10*1 + n + 1, sharex=sp1, sharey=sp1)
        plt.plot(e_state_mon.t/b2.ms, e_state_mon.v[n]/b2.mV, label='v')
        plt.plot(e_state_mon.t/b2.ms, e_state_mon.vt_n[n]/b2.mV , label='v_T')
        #plt.plot(e_state_mon.t/b2.ms, e_state_mon.g_n[n] , label='g_n')
        for t in i_spike_mon.t:
            plt.axvline(t/b2.ms, ls='--', c='C1', lw=1) # orange == input
        for t in e_spike_mon.spike_trains()[n]:
            plt.axvline(t/b2.ms, ls='--', c='C2', lw=1) # green == main
        plt.legend()
        plt.xlabel("t, ms")
        plt.ylabel(f"v, mV   (n={n+1})")

    # plot initial and final shifts
    plt.figure()
    plt.subplot(2,1,1)
    # for i in range(nn):
    #     plt.plot(np.arange(1,ni + 1,1), init_shifts[i], marker = '*', label = f'{i}_neuron')
    plt.plot(np.arange(1,ni + 1,1), init_shifts[0], marker = '.', label = f'{1}st_neuron', linestyle = '-')
    plt.plot(np.arange(1,ni + 1,1), init_shifts[1], marker = '*', label = f'{2}nd_neuron', linestyle = '-')
    plt.plot(np.arange(1,ni + 1,1), init_shifts[2], marker = 'p', label = f'{3}rd_neuron', linestyle = '-')
    plt.legend()
    plt.xlabel("$synapse\ index$")
    plt.ylabel("$delay$, ms")
    #plt.title("Initial shifts values")
    plt.grid()
    plt.subplot(2,1,2)
    # for i in range(nn):
    #     plt.plot(np.arange(1,ni + 1,1), final_shifts[i], marker = '*', label = f'{i}_neuron')
    plt.plot(np.arange(1,ni + 1,1), final_shifts[0], marker = '.', label = f'{1}st_neuron', linestyle = '-')
    plt.plot(np.arange(1,ni + 1,1), final_shifts[1], marker = '*', label = f'{2}nd_neuron', linestyle = '-')
    plt.plot(np.arange(1,ni + 1,1), final_shifts[2], marker = 'p', label = f'{3}rd_neuron', linestyle = '-')
    plt.legend()
    plt.xlabel("$synapse\ index$")
    plt.ylabel("$delay$, ms")
    #plt.title("Final shifts values")
    plt.grid()
    plt.show()



(train_left_spike_time, train_right_spike_time, train_central_spike_time) = DataImport(True)

if os.path.exists(file_path_initial_shifts):
    with open(file_path_initial_shifts, 'rb') as f:
        shifts = np.load(f) * b2.second
else:
    shifts = b2.rand(ni*nn) * max_spike_time * b2.ms
    np.save(file_path_initial_shifts, shifts)

snn = SNN(nn, ni, 1, shifts)

e_state_mon = b2.StateMonitor(snn.group_n, ['v', 'vt_n', 'g_n'], record=True)
s_state_mon = b2.StateMonitor(snn.synapses_e, ['shift', 'shift_correction'], record=True)
i_spike_mon = b2.SpikeMonitor(snn.input_group)
e_spike_mon = b2.SpikeMonitor(snn.group_n)
snn.net.add(e_state_mon)
snn.net.add(s_state_mon)
snn.net.add(i_spike_mon)
snn.net.add(e_spike_mon)

num = 1
for i in range(num):
    snn.input_group.set_spikes(list(range(ni)), train_left_spike_time[i] * b2.ms, train_period)
    snn.net.run(time_run * b2.ms)
    snn.input_group.set_spikes(list(range(ni)), [30000]*ni * b2.ms, 0 * b2.ms)
    snn.net.run(time_relax * b2.ms)

    snn.input_group.set_spikes(list(range(ni)), train_right_spike_time[i] * b2.ms, train_period)
    snn.net.run(time_run * b2.ms)
    snn.input_group.set_spikes(list(range(ni)), [30000]*ni * b2.ms, 0 * b2.ms)
    snn.net.run(time_relax * b2.ms)

    snn.input_group.set_spikes(list(range(ni)), train_central_spike_time[i] * b2.ms, train_period)
    snn.net.run(time_run * b2.ms)
    snn.input_group.set_spikes(list(range(ni)), [30000]*ni * b2.ms, 0 * b2.ms)
    snn.net.run(time_relax * b2.ms)


np.save(file_path_shifts, snn.synapses_e.shift)

init_shifts = (shifts/b2.ms*10//1).reshape(ni, nn).transpose()
print("Init shifts:")
print(init_shifts)
final_shifts = (snn.synapses_e.shift/b2.ms*10//1).reshape(ni, nn).transpose()
print("Final shifts:")
print(final_shifts)

ind_input_to_show = [0,9]
show_plots(e_state_mon, s_state_mon, i_spike_mon, e_spike_mon, ind_input_to_show, init_shifts, final_shifts)

snn.net.remove(e_state_mon)
snn.net.remove(s_state_mon)
snn.net.remove(i_spike_mon)
snn.net.remove(e_spike_mon)


