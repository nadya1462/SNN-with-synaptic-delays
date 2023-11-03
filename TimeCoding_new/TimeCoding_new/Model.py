from asyncio import events
import brian2 as b2
from definitions import *

class SNN:
    def __init__(self, nn, ni, train_mode, initial_shifts): 

        # Input_group neurons
        times = [0]*ni * b2.ms
        self.input_group = b2.SpikeGeneratorGroup(ni, list(range(ni)), times, name = "input_group")

        # Main neurons
        model_n = '''
          dv/dt = (v_rest_n - v + g_n * (E_exc_n - v)) / tau_n: volt
          tau_n : second
          vt_n : volt
          v_rest_n : volt
          v_reset_n : volt
          E_exc_n : volt
          g_n : 1
          train_mode_n : 1
          tpre_first : second
          '''
        self.group_n = b2.NeuronGroup(nn, model=model_n, threshold='v>vt_n', reset='v=v_reset_n', 
          refractory=max_spike_time*b2.ms, method='euler', events = {"suppress_event" : "v < v_reset_n"})
        self.group_n.train_mode_n = train_mode
        ## RULE 1!!! -------- 0.6 = max_spike_time*(1-k)^4 < tau_n < max_spike_time/ni = 1  (4 -- number of trains to syncronize 4 input spikes)
        self.group_n.tau_n = 0.7 * b2.ms     
        self.group_n.vt_n = -52 * b2.mV
        self.group_n.v_rest_n = -65 * b2.mV
        self.group_n.v_reset_n = self.group_n.v_rest_n
        self.group_n.v = self.group_n.v_rest_n
        self.group_n.g_n = 0
        self.group_n.E_exc_n = 0 * b2.mV

        # EXCISITORY synapses 
        model_e = '''
          g = g_amp * exp((tpre + shift + shift_correction - t) / tau_s) * int((t > tpre + shift + shift_correction) and (tpre >= 0 * ms)) : 1
          g_n_post = g : 1 (summed)
          shift : second
          shift_correction : second
          tau_s : second
          g_amp : 1
          tpre : second
          train_mode_e : 1
          k : 1
          ''' 
        on_pre_e = '''
          tpre = t
          tpre_first = int((tpre_first < 0 * ms) or (t - tpre_first > max_spike_time * ms)) * t + int((tpre_first >= 0 * ms) and (t - tpre_first <= max_spike_time * ms)) * tpre_first
          shift_correction += train_mode_e * k * ((max_spike_time * ms + tpre_first) - (t + shift + shift_correction))
          '''
        on_post_e = { 
          "post" : '''
            shift += shift_correction
            shift_correction = 0 * ms
            k = k * 0.8
            ''',
          "suppress" : 
            "shift_correction = 0 * ms" }
        
        self.synapses_e = b2.Synapses(self.input_group, self.group_n, model=model_e, on_pre=on_pre_e, 
          on_post=on_post_e, method='euler', on_event={"suppress" : "suppress_event"})
        self.synapses_e.connect()
        self.synapses_e.train_mode_e = train_mode
        self.synapses_e.k = 0.4     # learning rate 0..1
        self.synapses_e.tpre = -1 * b2.ms 
        self.synapses_e.tpre_first = -1 * b2.ms 
        self.synapses_e.tau_s = 0.3 * b2.ms
        self.synapses_e.g_amp = 0.2
        self.synapses_e.shift = initial_shifts

        # INHIBITORY synapses
        model_i = '''
          delta_v_post_i : volt
          ''' 
        on_pre_i = '''
          v_post -= delta_v_post_i
          '''
        self.synapses_i = b2.Synapses(self.group_n, self.group_n, model=model_i, on_pre=on_pre_i, method='euler')
        self.synapses_i.connect(condition='i != j')
        self.synapses_i.delta_v_post_i = 20 * b2.mV

        self.net = b2.Network(self.input_group, self.group_n, self.synapses_e, self.synapses_i)
