"""produces a simulation of the backpropagation of the action potential into the apical trunk"""

# _title_   : bAP.py
# _author_  : Matus Tomko
# _mail_    : matus.tomko __at__ fmph.uniba.sk

import os
from collections import OrderedDict
from quantities import mV

import efel
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui


h.nrn_load_dll('./Mods/nrnmech.dll')
h.xopen('pyramidal_cell_weak_bAP_updated.hoc')
cell = h.CA1_PC_Tomko()

stim = h.IClamp(cell.soma[0](0.5))
stim.delay = 200
stim.amp = 0.44
stim.dur = 500

v_vec_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)
t_vec = h.Vector()
t_vec.record(h._ref_t)

h.dt = 0.025
h.tstop = 900
h.v_init = -65
h.celsius = 34
h.init()
h.finitialize(-65)
h.cvode_active(1)
h.run()

np.savez('output.npz', delay=stim.delay*1e-3, amp=stim.amp*1e-9, t_vec = np.array(t_vec)*1e-3, v_vec = np.array(v_vec_soma)*1e-3)



