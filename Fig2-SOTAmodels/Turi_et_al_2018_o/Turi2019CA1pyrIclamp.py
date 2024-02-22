## exec(open('Turi2019CA1pyrIclamp.py').read())

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sbn
# sbn.set()

h('load_file("cells/pyramidal_cell.hoc")')
h('load_file("cells/pyramidal_cell_LJP.hoc")')

pytestcell5 = h.PyramidalCell()
v_vec = h.Vector()             # Membrane potential vector
t_vec = h.Vector()             # Time stamp vector
v_vec.record(pytestcell5.soma(0.5)._ref_v)
t_vec.record(h._ref_t)

stim = h.IClamp(pytestcell5.soma(0.5))
stim.delay = 200
stim.amp = 0.165 #0.132
stim.dur = 500
h.celsius = 34
h.finitialize()
h.tstop = 800
h.run()

np.savez('output.npz', delay=stim.delay*1e-3, amp=stim.amp*1e-9, t_vec = np.array(t_vec)*1e-3, v_vec = np.array(v_vec)*1e-3)

############################################

pytestcell5 = h.PyramidalCell_LJP()
v_vec = h.Vector()             # Membrane potential vector
t_vec = h.Vector()             # Time stamp vector
v_vec.record(pytestcell5.soma(0.5)._ref_v)
t_vec.record(h._ref_t)

stim = h.IClamp(pytestcell5.soma(0.5))
stim.delay = 200
stim.amp = 0.295 #0.258
stim.dur = 500
h.celsius = 34
h.finitialize()
h.tstop = 800
h.run()

np.savez('output_LJP.npz', delay=stim.delay*1e-3, amp=stim.amp*1e-9, t_vec = np.array(t_vec)*1e-3, v_vec = np.array(v_vec)*1e-3)
