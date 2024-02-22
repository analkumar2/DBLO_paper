from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from neuron.units import s, V, m
h.load_file("stdrun.hoc")
h.load_file('import3d.hoc')
h.celsius = 34

cell = h.Import3d_SWC_read()
cell.input('reconstruction.swc')
i3d = h.Import3d_GUI(cell, 0)
i3d.instantiate(None)

for sec in h.allsec():
	if sec.name()[:4] == "axon":
		h.delete_section(sec=sec)

axon = [h.Section(name="axon[0]"), h.Section(name="axon[1]"), h.Section(name="axon[2]")]

for sec in axon:
    sec.L = 30
    sec.diam = 1
    sec.nseg = 1 + 2 * int(sec.L / 40.0)
axon[0].connect(h.soma[0], 0.5, 0.0)
axon[1].connect(axon[0], 1.0, 0.0)

soma = h.soma[0]
for sec in h.allsec():
	if 'soma' in sec.name():
		sec.cm = 1.0
		sec.Ra = 118.68070828445255
		sec.insert("pas")
		sec(0.5).pas.g = 0.00085515989842588011
		sec(0.5).pas.e = -82.34514617919922e-3*V
		sec.insert('NaTs')
		sec.insert('K_P')
		sec.insert('Kv3_1')
		sec.insert('Im')
		sec.insert('Ih')
		sec.insert('Nap')
		sec.insert('K_T')
		sec.insert('SK')
		sec.insert('Ca_HVA')
		sec.insert('Ca_LVA')
		sec.insert('CaDynamics')
		sec.ena = 53.0
		sec.ek = -107
		sec(0.5).Im.gbar = 3.5831046068113176e-07
		sec(0.5).Ih.gbar = 0.00082455944527867655
		sec(0.5).NaTs.gbar = 1.7057586476320825
		sec(0.5).Nap.gbar = 0.001237343601431547
		sec(0.5).K_P.gbar = 0.044066924433216317
		sec(0.5).K_T.gbar = 0.0036854358538674914
		sec(0.5).SK.gbar = 0.15513355160133413
		sec(0.5).Kv3_1.gbar = 0.23628414673253201
		sec(0.5).Ca_HVA.gbar = 0.00055676424421458449
		sec(0.5).Ca_LVA.gbar = 0.0085727072024145701
		sec(0.5).CaDynamics.gamma = 3.5960605745938294e-05
		sec(0.5).CaDynamics.decay = 446.26912463264068
	elif 'axon' in sec.name():
		sec.cm = 1.0
		sec.Ra = 118.68070828445255
		sec.insert("pas")
		sec(0.5).pas.g = 0.00095157282638217886
		sec(0.5).pas.e = -82.34514617919922e-3*V
	elif 'dend' in sec.name():
		sec.cm = 3.314938540399873
		sec.Ra = 118.68070828445255
		sec.insert("pas")
		sec(0.5).pas.g = 6.8117062501849471e-06
		sec(0.5).pas.e = -82.34514617919922e-3*V
	elif 'apic' in sec.name():
		sec.cm = 3.314938540399873
		sec.Ra = 118.68070828445255
		sec.insert("pas")
		sec(0.5).pas.g = 7.2396479517848947e-05
		sec(0.5).pas.e = -82.34514617919922e-3*V

for sec in h.allsec():
    sec.nseg = 1 + 2 * int(sec.L / 40.0)


iclamp = h.IClamp(soma(0.5))
iclamp.delay = 0.200 *s
iclamp.dur = 0.500*s
iclamp.amp = 0.175

v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
t = h.Vector().record(h._ref_t)  # Time stamp vector

# print(soma.psection())

h.dt = 5e-5*s
h.v_init = -82.34514617919922e-3*V
h.finitialize(-82.34514617919922e-3*V)
h.tstop = 1*s


h.run()

plt.plot(t,v)
plt.show()
# np.savez('output.npz', delay=iclamp.delay*1e-3, amp=iclamp.amp*1e-9, t_vec = np.array(t)*1e-3, v_vec = np.array(v)*1e-3)
