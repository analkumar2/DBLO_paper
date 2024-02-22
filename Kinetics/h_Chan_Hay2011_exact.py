# exec(open('../../Compilations/Kinetics/h_Chan_Hay2011.py').read())
# Base is Nandi 2022. They took it from Hay 2011. Not parameterized. 

import numpy as np
import pickle
import pandas as pd
import moose
import matplotlib.pyplot as plt

SOMA_A = 3.14e-8
F = 96485.3329
R = 8.314
celsius = 34
dt = 0.05e-3
ENa = 53e-3 #0.092 #from Deepanjali data
EK = -107e-3 #-0.099 #from Deepanjali data
Eh = -0.045
ECa = 0.140 #from Deepanjali data
Em = -0.065


Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
# dV = (Vmax-Vmin)/Vdivs
# v = np.arange(Vmin,Vmax, dV)
v = np.linspace(Vmin,Vmax, Vdivs)*1e3 #1e3 because this is converted from NEURON which uses mV and ms
Camin = 1e-12
Camax = 3
Cadivs = 4000
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

mshift = 0
def h_Chan(name):
    h = moose.HHChannel( '/library/' + name )
    h.Ek = Eh
    h.Gbar = 300.0*SOMA_A
    h.Gk = 0.0
    h.Xpower = 1.0
    h.Ypower = 0
    h.Zpower = 0

    def vtrap(x,y):
        a = x / (np.exp(x / y) - 1)
        a[np.abs(x/y)<1e-6] = y * (1 - x[np.abs(x/y)<1e-6] / y / 2)
        return a

    # mAlpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9)
    mAlpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9)
    mBeta  =  0.001*193*np.exp(v/33.1)
    mInf = mAlpha/(mAlpha + mBeta) + mshift
    mTau = 1/(mAlpha + mBeta)

    xgate = moose.element( h.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = mInf/mTau*1e3
    xgate.tableB = 1.0/mTau*1e3

    return h


if __name__ == "__main__":
    moose.Neutral('library')
    h_Chan('h_Chan')
    plt.figure()
    plt.plot(v*1e-3, moose.element('library/h_Chan/gateX').tableA/moose.element('library/h_Chan/gateX').tableB, label='nInf')
    plt.ylabel('Inf')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(v*1e-3, 1/moose.element('library/h_Chan/gateX').tableB, label='nTau')
    plt.ylabel('Tau')
    plt.legend()
    plt.grid()
    plt.show()