# KM channel taken from mod files of Migliore2018: kmb.mod
# Problems:

import numpy as np
import pickle
import pandas as pd
import moose
import matplotlib.pyplot as plt

SOMA_A = 3.14e-8
F = 96485.3329
R = 8.314
celsius = 32
dt = 0.05e-3
ENa = 0.092
EK = -0.099
Eh = -0.030
ECa = 0.140
Em = -0.065

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
# dV = (Vmax-Vmin)/Vdivs
# v = np.arange(Vmin,Vmax, dV)
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 0.04e-3
Camax = 500e-3
Cadivs = 8000
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def K_M_Chan(name):
    K_M = moose.HHChannel( '/library/' + name )
    K_M.Ek = EK
    K_M.Gbar = 300.0*SOMA_A
    K_M.Gk = 0.0
    K_M.Xpower = 1.0
    K_M.Ypower = 0.0
    K_M.Zpower = 0

    vhalfl=-40
    kl=-10
    vhalft=-42
    a0t=0.003
    zetat=7
    gmt=.4
    q10=5
    b0=60
    st=1
    sh =0
    gbar=.0001e4

    qt=q10**((celsius-35)/10)
    inf = (1/(1 + np.exp((v*1e3-vhalfl-sh)/kl)))
    a = np.exp(0.0378*zetat*(v*1e3-vhalft-sh))
    tau = b0 + np.exp(0.0378*zetat*gmt*(v*1e3-vhalft-sh))/(a0t*(1+a))

    xgate = moose.element( K_M.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = inf/tau*1e3
    xgate.tableB = 1.0/tau*1e3

    return K_M

if __name__ == "__main__":
    moose.Neutral('library')
    K_M_Chan('K_M_Chan')
    plt.figure()
    plt.plot(v*1e-3, moose.element('library/K_M_Chan/gateX').tableA/moose.element('library/K_M_Chan/gateX').tableB, label='nInf')
    plt.ylabel('Inf')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(v*1e-3, 1/moose.element('library/K_M_Chan/gateX').tableB, label='nTau')
    plt.ylabel('Tau')
    plt.legend()
    plt.grid()
    plt.show()