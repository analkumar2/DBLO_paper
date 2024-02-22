# KD channel taken from mod files of Migliore2018: kdb.mod
# Problems:

import numpy as np
import pickle
import pandas as pd
import moose

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
Camin = 1e-12
Camax = 3
Cadivs = 4000
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def K_D_Chan(name):
    K_D = moose.HHChannel( '/library/' + name )
    K_D.Ek = EK
    K_D.Gbar = 300.0*SOMA_A
    K_D.Gk = 0.0
    K_D.Xpower = 1.0
    K_D.Ypower = 0.0
    K_D.Zpower = 0

    vhalfn=-33
    a0n=0.005
    zetan=3
    gmn=0.7
    nmax=2
    q10=1
    sh = 0
    gkdbar=.01e4

    qt=q10**((celsius-24)/10)
    a = np.exp(1.e-3*zetan*(v*1e3-vhalfn-sh)*9.648e4/(8.315*(273.16+celsius)))
    ninf = 1/(1+a)
    taun = np.exp(1.e-3*zetan*gmn*(v*1e3-vhalfn-sh)*9.648e4/(8.315*(273.16+celsius)))/(qt*a0n*(1+a))
    taun[taun<nmax] = nmax/qt

    xgate = moose.element( K_D.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = ninf/taun*1e3
    xgate.tableB = 1.0/taun*1e3

    return K_D
