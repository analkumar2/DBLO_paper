# KA channel taken from mod files of Migliore2018: kaprox.mod
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
ENa = 0.092 #from Deepanjali data
EK = -0.099 #from Deepanjali data
Eh = -0.030
ECa = 0.140 #from Deepanjali data
Em = -0.065

#################################
#################################

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

def K_A_Chan(name):
    K_A = moose.HHChannel( '/library/' + name )
    K_A.Ek = EK
    K_A.Gbar = 300.0*SOMA_A
    K_A.Gk = 0.0
    K_A.Xpower = 1.0
    K_A.Ypower = 1.0
    K_A.Zpower = 0

    vhalfn=11
    vhalfl=-56
    a0l=0.05
    a0n=0.05
    zetan=-1.5
    zetal=3
    gmn=0.55
    gml=1
    lmin=2
    nmin=0.1
    pw=-1
    tq=-40
    qq=5
    q10=5
    qtl=1
    gkabar=.008e4

    qt=q10**((celsius-24)/10)
    zeta=zetan+pw/(1+np.exp((v*1e3-tq)/qq))
    alpn = np.exp(1.e-3*zeta*(v*1e3-vhalfn)*9.648e4/(8.315*(273.16+celsius)))
    a = alpn
    ninf = 1/(1+a)
    zeta=zetan+pw/(1+np.exp((v*1e3-tq)/qq))
    betn = np.exp(1.e-3*zeta*gmn*(v*1e3-vhalfn)*9.648e4/(8.315*(273.16+celsius)))
    taun = betn/(qt*a0n*(1+a))
    taun[taun<nmin] = nmin

    alpl = np.exp(1.e-3*zetal*(v*1e3-vhalfl)*9.648e4/(8.315*(273.16+celsius)))
    betl = np.exp(1.e-3*zetal*gml*(v*1e3-vhalfl)*9.648e4/(8.315*(273.16+celsius)))
    a = alpl
    linf = 1/(1+ a)
    taul = 0.26*(v*1e3+50)/qtl
    taul[taul<lmin/qtl] = lmin/qtl


    xgate = moose.element( K_A.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = ninf/taun*1e3
    xgate.tableB = 1.0/taun*1e3

    ygate = moose.element( K_A.path + '/gateY' )
    ygate.min = Vmin
    ygate.max = Vmax
    ygate.divs = Vdivs
    ygate.tableA = linf/taul*1e3
    ygate.tableB = 1.0/taul*1e3

    return K_A

if __name__ == "__main__":
    moose.Neutral('library')
    K_A = K_A_Chan("K_A_Chan")
    nInf = moose.element('library/K_A_Chan/gateX').tableA/moose.element('library/K_A_Chan/gateX').tableB
    nTau = 1/moose.element('library/K_A_Chan/gateX').tableB
    lInf = moose.element('library/K_A_Chan/gateY').tableA/moose.element('library/K_A_Chan/gateY').tableB
    lTau = 1/moose.element('library/K_A_Chan/gateY').tableB

    plt.figure()
    plt.plot(v, nInf, label='nInf')
    plt.plot(v, lInf, label='lInf')
    # plt.plot(hhh, nInf_fitted, label='nInf_fitted')
    # plt.plot(hhh, lInf_fitted, label='lInf_fitted')
    plt.ylabel('Inf')
    plt.legend()
    plt.figure()
    plt.plot(v, nTau, label='nTau')
    plt.plot(v, lTau, label='lTau')
    # plt.plot(hhh, nTau_fitted, label='nTau_fitted')
    # plt.plot(hhh, lTau_fitted, label='lTau_fitted')
    plt.ylabel('Tau')
    plt.legend()
    plt.show()