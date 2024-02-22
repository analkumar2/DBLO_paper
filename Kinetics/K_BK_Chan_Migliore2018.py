# K_BK channel taken from mod files of Migliore2018: cagk.mod
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
Camax = 400e-3
Cadivs = 8000 #Enough for K_BK
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def K_BK_Chan(name):
    K_BK = moose.HHChannel2D( '/library/' + name )
    K_BK.Ek = EK
    K_BK.Gbar = 300.0*SOMA_A
    K_BK.Gk = 0.0
    K_BK.Xpower = 1.0
    K_BK.Ypower = 0.0
    K_BK.Zpower = 0.0
    K_BK.Xindex = 'VOLT_C1_INDEX'

    xgate = moose.element( K_BK.path + '/gateX' )
    xgate.xminA = Vmin
    xgate.xmaxA = Vmax
    xgate.xdivsA = Vdivs
    xgate.yminA = Camin
    xgate.ymaxA = Camax
    xgate.ydivsA = Cadivs
    xgate.xminB = Vmin
    xgate.xmaxB = Vmax
    xgate.xdivsB = Vdivs
    xgate.yminB = Camin
    xgate.ymaxB = Camax
    xgate.ydivsB = Cadivs

    d1 = .84
    d2 = 1.
    k1 = .48e-3
    k2 = .13e-6
    abar = .28
    bbar = .48
    st=1
    F_KC = F/1000
    gbar=.01e4

    exp1k1d1 = k1*np.exp(-2*d1*F_KC*v*1e3/R/(273.15 + celsius))
    exp1k2d2 = k2*np.exp(-2*d2*F_KC*v*1e3/R/(273.15 + celsius))

    tblA = np.zeros([Vdivs,Cadivs])
    tblB = np.zeros([Vdivs,Cadivs])
    for i in np.arange(Cadivs):
        tblA[:,i] = ca[i]*abar/(ca[i] + exp1k1d1)
        tblB[:,i] = tblA[:,i] + bbar/(1 + ca[i]/exp1k2d2)
        print(i, end='\r')

    xgate.tableA = tblA*1e3
    xgate.tableB = tblB*1e3
    # print(np.array(xgate.tableA)/np.array(xgate.tableB))

    addmsg4 = moose.Mstring( K_BK.path + '/addmsg4' )
    addmsg4.value = '../Ca_conc concOut . concen'
    return K_BK
