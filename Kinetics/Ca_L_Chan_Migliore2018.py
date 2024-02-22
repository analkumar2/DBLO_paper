# L type calcium channel cal2.mod Migliore2018
# exec(open('Ca_L_Chan_(Migliore2018).py').read())

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
Camax = 1
Cadivs = 10000 # Is enough for Ca_L
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def Ca_L_Chan(name):
    Ca_L = moose.HHChannel2D( '/library/' + name )
    Ca_L.Ek = ECa
    Ca_L.Gbar = 300.0*SOMA_A
    Ca_L.Gk = 0.0
    Ca_L.Xpower = 2.0
    Ca_L.Ypower = 0.0
    Ca_L.Zpower = 1.0
    Ca_L.Xindex = 'VOLT_INDEX'
    # Ca_L.Yindex = 'C1_INDEX'
    Ca_L.Zindex = 'VOLT_C1_INDEX'
    Ca_L.instant = 4

    xgate = moose.element( Ca_L.path + '/gateX' )
    xgate.xminA = Vmin
    xgate.xmaxA = Vmax
    xgate.xdivsA = Vdivs
    # xgate.yminA = Camin
    # xgate.ymaxA = Camax
    # xgate.ydivsA = Cadivs
    xgate.xminB = Vmin
    xgate.xmaxB = Vmax
    xgate.xdivsB = Vdivs
    # xgate.yminB = Camin
    # xgate.ymaxB = Camax
    # xgate.ydivsB = Cadivs

    zgate = moose.element( Ca_L.path + '/gateZ' )
    zgate.xminA = Vmin
    zgate.xmaxA = Vmax
    zgate.xdivsA = Vdivs
    zgate.yminA = Camin
    zgate.ymaxA = Camax
    zgate.ydivsA = Cadivs
    zgate.xminB = Vmin
    zgate.xmaxB = Vmax
    zgate.xdivsB = Vdivs
    zgate.yminB = Camin
    zgate.ymaxB = Camax
    zgate.ydivsB = Cadivs

    cao = 2
    z = 2
    T = celsius+273.15
    ki=.001
    cai = 50.e-6
    cao = 2
    q10 = 5
    mmin=0.2
    tfa = 1
    a0m =0.1
    zetam = 2
    vhalfm = 4
    gmm=0.1
    gcalbar=.003e4

    qt=q10**((celsius-25)/10)
    a = 15.69*(-1.0*v*1e3+81.5)/(np.exp((-1.0*v*1e3+81.5)/10.0)-1.0)
    b = 1/(a + 0.29*np.exp(-v*1e3/10.86))
    minf = a*b
    alpmt = np.exp(0.0378*zetam*(v*1e3-vhalfm))
    betmt = np.exp(0.0378*zetam*gmm*(v*1e3-vhalfm))
    tau = betmt/(qt*a0m*(1+alpmt))
    tau[tau<mmin/qt]=mmin/qt
    tblA = np.zeros([Vdivs,1])
    tblB = np.zeros([Vdivs,1])
    for i in np.arange(1):
        tblA[:,i] = minf/tau
        tblB[:,i] = 1/tau
        print(i, end='\r')
    xgate.tableA = tblA*1e3
    xgate.tableB = tblB*1e3

    ezfrt  = np.exp(z*v*F/R/T)
    tblA = np.zeros([Vdivs,Cadivs])
    tblB = np.zeros([Vdivs,Cadivs])
    for i in np.arange(Cadivs):
        tblA[:,i] = ki/(ki+ca[i])*v*(ca[i]/cao*ezfrt-1)/(ezfrt-1)/(v-ECa)
        # print(i, end='\r')
    tblB = tblA*0 +1
    zgate.tableA = tblA
    zgate.tableB = tblB

    addmsg4 = moose.Mstring( Ca_L.path + '/addmsg4' )
    addmsg4.value = '../Ca_conc concOut . concen'

    addmsg2 = moose.Mstring( Ca_L.path + '/addmsg2' )
    addmsg2.value = '. IkOut ../Ca_conc current'
    return Ca_L
