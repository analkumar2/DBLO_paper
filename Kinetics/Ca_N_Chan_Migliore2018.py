# N type calcium channel can2.mod Migliore2018
# exec(open('Ca_N_Chan_(Migliore2018).py').read())

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
Cadivs = 10000 #Enough for Ca_N
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def Ca_N_Chan(name):
    Ca_N = moose.HHChannel2D( '/library/' + name )
    Ca_N.Ek = ECa
    Ca_N.Gbar = 300.0*SOMA_A
    Ca_N.Gk = 0.0
    Ca_N.Xpower = 2.0
    Ca_N.Ypower = 1.0
    Ca_N.Zpower = 1.0
    Ca_N.Xindex = 'VOLT_INDEX'
    Ca_N.Yindex = 'VOLT_INDEX'
    Ca_N.Zindex = 'VOLT_C1_INDEX'
    Ca_N.instant = 4

    xgate = moose.element( Ca_N.path + '/gateX' )
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

    ygate = moose.element( Ca_N.path + '/gateY' )
    ygate.xminA = Vmin
    ygate.xmaxA = Vmax
    ygate.xdivsA = Vdivs
    # ygate.yminA = Camin
    # ygate.ymaxA = Camax
    # ygate.ydivsA = Cadivs
    ygate.xminB = Vmin
    ygate.xmaxB = Vmax
    ygate.xdivsB = Vdivs
    # ygate.yminB = Camin
    # ygate.ymaxB = Camax
    # ygate.ydivsB = Cadivs

    zgate = moose.element( Ca_N.path + '/gateZ' )
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
    q10=5
    mmin = 0.2
    hmin = 3
    a0m =0.03
    zetam = 2
    vhalfm = -14
    gmm=0.1
    gcanbar=.0003e4

    alph = 1.6e-4*np.exp(-v*1e3/48.4)
    beth = 1/(np.exp((-v*1e3+39.0)/10.)+1.)
    alpm = 0.1967*(-1.0*v*1e3+19.88)/(np.exp((-1.0*v*1e3+19.88)/10.0)-1.0)
    betm = 0.046*np.exp(-v*1e3/20.73)
    alpmt = np.exp(0.0378*zetam*(v*1e3-vhalfm))
    betmt = np.exp(0.0378*zetam*gmm*(v*1e3-vhalfm))

    qt=q10**((celsius-25)/10)
    a = alpm
    b = 1/(a + betm)
    minf = a/(a+b)
    minf = a*b
    taum = betmt/(qt*a0m*(1+alpmt))
    taum[taum<mmin/qt] = mmin/qt
    tblA = np.zeros([Vdivs,1])
    tblB = np.zeros([Vdivs,1])
    for i in np.arange(1):
        tblA[:,i] = minf/taum
        tblB[:,i] = 1/taum
        # print(i, end='\r')
    xgate.tableA = tblA*1e3
    xgate.tableB = tblB*1e3

    qt=q10**((celsius-25)/10)
    a = alph
    b = 1/(a + beth)
    hinf = a*b
    tauh = 80*np.ones(len(hinf))
    tauh[tauh<hmin] = hmin
    tblA = np.zeros([Vdivs,1])
    tblB = np.zeros([Vdivs,1])
    for i in np.arange(1):
        tblA[:,i] = hinf/tauh
        tblB[:,i] = 1/tauh
        # print(i, end='\r')
    ygate.tableA = tblA*1e3
    ygate.tableB = tblB*1e3

    ezfrt  = np.exp(z*v*F/R/T)
    tblA = np.zeros([Vdivs,Cadivs])
    tblB = np.zeros([Vdivs,Cadivs])
    for i in np.arange(Cadivs):
        tblA[:,i] = ki/(ki+ca[i])*v*(ca[i]/cao*ezfrt-1)/(ezfrt-1)/(v-ECa)
        # print(i, end='\r')
    tblB = tblA*0 +1
    zgate.tableA = tblA
    zgate.tableB = tblB

    addmsg4 = moose.Mstring( Ca_N.path + '/addmsg4' )
    addmsg4.value = '../Ca_conc concOut . concen'

    addmsg2 = moose.Mstring( Ca_N.path + '/addmsg2' )
    addmsg2.value = '. IkOut ../Ca_conc current'
    return Ca_N
