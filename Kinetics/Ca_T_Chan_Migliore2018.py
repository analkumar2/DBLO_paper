# T type calcium channel cat.mod Migliore2018
# exec(open('Ca_T_Chan_(Migliore2018).py').read())

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

#################################
#################################

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
# dV = (Vmax-Vmin)/Vdivs
# v = np.arange(Vmin,Vmax, dV)
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 1e-12
Camax = 1
Cadivs = 400 # Enough for channels with ca dependence only because of ghk
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def Ca_T_Chan(name):
    Ca_T = moose.HHChannel2D( '/library/' + name )
    Ca_T.Ek = ECa
    Ca_T.Gbar = 300.0*SOMA_A
    Ca_T.Gk = 0.0
    Ca_T.Xpower = 2.0
    Ca_T.Ypower = 1.0
    Ca_T.Zpower = 1.0
    Ca_T.Xindex = 'VOLT_INDEX'
    Ca_T.Yindex = 'VOLT_INDEX'
    Ca_T.Zindex = 'VOLT_C1_INDEX'
    Ca_T.instant = 4

    xgate = moose.element( Ca_T.path + '/gateX' )
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

    ygate = moose.element( Ca_T.path + '/gateY' )
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

    zgate = moose.element( Ca_T.path + '/gateZ' )
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
    q10 = 5
    mmin=0.2
    hmin=10
    a0h =0.015
    zetah = 3.5
    vhalfh = -75
    gmh=0.6
    a0m =0.04
    zetam = 2
    vhalfm = -28
    gmm=0.1
    z = 2
    T = celsius+273.15
    gcatbar=.003e4

    qt=q10**((celsius-25)/10)
    a = 0.2*(-1.0*v*1e3+19.26)/(np.exp((-1.0*v*1e3+19.26)/10.0)-1.0)
    b = 0.009*np.exp(-v*1e3/22.03)
    minf = a/(a+b)
    alpmt = np.exp(0.0378*zetam*(v*1e3-vhalfm))
    betmt = np.exp(0.0378*zetam*gmm*(v*1e3-vhalfm))
    mtau = betmt/(qt*a0m*(1+alpmt))
    mtau[mtau<mmin]=mmin
    tblA = np.zeros([Vdivs,1])
    tblB = np.zeros([Vdivs,1])
    for i in np.arange(1):
        tblA[:,i] = minf/mtau
        tblB[:,i] = 1/mtau
        print(i, end='\r')
    xgate.tableA = tblA*1e3
    xgate.tableB = tblB*1e3

    qt=q10**((celsius-25)/10)
    a = 1.e-6*np.exp(-v*1e3/16.26)
    b = 1/(np.exp((-v*1e3+29.79)/10.)+1.)
    hinf = a/(a+b)
    alph = np.exp(0.0378*zetah*(v*1e3-vhalfh))
    beth = np.exp(0.0378*zetah*gmh*(v*1e3-vhalfh))
    htau = beth/(a0h*(1+alph))
    htau[htau<hmin]=hmin
    tblA = np.zeros([Vdivs,1])
    tblB = np.zeros([Vdivs,1])
    for i in np.arange(1):
        tblA[:,i] = hinf/htau
        tblB[:,i] = 1/htau
        print(i, end='\r')
    ygate.tableA = tblA*1e3
    ygate.tableB = tblB*1e3

    ezfrt  = np.exp(z*v*F/R/T)
    tblA = np.zeros([Vdivs,Cadivs])
    tblB = np.zeros([Vdivs,Cadivs])
    for i in np.arange(Cadivs):
        tblA[:,i] = v*(ca[i]/cao*ezfrt-1)/(ezfrt-1)/(v-ECa)
        print(i, end='\r')
    tblB = tblA*0 +1
    zgate.tableA = tblA
    zgate.tableB = tblB

    addmsg4 = moose.Mstring( Ca_T.path + '/addmsg4' )
    addmsg4.value = '../Ca_conc concOut . concen'

    addmsg2 = moose.Mstring( Ca_T.path + '/addmsg2' )
    addmsg2.value = '. IkOut ../Ca_conc current'
    return Ca_T
