# exec(open('../../Compilations/Kinetics/K_SK_Chan_Custom4.py').read())
# Hirschber 1998 exp kinetics. Parameterized but since this uses Ca, not by  alge

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
cac, hillcoeff, m, c = 0.00056, 4.6, 47000, 26
#################################

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
# dV = (Vmax-Vmin)/Vdivs
# v = np.arange(Vmin,Vmax, dV)
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 50e-6 #If changing this, be careful that the dCa is at most 0.01e-3
Camax = 3000e-6 #3e-3 works because the tables becomes steady after 3e-3 for K_SK
Cadivs = 4000
# dCa = (Camax-Camin)/Cadivs
# ca = np.arange(Camin,Camax, dCa)
ca = np.linspace(Camin,Camax, Cadivs)

def ChanGate(v,vhalf_inf, slope_inf, A, B, C, D, E, F):
    # alge model
    Inf = 1/(1+np.exp((v-vhalf_inf)/-slope_inf))
    yl = (v-A)/-B
    yr = (v-A)/E
    Tau = (C + (1 + yl/(np.sqrt(1+yl**2)))/2) * (D + (1 + yr/(np.sqrt(1+yr**2)))/2) * F
    Tau[Tau<0.00002] = 0.00002
    return [Inf,Tau]

def SKChanGate(ca, cac, hillcoeff, m, c):
    car = (ca/cac)**hillcoeff
    Inf = car / ( 1 + car )
    # Tau =  1 / beta / (1 + car)
    Tau = 1/(m*ca + c)
    Tau[Tau<0.0002] = 0.0002
    return [Inf, Tau]

def K_SK_Chan(name):
    K_SK = moose.HHChannel( '/library/' + name )
    K_SK.Ek = EK
    K_SK.Gbar = 300.0*SOMA_A
    K_SK.Gk = 0.0
    K_SK.Xpower = 0
    K_SK.Ypower = 0
    K_SK.Zpower = 3
    K_SK.useConcentration = 1

    [mInf, mTau] = SKChanGate(ca, cac, hillcoeff, m, c)

    zgate = moose.element( K_SK.path + '/gateZ' )
    zgate.min = Camin
    zgate.max = Camax
    zgate.divs = Cadivs
    zgate.tableA = mInf/mTau
    zgate.tableB = 1.0/mTau

    addmsg3 = moose.Mstring( K_SK.path + '/addmsg3' )
    addmsg3.value = '../Ca_conc concOut . concen'
    return K_SK
