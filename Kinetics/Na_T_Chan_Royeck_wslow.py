# exec(open('../../Compilations/Kinetics/Na_T_Chan_Hay2011.py').read())
# Base inf is exp Royeck et al., 2008. Base tau is Hay 2011

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
Eh = -0.030
ECa = 0.140 #from Deepanjali data
Em = -0.065

###################################
m_vhalf_inf, m_slope_inf = -0.05132511, 0.00747505
h_vhalf_inf, h_slope_inf = -6.33730481e-02, -9.05936563e-03
s_vhalf_inf, s_slope_inf, s_A, s_B, s_C, s_D, s_E, s_F = 1,-0.01, 100,1e-50,0,0,-1e-50, 1e-3 #The F term determines the Tau
###################################

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

def ChanGate(v,vhalf_inf, slope_inf, A, B, C, D, E, F):
    # alge model
    Inf = 1/(1+np.exp((v-vhalf_inf)/-slope_inf))
    yl = (v-A)/-B
    yr = (v-A)/E
    Tau = (C + (1 + yl/(np.sqrt(1+yl**2)))/2) * (D + (1 + yr/(np.sqrt(1+yr**2)))/2) * F
    Tau[Tau<0.00002] = 0.00002
    return [Inf,Tau]

mvhalf, hvhalf = -38, -66
def Na_T_Chan(name):
    Na_T = moose.HHChannel( '/library/' + name )
    Na_T.Ek = ENa
    Na_T.Gbar = 300.0*SOMA_A
    Na_T.Gk = 0.0
    Na_T.Xpower = 3.0
    Na_T.Ypower = 1.0
    Na_T.Zpower = 1.0
    Na_T.useConcentration = False

    malphaF = 0.182
    mbetaF = 0.124
    # mvhalf = mvhalf
    mk = 6

    halphaF = 0.015
    hbetaF = 0.015
    # hvhalf = hvhalf
    hk = 6

    qt = 2.3**((celsius-21)/10)


    def vtrap(x,y):
        if abs(x/y)<1e-6:
            return y * (1 - x / y / 2)
        else:
            return x / (np.exp(x / y) - 1)

    mAlpha = malphaF * np.array([vtrap(-(vv - mvhalf), mk) for vv in v])
    mBeta = mbetaF * np.array([vtrap(vv - mvhalf, mk) for vv in v])
    mTau = (1/(mAlpha + mBeta))/qt

    hAlpha = halphaF * np.array([vtrap(vv - hvhalf, hk) for vv in v])
    hBeta = hbetaF * np.array([vtrap(-(vv - hvhalf), hk) for vv in v])
    hTau = (1/(hAlpha + hBeta))/qt

    mInf = ChanGate(v*1e-3,m_vhalf_inf, m_slope_inf, 1, 1, 1, 1, 1, 1)[0]
    hInf = ChanGate(v*1e-3,h_vhalf_inf, h_slope_inf, 1, 1, 1, 1, 1, 1)[0]
    sInf, sTau = ChanGate(v*1e-3,s_vhalf_inf, s_slope_inf, s_A, s_B, s_C, s_D, s_E, s_F)

    xgate = moose.element( Na_T.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = mInf/mTau*1e3
    xgate.tableB = 1.0/mTau*1e3

    ygate = moose.element( Na_T.path + '/gateY' )
    ygate.min = Vmin
    ygate.max = Vmax
    ygate.divs = Vdivs
    ygate.tableA = hInf/hTau*1e3
    ygate.tableB = 1.0/hTau*1e3

    zgate = moose.element( Na_T.path + '/gateZ' )
    zgate.min = Vmin
    zgate.max = Vmax
    zgate.divs = Vdivs
    zgate.tableA = sInf/sTau
    zgate.tableB = 1.0/sTau

    return Na_T


if __name__ == "__main__":
    moose.Neutral('library')
    Na_T_Chan('Na_T_Chan')
    plt.figure()
    plt.plot(v*1e-3, (moose.element('library/Na_T_Chan/gateX').tableA/moose.element('library/Na_T_Chan/gateX').tableB)**3, label='nInf')
    plt.plot(v*1e-3, moose.element('library/Na_T_Chan/gateY').tableA/moose.element('library/Na_T_Chan/gateY').tableB, label='hInf')
    plt.plot(v*1e-3, moose.element('library/Na_T_Chan/gateZ').tableA/moose.element('library/Na_T_Chan/gateZ').tableB, label='sInf')
    plt.ylabel('Inf')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(v*1e-3, 1/moose.element('library/Na_T_Chan/gateX').tableB, label='nTau')
    plt.plot(v*1e-3, 1/moose.element('library/Na_T_Chan/gateY').tableB, label='hTau')
    plt.plot(v*1e-3, 1/moose.element('library/Na_T_Chan/gateZ').tableB, label='sTau')
    plt.ylabel('Tau')
    plt.legend()
    plt.grid()
    plt.show()