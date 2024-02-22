# exec(open('../../Compilations/Kinetics/K_DR_Chan_Custom3.py').read())
# K_DR channel parameterized kinetics. Base is experimental kinetics by Migliore2018
# n_E is divided by 2. To compensate, F is multiplied by 1.1 and A shifted right by 0.006

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

#################################
n_vhalf_inf, n_slope_inf, n_A, n_B, n_C, n_D, n_E, n_F = 0.013,0.0087666, 1.26E-02-0.006,1.73E-02,0.00E+00,0.00E+00,3.43E-02/2,1.02E-01*1.1
#################################

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
# dV = (Vmax-Vmin)/Vdivs
# v = np.arange(Vmin,Vmax, dV)
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 0.04e-3
Camax = 1
Cadivs = 8000
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

def K_DR_Chan(name):
    K_DR = moose.HHChannel( '/library/' + name )
    K_DR.Ek = EK
    K_DR.Gbar = 300.0*SOMA_A
    K_DR.Gk = 0.0
    K_DR.Xpower = 1.0
    K_DR.Ypower = 0.0
    K_DR.Zpower = 0

    [nInf,nTau] = ChanGate(v,*[n_vhalf_inf, n_slope_inf, n_A, n_B, n_C, n_D, n_E, n_F])

    xgate = moose.element( K_DR.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = nInf/nTau
    xgate.tableB = 1.0/nTau

    return K_DR


if __name__ == "__main__":
    [nInf,nTau] = ChanGate(v,*[n_vhalf_inf, n_slope_inf, n_A, n_B, n_C, n_D, n_E, n_F])
    [nInf2,nTau2] = ChanGate(v,*[n_vhalf_inf, n_slope_inf, n_A-0.006, n_B, n_C, n_D, n_E/2, n_F*1.1])
    # hhh = np.arange(-55e-3 - 0.016, 36e-3 - 0.016, 10e-3)
    # parameterfitted_list = np.array([np.array([2.56015621e-09, 1.12713826e-02, 3.82738005e-18, 5.00000000e-01,
    #    2.43722849e-09, 2.81531338e-01]), np.array([5.01958073e-09, 1.30315203e-02, 8.51866701e-04, 1.14235667e-01,
    #    4.33268931e-09, 1.14243302e-01]), np.array([7.37120030e-09, 1.25073616e-02, 1.36142786e-06, 9.74136065e-02,
    #    5.34766250e-09, 9.74163312e-02]), np.array([8.99721426e-09, 9.24950966e-03, 5.35874431e-03, 1.49499585e-01,
    #    6.20488857e-09, 1.49505473e-01]), np.array([1.16121395e-08, 7.32873273e-03, 7.26493010e-11, 3.46096004e-01,
    #    8.67817670e-09, 5.00000000e-01]), np.array([1.54864626e-08, 7.78551206e-03, 7.15086937e-10, 8.17741596e-02,
    #    1.29630899e-08, 7.50310292e-02]), np.array([1.24100284e-08, 1.57381849e-01, 6.50940375e-02, 5.36024556e-02,
    #    1.76802248e-08, 6.50738526e-03]), np.array([1.53142103e-08, 5.27067710e-03, 9.55547838e-01, 4.99999411e-01,
    #    7.90179057e-09, 1.17767958e-02]), np.array([2.54855042e-08, 6.10498775e-03, 9.12926651e-01, 5.00000000e-01,
    #    2.17731422e-09, 6.85336947e-02]), np.array([2.92396179e-08, 6.27161627e-03, 9.02253739e-01, 5.00000000e-01,
    #    1.56575057e-09, 1.07168695e-01])])
    # nInf_fitted = parameterfitted_list[:,4]/1.8e-8
    # nTau_fitted = parameterfitted_list[:,5]
    plt.figure()
    plt.plot(v, nInf, label='nInf original')
    plt.plot(v, nInf2, label='nInf fast')
    # plt.plot(hhh, nInf_fitted, label='nInf_fitted')
    plt.ylabel('Inf')
    plt.legend()
    plt.figure()
    plt.plot(v, nTau, label='nTau original')
    plt.plot(v, nTau2, label='nTau fast')
    # plt.plot(hhh, nTau_fitted, label='nTau_fitted')
    plt.ylabel('Tau')
    plt.legend()
    plt.xlabel("Membrane Potential (V)")
    plt.title('Potassium channel activation time constant')
    plt.show()