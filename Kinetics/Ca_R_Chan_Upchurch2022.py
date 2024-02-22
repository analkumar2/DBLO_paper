# R type calcium channel car.mod Upchurch et al '22
# exec(open('Ca_R_Chan_Upchurch2022.py').read())

import numpy as np
import moose
import matplotlib.pyplot as plt

SOMA_A = 3.14e-8
F = 96485.3329
R = 8.314
celsius = 32
dt = 0.05e-3
ENa = 0.050
EK = -0.085
Eh = -0.045
ECa = 0.128
Em = -0.090

#################################
n_vhalf_inf, n_slope_inf, n_A, n_B, n_C, n_D, n_E, n_F = -0.0485,0.003, 9.99999185e-02,5.29018640e-05,2.70605390e-34,5.00000000e-01,3.63578193e-05,1.00000121e-02
h_vhalf_inf, h_slope_inf, h_A, h_B, h_C, h_D, h_E, h_F = -0.053,-0.001, 9.99999185e-02,5.29018093e-05,4.87443733e-35,5.00000000e-01,3.63577885e-05,1.00000121e-01
#################################

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 0.01e-3
Camax = 1e-3
Cadivs = 4000
ca = np.linspace(Camin,Camax, Cadivs)

def ChanGate(v,vhalf_inf, slope_inf, A, B, C, D, E, F):
    # alge model
    Inf = 1/(1+np.exp((v-vhalf_inf)/-slope_inf))
    yl = (v-A)/-B
    yr = (v-A)/E
    Tau = (C + (1 + yl/(np.sqrt(1+yl**2)))/2) * (D + (1 + yr/(np.sqrt(1+yr**2)))/2) * F
    Tau[Tau<0.00002] = 0.00002
    return [Inf,Tau]

def Ca_R_Chan(name):
    Ca_R = moose.HHChannel( '/library/' + name )
    Ca_R.Ek = ECa
    Ca_R.Gbar = 300.0*SOMA_A
    Ca_R.Gk = 0.0
    Ca_R.Xpower = 3.0
    Ca_R.Ypower = 1.0
    Ca_R.Zpower = 0.0

    [nInf,nTau] = ChanGate(v,*[n_vhalf_inf, n_slope_inf, n_A, n_B, n_C, n_D, n_E, n_F])
    [hInf,hTau] = ChanGate(v,*[h_vhalf_inf, h_slope_inf, h_A, h_B, h_C, h_D, h_E, h_F])

    xgate = moose.element( Ca_R.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = nInf/nTau
    xgate.tableB = 1.0/nTau

    ygate = moose.element( Ca_R.path + '/gateY' )
    ygate.min = Vmin
    ygate.max = Vmax
    ygate.divs = Vdivs
    ygate.tableA = hInf/hTau
    ygate.tableB = 1.0/hTau

    addmsg2 = moose.Mstring( Ca_R.path + '/addmsg2' )
    addmsg2.value = '. IkOut ../Ca_conc current'
    return Ca_R


if __name__ == "__main__":
    moose.Neutral('library')
    Ca_R_Chan('Ca_R')
    plt.figure()
    plt.plot(v, (moose.element('library/Ca_R/gateX').tableA/moose.element('library/Ca_R/gateX').tableB)**3, label='nInf')
    plt.plot(v, moose.element('library/Ca_R/gateY').tableA/moose.element('library/Ca_R/gateY').tableB, label='lInf')
    plt.ylabel('Inf')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(v, 1/moose.element('library/Ca_R/gateX').tableB, label='nTau')
    plt.plot(v, 1/moose.element('library/Ca_R/gateY').tableB, label='lTau')
    plt.ylabel('Tau')
    plt.legend()
    plt.grid()
    plt.show()