# Na_P channel Nap_Et2.mod Hay2011
# exec(open('Na_P_Chan_(Hay2011).py').read())

import numpy as np
import pickle
import pandas as pd
import moose

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

Vmin = -0.100
Vmax = 0.100
Vdivs = 3000
v = np.linspace(Vmin,Vmax, Vdivs)
Camin = 0.01e-3
Camax = 1e-3
Cadivs = 4000
ca = np.linspace(Camin,Camax, Cadivs)

Xpower = 3.0
Ypower = 1.0
mvhalf = -52.6
mvslope = 4.6
def Na_P_Chan(name):
    Na_P = moose.HHChannel( '/library/' + name )
    Na_P.Ek = ENa
    Na_P.Gbar = 300.0*SOMA_A
    Na_P.Gk = 0.0
    Na_P.Xpower = Xpower
    Na_P.Ypower = Ypower
    Na_P.Zpower = 0.0

    qt = 2.3**((34-21)/10)
    mInf = 1.0/(1+np.exp((v*1e3- mvhalf)/-mvslope))
    mAlpha = (0.182 * (v*1e3- -38))/(1-(np.exp(-(v*1e3- -38)/6)))
    mBeta  = (0.124 * (-v*1e3 -38))/(1-(np.exp(-(-v*1e3 -38)/6)))
    mTau = 6*(1/(mAlpha + mBeta))/qt
    hInf = 1.0/(1+np.exp((v*1e3- -48.8)/10))
    hAlpha = -2.88e-6 * (v*1e3 + 17) / (1 - np.exp((v*1e3 + 17)/4.63))
    hBeta = 6.94e-6 * (v*1e3 + 64.4) / (1 - np.exp(-(v*1e3 + 64.4)/2.63))
    hTau = (1/(hAlpha + hBeta))/qt

    xgate = moose.element( Na_P.path + '/gateX' )
    xgate.min = Vmin
    xgate.max = Vmax
    xgate.divs = Vdivs
    xgate.tableA = mInf/mTau*1e3
    xgate.tableB = 1/mTau*1e3

    ygate = moose.element( Na_P.path + '/gateY' )
    ygate.min = Vmin
    ygate.max = Vmax
    ygate.divs = Vdivs
    ygate.tableA = hInf/hTau*1e3
    ygate.tableB = 1/hTau*1e3

    return Na_P
