# As seen in Migliore2018
# Just to setup. The actual values are set up after model building.
# Calcium pool decay and conversion from calcium current to concentration.

import numpy as np
import pickle
import pandas as pd
import moose

def Ca_Conc ( name ):
    Ca = moose.CaConc( '/library/' + name )
    depth    = 0.1e-6
    taur    = 200e-3
    cainf    = 50e-6
    # B = 28789637.7

    Ca.tau = taur/7
    Ca.Ca_base = cainf
    Ca.diameter = 500e-6
    Ca.length = 1000e-6
    Ca.thick = 177.9e-6
    return Ca
