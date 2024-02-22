import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4

import sys

sys.path.insert(1, "../helperScripts")

import numpy as np
import matplotlib.pyplot as plt
import features as fts
import MOOSEModel as mm
import expcells
import brute_curvefit as bcf
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from pprint import pprint
from goMultiprocessing import Multiprocessthis_appendsave
import pickle
import json
from scipy import signal
import warnings

### get exp cell features ###
LJP = 15e-3
samplingrate = 20000
stim_start_exp = 0.3469
stim_end_exp = stim_start_exp +0.5
validcells = [
        "2023_01_04_cell_1",
        "2023_01_04_cell_2",
        "2023_01_04_cell_3",
        "2023_01_04_cell_4",
        "2023_01_04_cell_5",
        "2023_01_04_cell_6",
        # "2023_01_20_cell_1", #Invalid exp
        "2023_01_20_cell_2",
        "2023_01_20_cell_3",
        "2023_01_20_cell_4",
        "2023_02_13_cell_1",
        "2023_02_13_cell_2",
        "2023_02_13_cell_3",
        "2023_02_13_cell_4",
    ]

df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")
df_exppasF = pd.read_pickle("../helperScripts/exppasF.pkl")

Features_list = ["ISIavg_1.5e-10","freq_1.5e-10"]
######################################################
stimamp = 30e-12
stim_start_chirp = 0.3
stim_end_chirp = 13.3
stim_start = 0.5
stim_end = 1
tstop = 14.5
stimlist_chirp = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start_chirp} & t<{stim_end_chirp}) * sin(2*3.14159265359*(t-{stim_start_chirp})^3) * {stimamp}",
]
stimlist_chirp2 = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start_chirp} & t<{stim_start_chirp+501}) * sin(2*3.14159265359*(t-{stim_start_chirp})^2) * {stimamp}",
]
stimlist_CC = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.5}) * {-25e-12}",
]
stimlist_epsp = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.5}) * {0e-12}",
]
stimlist_150pA = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.5}) * {150e-12}",
]
stimlist_bis = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.2}) * {150e-12} + (t>{stim_start+0.5} & t<{stim_start+0.7}) * {-50e-12}",
]

baseModel = {
    "Parameters": {
        "notes2": "",
        "Morphology": {
            "sm_len": 15e-6,
            "sm_diam": 15e-6,
            "dend_len": 500e-6,
            "dend_diam_start": 4e-6,
            "dend_diam_end": 4e-6,
            "num_dend_segments": 0,
        },
        "Passive": {
            "Em": -82e-3,
            "sm_RM": 0.1,
            "sm_CM": 0.17,
            "sm_RA": 1.59,
            "dend_RM": 1.54,
            "dend_CM": 0.021,
            "dend_RA": 0.73,
        },
        "Channels": {
            "Na_T_Chan": {
                "Gbar": 1e-6,
                "Erev": 0.06,
                "Kinetics": "../Kinetics/Na_T_Chan_Royeck_wslow",
            },
            "K_DR_Chan": {
                "Gbar": 1e-5,
                "Erev": -0.100,
                "Kinetics": "../Kinetics/K_DR_Chan_Custom3",
            },
        },
    }
}
####################################################

def ourfunc(i):
    model = deepcopy(baseModel)
    model["Parameters"]["Passive"]["Em"] = np.random.uniform(-100e-3, -70e-3)
    model["Parameters"]["Passive"]["sm_RM"] = np.random.uniform(0.09,0.11) #np.random.uniform(0.01,0.5)
    model["Parameters"]["Passive"]["sm_CM"] = np.random.uniform(0.085,0.34) #np.random.uniform(0.05,0.3)

    model["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"] = 10 ** np.random.uniform(
        -6, -5
        # -8, -4
    )
    model["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"] = 10 ** np.random.uniform(
        # -7, -2
        -5, -3
    )
    model["Parameters"]["Channels"]["K_DR_Chan"]["Erev"] = np.random.uniform(-100e-3, -50e-3)

    t0, I, Vtrace0, Ca = mm.runModel(model, 0e-12)
    if any(Vtrace0[t0>stim_start]>-70e-3):
        return [{}]

    tm25, I, Vtracem25, Ca = mm.runModel(model, -25e-12)
    pasfeatures = fts.calcRinCin(tm25, Vtracem25, stim_start, stim_end)
    pasfeatures_exp = df_exppasF[["E_rest", "Rin", "Cin", "tau"]]
    if any(pasfeatures[:-1]<np.array(pasfeatures_exp.min())) or any(pasfeatures[:-1]>np.array(pasfeatures_exp.max())):
        return [{}]

    features = fts.modelfeatures(model, stim_start, stim_end, refreshKin=True)
    model["Features"] = features

    for f in Features_list:
        if model["Features"][f] == None:
            return [{}]
        if (model["Features"][f]< df_expsummaryactiveF.loc[f, "10th quantile"]/2) or (model["Features"][f]> df_expsummaryactiveF.loc[f, "90th quantile"]*2):
            return [{}]

    print('yoohoooo', model["Parameters"]["Channels"]["K_DR_Chan"]["Erev"], model["Features"]["DBLO_1.5e-10"])
    return [model]


Multiprocessthis_appendsave(
   ourfunc, range(5000), [], ["tempactivemodels.pkl"], seed=np.random.randint(0, 2**32 - 1), npool=90
)

##############################################################################
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open("tempactivemodels.pkl", "rb") as f, open(
    'activemodels.json', "a"
) as file:
    while True:
        model = pickle.load(f)
        # pprint(model)
        if len(model) > 0:
            json.dump(model, file, cls=NpEncoder)
            file.write("\n")