import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4

import sys

sys.path.insert(1, "../helperScripts")

import numpy as np
import matplotlib.pyplot as plt
import features_ as fts
import MOOSEModel_ as mm
import expcells
import brute_curvefit as bcf
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from pprint import pprint
from goMultiprocessing import Multiprocessthis_appendsave
import pickle
import json
import scipy.signal as scs
import warnings


Featurelist = [
    "CellID",
    # "E_rest_0",
    # "Input resistance",
    # "Cell capacitance",
    # "Time constant",
    # "sagV_m50",
    # "sagrat_m50",
    "freq_0",
    # "AP1_amp_1.5e-10",
    # "APp_amp_1.5e-10",
    # "AP1_time_1.5e-10",
    # "APp_time_1.5e-10",
    # "APavgpratio_amp_1.5e-10",
    # "AP1_width_1.5e-10",
    # "APp_width_1.5e-10",
    # "AP1_thresh_1.5e-10",
    # "APp_thresh_1.5e-10",
    # "AP1_lat_1.5e-10",
    # "ISI1_1.5e-10",
    # "ISIl_1.5e-10",
    "ISIavg_1.5e-10",
    # "ISImedian_1.5e-10",
    "freq_1.5e-10",
    # "Adptn_id_1.5e-10",
    # "fAHP_AP1_amp_1.5e-10",
    # "DBLO_1.5e-10",
    # "AbsDBL_1.5e-10",
    # "AP1_amp_3e-10",
    # "APp_amp_3e-10",
    # "AP1_time_3e-10",
    # "APp_time_3e-10",
    # "APavgpratio_amp_3e-10",
    # "AP1_width_3e-10",
    # "APp_width_3e-10",
    # "AP1_thresh_3e-10",
    # "APp_thresh_3e-10",
    # "AP1_lat_3e-10",
    # "ISI1_3e-10",
    # "ISIl_3e-10",
    # "ISIavg_3e-10",
    # "ISImedian_3e-10",
    # "freq_3e-10",
    # "Adptn_id_3e-10",
    # "fAHP_AP1_amp_3e-10",
    # "DBLO_3e-10",
    # "AbsDBL_3e-10",
    # "freq300to150ratio",
]
# pasmodelFlist = ['E_rest_0', 'Input resistance', 'Cell capacitance', 'Time constant', 'sagV_m50', 'sagrat_m50', "ISImedian_1.5e-10"]

### get exp cell features ###
LJP = 15e-3
samplingrate = 20000

validcells = [
    "2023_01_04_cell_1",
    "2023_01_04_cell_2",
    "2023_01_04_cell_3",
    "2023_01_04_cell_4",
    "2023_01_04_cell_5", #At 300pA, firing is not proper
    "2023_01_04_cell_6",
    # "2023_01_20_cell_1", #invalid exp
    "2023_01_20_cell_2",
    "2023_01_20_cell_3",
    "2023_01_20_cell_4",
    "2023_02_13_cell_1",
    "2023_02_13_cell_2",
    "2023_02_13_cell_3",
    "2023_02_13_cell_4",
]
df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")
print(df_expsummaryactiveF)

df_exppasF = pd.read_pickle("../helperScripts/exppasF.pkl")
print(df_exppasF)
####################################
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
        "notes": "",
        "Morphology": {
            "sm_len": 15e-6,
            "sm_diam": 15e-6,
            "dend_len": 30e-6,
            "dend_diam_start": 2e-6,
            "dend_diam_end": 2e-6,
            "num_dend_segments": 1,
        },
        "Passive": {
            "Em": -80e-3,
            "sm_RM": 0.19,
            "sm_CM": 0.09,
            "sm_RA": 50,
            "dend_RM": 0.19,
            "dend_CM": 0.09,
            "dend_RA": 50,
        },
        "Channels": {
            "Na_T_Chan": {
                "Gbar": 1e-4,
                "Erev": 0.06,
                "Kinetics": "../Kinetics/Na_T_Chan_Royeck_wslow",
            },
            "K_DR_Chan": {
                "Gbar": 1e-3,
                "Erev": -0.100,
                "Kinetics": "../Kinetics/K_DR_Chan_Custom3",
            },
        },
    }
}


def ourfunc(i):
    model = deepcopy(baseModel)
    model["Parameters"]["Passive"]["sm_RM"] = np.random.uniform(0.01, 0.2)
    model["Parameters"]["Passive"]["dend_RM"] = model["Parameters"]["Passive"]["sm_RM"]
    model["Parameters"]["Passive"]["sm_RA"] = 10**np.random.uniform(-5, 2)
    model["Parameters"]["Passive"]["dend_RA"] = model["Parameters"]["Passive"]["sm_RA"]
    model["Parameters"]["Passive"]["sm_CM"] = np.random.uniform(0.086, 0.124)
    model["Parameters"]["Passive"]["dend_CM"] = model["Parameters"]["Passive"]["sm_CM"]
    model["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"] = 10 ** np.random.uniform(-7, -5)
    model["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"] = 10 ** np.random.uniform(-6, -3)

    tm25, Itracem25, Vtracem25, Ca = mm.runModel(model, -25e-12, refreshKin=False)
    E_rest, Rin, Cin, tau, sag_rat = fts.calcRinCin(tm25,Vtracem25,stim_start,stim_end)

    if E_rest<df_exppasF["E_rest"].min() or E_rest>df_exppasF["E_rest"].max():
        return [{}]
    if Rin<df_exppasF["Rin"].min() or Rin>df_exppasF["Rin"].max():
        return [{}]
    if Cin<df_exppasF["Cin"].min() or Cin>df_exppasF["Cin"].max():
        return [{}]
    if tau<df_exppasF["tau"].min() or tau>df_exppasF["tau"].max():
        return [{}]

    tvec, Ivec, Vmvec, Cavec = mm.runModel(
        model,
        CurrInjection=150e-12,
        vClamp=None,
        refreshKin=True,
        Truntime=None,
        syn=False,
        synwg=0,
        synfq=5,
    )

    peaksidx, _ = scs.find_peaks(Vmvec[(tvec>=stim_start) & (tvec<stim_end)])
    peakstimes = tvec[(tvec>=stim_start) & (tvec<stim_end)][peaksidx]
    peaksVm = Vmvec[(tvec>=stim_start) & (tvec<stim_end)][peaksidx]
    if len(peakstimes)<2:
        return [{}]

    ISDBL = [np.min(Vmvec[(tvec>=stim_start) & (tvec<stim_end)][peaksidx[i]: peaksidx[i+1]]) for i in range(len(peaksidx)-1)]

    t0, Itrace0, Vtrace0, Ca = mm.runModel(model, 0e-12, refreshKin=False)
    peaksidx0, _ = scs.find_peaks(Vtrace0[(t0>=stim_start) & (t0<stim_end)])

    Featurelist_ = Featurelist[1:]  ## To remove CellID

    modelF = fts.modelfeatures(
        model, stim_start=0.5, stim_end=1, refreshKin=False
    )

    model["Features"] = modelF
    (
        model["Features"]["freq_0"],
        model["Features"]["freq_1.5e-10"],
        model["Features"]["ISIavg_1.5e-10"],
        model["Features"]["AP1_amp_1.5e-10"],
        model["Features"]["DBLO_1.5e-10"],
    ) = (
        len(peaksidx0)*2,
        len(peakstimes)*2,
        np.nanmean(np.diff(peakstimes)),
        peaksVm[0] - model["Features"]["E_rest_0"] if len(peaksVm)>0 else None,
        np.nanmean(ISDBL) - model["Features"]["E_rest_0"] if len(ISDBL)>0 else None,
    )
    if model["Features"]["freq_1.5e-10"]*model["Features"]["ISIavg_1.5e-10"]<0.950:
        return [{}]

    for f in Featurelist_:
        if model["Features"][f] is None:
            return [{}]

    features = df_expsummaryactiveF.assign(Model=pd.Series(model["Features"]))
    features["is_between"] = ""
    for rrow in Featurelist_:
        features.loc[rrow, "is_between"] = (
            features.loc[rrow, "Model"] >= features.loc[rrow, "10th quantile"]
        ).all() & (
            features.loc[rrow, "Model"] <= features.loc[rrow, "90th quantile"]
        ).all()
        if not features.loc[rrow, "is_between"]:
            return [{}]

    print('yooohoooo', model["Features"]["DBLO_1.5e-10"],model["Features"]["AP1_amp_1.5e-10"],)
    return [model]

# Multiprocessthis_appendsave(ourfunc, range(5000), [], ['temp_activemodels.pkl'], seed=np.random.randint(1,100000), npool=100)


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

#######################################################################

with open('temp_activemodels.pkl', "rb") as f, open('activemodels.json', "a") as file:
    while True:
        try:
            model = pickle.load(f)
            if len(model) > 0:
                json.dump(model, file, cls=NpEncoder)
                file.write("\n")
        except Exception:
            break
