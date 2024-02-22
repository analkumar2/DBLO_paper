### Here, we make active models with imp morphology, -90mV Erev, LJP correction, fast K_DR

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

# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=RuntimeWarning)
# warnings.simplefilter(action="ignore", category=RuntimeError)

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
    # "DBL_1.5e-10",
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
    # "DBL_3e-10",
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
            "sm_RM": 1.48,
            "sm_CM": 0.015,
            "sm_RA": 1.59,
            "dend_RM": 1.54,
            "dend_CM": 0.021,
            "dend_RA": 0.73,
        },
        "Channels": {
            "Na_T_Chan": {
                "Gbar": 2e-5,
                "Erev": 0.06,
                "Kinetics": "../Kinetics/Na_T_Chan_Royeck_wslow",
            },
            "K_DR_Chan": {
                "Gbar": 1e-6,
                "Erev": -0.100,
                "Kinetics": "../Kinetics/K_DR_Chan_Custom3",
            },
            "h_Chan": {
                "Gbar": 1e-8,
                "Erev": -0.040,
                "Kinetics": "../Kinetics/h_Chan_Hay2011_exact",
            },
        },
    }
}
# mm.plotModel(baseModel)

# Load models from the JSON file
basemodels_list = []
file_path = "../helperScripts/1compt.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        basemodels_list.append(basemodel)

######################################################################################
def ourfunc(i):
    Na_T_Chan_Gbar = 10 ** np.random.uniform(
            -6, -4
            # -4.5, -3.7
        )
    K_DR_Chan_Gbar = 10 ** np.random.uniform(
            -7, -4
            # -5.5, -4.3
        )

    # print('#################################################')
    basemodel["Parameters"]["Channels"]["Na_T_Chan"] = baseModel["Parameters"]["Channels"]["Na_T_Chan"]
    basemodel["Parameters"]["Channels"]["K_DR_Chan"] = baseModel["Parameters"]["Channels"]["K_DR_Chan"]

    basemodel["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"] = Na_T_Chan_Gbar
    basemodel["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"] = K_DR_Chan_Gbar

    Featurelist_ = Featurelist[1:]  ## To remove CellID

    modelF = fts.modelfeatures(
        basemodel, stim_start=0.5, stim_end=1, refreshKin=False
    )

    basemodel["Features"] = modelF

    for f in Featurelist_:
        if modelF[f] is None:
            return [{}]

    features = df_expsummaryactiveF.assign(Model=pd.Series(basemodel["Features"]))
    features["is_between"] = ""
    for rrow in Featurelist_:
        features.loc[rrow, "is_between"] = (
            features.loc[rrow, "Model"] >= features.loc[rrow, "10th quantile"]
        ).all() & (
            features.loc[rrow, "Model"] <= features.loc[rrow, "90th quantile"]
        ).all()
        if not features.loc[rrow, "is_between"]:
            return [{}]

    if basemodel["Features"]["freq_1.5e-10"]*basemodel["Features"]["ISIavg_1.5e-10"] < 0.9: ## So that Depolarization blocks are taken care of
        return [{}]
        
    print('yoooohooooo', basemodel["Features"]["DBLO_1.5e-10"], basemodel["Features"]["AP1_amp_1.5e-10"])
    # plt.show()
    return [basemodel]


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
seeed = np.random.randint(0, 2**32 - 1)
# seeed = 1964186147
print(seeed)

for cells in validcells:
    print(cells)
    for basemodel in basemodels_list:
        if basemodel["Parameters"]["notes"] == cells:
            break

    Multiprocessthis_appendsave(
       ourfunc, range(500), [], ["tempactivemodels_imp.pkl"], seed=seeed, npool=100
    )

    with open("tempactivemodels_imp.pkl", "rb") as f, open('activemodels_NaTRoyeck.json', "a") as file:
        while True:
            try:
                model = pickle.load(f)
                if len(model) > 0:
                    json.dump(model, file, cls=NpEncoder)
                    file.write("\n")
            except Exception:
                break
