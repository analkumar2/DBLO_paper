### Here, we make active models with all the channels

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
    # "DBL_1.5e-10",
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
    # "DBL_3e-10",
    # "freq300to150ratio",
]
# pasmodelFlist = ['E_rest_0', 'Input resistance', 'Cell capacitance', 'Time constant', 'sagV_m50', 'sagrat_m50', "ISImedian_1.5e-10"]

### get exp cell features ###
LJP = 15e-3
samplingrate = 20000
df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")

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
                "Gbar": 1e-4,
                "Erev": 0.06,
                "Kinetics": "../Kinetics/Na_T_Chan_Royeck_wslow",
            },
            "K_DR_Chan": {
                "Gbar": 1e-3,
                "Erev": -0.100,
                "Kinetics": "../Kinetics/K_DR_Chan_Custom3",
                "KineticVars": {
                    "n_vhalf_inf": 0.013,
                    "n_slope_inf": 0.0087666,
                    "n_A": 0.0126,
                    "n_B": 0.0173,
                    "n_C": 0.0,
                    "n_D": 0.0,
                    "n_E": 0.0343,
                    "n_F": 0.102,
                },
            },
            "h_Chan": {
                "Gbar": 1e-8,
                "Erev": -0.040,
                "Kinetics": "../Kinetics/h_Chan_Hay2011_exact",
            },
        },
    }
}

# Load models from the JSON file
basemodels_list = []
file_path = "../helperScripts/1compt.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        basemodels_list.append(basemodel)

######################################################################################
def ourfunc(i):
    model = deepcopy(baseModel)
    pasmodel = basemodels_list[np.random.randint(0, len(basemodels_list))]
    model["Parameters"]["notes"] = pasmodel["Parameters"]["notes"]
    model["Parameters"]["Morphology"] = pasmodel["Parameters"]["Morphology"]
    model["Parameters"]["Passive"] = pasmodel["Parameters"]["Passive"]
    if "Gbar" in pasmodel["Parameters"]["Channels"]["h_Chan"].keys():
        model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = pasmodel["Parameters"][
            "Channels"
        ]["h_Chan"]["Gbar"]
        model["Parameters"]["Channels"]["h_Chan"].pop("gbar", None)
    else:
        model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = pasmodel["Parameters"][
            "Channels"
        ]["h_Chan"]["gbar"]*np.pi*model["Parameters"]["Morphology"]["sm_len"]*model["Parameters"]["Morphology"]["sm_diam"]
        model["Parameters"]["Channels"]["h_Chan"].pop("gbar", None)


    model["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"] = 10 ** np.random.uniform(
        -7, -5
    )
    model["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"] = 10 ** np.random.uniform(
        -6, -3
    )

    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_A"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_A"] = np.random.uniform(_-10e-3, _+10e-3)
    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_B"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_B"] = np.random.uniform(_/5, _*5)
    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_C"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_C"] = np.random.uniform(0, _+10e-3)
    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_D"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_D"] = np.random.uniform(0, _+10e-3)
    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_E"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_E"] = np.random.uniform(_/5, _*2)
    _ = model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_F"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_F"] = np.random.uniform(_/2, _*5)

    Featurelist_ = Featurelist[1:]  ## To remove CellID

    modelF = fts.modelfeatures(
        model, stim_start=0.5, stim_end=1, refreshKin=True
    )

    for f in Featurelist_:
        if modelF[f] == None:
            return [{}]


    model["Features"] = modelF

    if model["Features"]["freq_1.5e-10"]*model["Features"]["ISIavg_1.5e-10"] < 0.9: ## So that Depolarization blocks are taken care of
        return [{}]

    for rrow in Featurelist_:
        if model["Features"][rrow] < df_expsummaryactiveF.loc[rrow, "10th quantile"]:
            return [{}]
        if model["Features"][rrow] > df_expsummaryactiveF.loc[rrow, "90th quantile"]:
            return [{}]
    
    print('yoooohooooo', model["Features"]["DBL_1.5e-10"])
    return [model]

seeed = np.random.randint(0, 2**32 - 1)
# seeed = 1964186147
print(seeed)

# for i in range(1000):
#     model = ourfunc(i)

Multiprocessthis_appendsave(
   ourfunc, range(50000), [], ["tempactivemodels.pkl"], seed=seeed, npool=100
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
