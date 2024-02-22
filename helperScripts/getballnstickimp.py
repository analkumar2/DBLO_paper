### Here, we fit our ball and stick to exactly match the 14 exp recordings to get 14 validmodels

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

### get exp cell features ###
LJP = 15e-3
samplingrate = 20000

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
# validcells = validcells[:1]
df_chirp = pd.read_pickle("expchirp.pkl")

onecomptmodels_list = []
file_path = "1compt.json"
with open(file_path, "r") as file:
    for line in file:
        onecomptmodel = json.loads(line)
        onecomptmodels_list.append(onecomptmodel)
#########################################


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
            "dend_len": 500e-6,
            "dend_diam_start": 4e-6,
            "dend_diam_end": 4e-6,
            "num_dend_segments": 10,
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
            "h_Chan": {
                "Gbar": 7e-9,
                "Erev": -0.040,
                "Kinetics": "../Kinetics/h_Chan_Hay2011_exact",
            }
        },
    }
}

################################################################################
def ourfunc_helper(Em, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA, h_Chan_Gbar):
    model = deepcopy(baseModel)

    model["Parameters"]["Passive"]["Em"] = Em
    model["Parameters"]["Passive"]["sm_RM"] = sm_RM
    model["Parameters"]["Passive"]["sm_CM"] = sm_CM
    model["Parameters"]["Passive"]["sm_RA"] = sm_RA
    model["Parameters"]["Passive"]["dend_RM"] = dend_RM
    model["Parameters"]["Passive"]["dend_CM"] = dend_CM
    model["Parameters"]["Passive"]["dend_RA"] = dend_RA

    model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = h_Chan_Gbar

    return model

def ourfunc(ttt, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA):
    model = ourfunc_helper(Em, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA, h_Chan_Gbar)

    tchirp, Ichirp, Vchirp, Cavec = mm.runModel(
        model,
        CurrInjection=stimlist_chirp,
        vClamp=None,
        refreshKin=False,
        Truntime=tstop,
        syn=False,
        synwg=0,
        synfq=5,
    )
    freq_l, imp_l, ph_l = fts.calcImpedance(Ichirp, Vchirp, tchirp[1] - tchirp[0])
    model["Features"] = {}
    (
        model["Features"]["freq"],
        model["Features"]["impedance"],
        model["Features"]["phase"],
    ) = (
        freq_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
        imp_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
        ph_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
    )
    cutoff_freq = 50  # Frequency below which to keep the signal
    normalized_cutoff = cutoff_freq / (0.5 * len(model["Features"]["freq"]))
    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, model["Features"]["impedance"])
    return filtered_signal

restrict = [[0.1, 0.001, 0.1, 0.1, 0.001, 0.1],[10, 0.1, 10, 10, 0.1, 10]]
if not os.path.exists("imp.json"):
    for i in range(len(df_chirp)):
        for onecomptmodel in onecomptmodels_list:
            if onecomptmodel["Parameters"]["notes"] == df_chirp.loc[i, 'cellID']:
                break
        Em = onecomptmodel["Parameters"]["Passive"]["Em"]
        h_Chan_Gbar = onecomptmodel["Parameters"]["Channels"]["h_Chan"]["Gbar"]
        paramfitted, error = bcf.brute_scifit(
            ourfunc,
            [1,2,3],
            df_chirp.loc[i,'impedance'],
            restrict,
            ntol=1000,
            returnnfactor=0.01,
            maxfev=1000,
            printerrors=False,
            parallel=10,
            savetofile=False,
        )
        print(paramfitted, error)
        model = ourfunc_helper(Em,*paramfitted, h_Chan_Gbar)
        model["Parameters"]["notes"] = df_chirp.loc[i, 'cellID']
        with open("imp.json", "a") as impexactvalidfile:
            json.dump(model, impexactvalidfile)
            impexactvalidfile.write("\n")

# ########################################################
# ##### Now, fitting to -25pA response ######
# # Load models from the JSON file
# basemodels_list = []
# file_path = "imp.json"
# with open(file_path, "r") as file:
#     for line in file:
#         basemodel = json.loads(line)
#         basemodels_list.append(basemodel)

# def ourfunc_helper(Em, h_Chan_Gbar):
#     model = deepcopy(baseModel)

#     model["Parameters"]["Passive"]["Em"] = Em
#     model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = h_Chan_Gbar

#     return model

# def ourfunc(ttt, Em, h_Chan_Gbar):
#     model = ourfunc_helper(Em, h_Chan_Gbar)

#     tm25, Ivec, Vtracem25, Cavec = mm.runModel(
#         model,
#         CurrInjection=-25e-12,
#         vClamp=None,
#         refreshKin=False,
#         Truntime=None,
#         syn=False,
#         synwg=0,
#         synfq=5,
#     )
#     stim_start_exp = 0.3469
#     return Vtracem25[(tm25>=0.5-stim_start_exp) & (tm25<0.5-stim_start_exp+1)]

# if not os.path.exists("imp_pas.json"):
#     for baseModel in tqdm(basemodels_list):
#         cell = expcells.expcell(baseModel["Parameters"]["notes"], f"../expdata/Chirp/{baseModel['Parameters']['notes']}")
#         cellT, cellV = expcells.expdata(cell.preFIfile, 3)
#         cellV = cellV-LJP
#         stim_start_exp = 0.3469
#         cellV = cellV[(cellT>=0) & (cellT<1)]

#         restrict = [[-100e-3, 5], [-78.5e-03, 50]]
#         paramfitted, error = bcf.brute_scifit(
#             ourfunc,
#             [1,2,3],
#             cellV,
#             restrict,
#             ntol=1000,
#             returnnfactor=0.01,
#             maxfev=1000,
#             printerrors=False,
#             parallel=True,
#             savetofile=False,
#         )
#         print(paramfitted, error)
#         model = ourfunc_helper(*paramfitted)
#         with open("imp_pas.json", "a") as impexactvalidfile:
#             json.dump(model, impexactvalidfile)
#             impexactvalidfile.write("\n")


# #########################################################################

########### Checking the fits ###########################################
# Load models from the JSON file
models_list = []
file_path = "imp.json"
with open(file_path, "r") as file:
    for line in file:
        model = json.loads(line)
        models_list.append(model)

for model in tqdm(models_list):
    fig,ax = plt.subplots(1,2)
    tm25, Ivec, Vtracem25, Cavec = mm.runModel(
        model,
        CurrInjection=-25e-12,
        vClamp=None,
        refreshKin=False,
        Truntime=None,
        syn=False,
        synwg=0,
        synfq=5,
    )
    stim_start_exp = 0.3469
    modelT = tm25[(tm25>=0.5-stim_start_exp) & (tm25<0.5-stim_start_exp+1)]
    modelV = Vtracem25[(tm25>=0.5-stim_start_exp) & (tm25<0.5-stim_start_exp+1)]

    cell = expcells.expcell(model["Parameters"]["notes"], f"../expdata/Chirp/{model['Parameters']['notes']}")
    cellT, cellV = expcells.expdata(cell.preFIfile, 3)
    cellV = cellV[(cellT>=0) & (cellT<1)]
    cellT = cellT[(cellT>=0) & (cellT<1)]

    ax[1].plot(cellT-stim_start_exp+0.5, cellV-LJP, label=f'{model["Parameters"]["notes"]}', c='#1f77b4')
    ax[1].plot(modelT, modelV, label=f'model', c='#2ca02c')
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Voltage (V)')
    ax[1].set_title('Response to -25pA current injection')

    ####
    cellchirpImp = df_chirp[df_chirp['cellID'] == model["Parameters"]["notes"]]['impedance'].values[0]
    cellchirpfreq = df_chirp[df_chirp['cellID'] == model["Parameters"]["notes"]]['freq'].values[0]

    tchirp, Ichirp, Vchirp, Cavec = mm.runModel(
        model,
        CurrInjection=stimlist_chirp,
        vClamp=None,
        refreshKin=False,
        Truntime=tstop,
        syn=False,
        synwg=0,
        synfq=5,
    )
    freq_l, imp_l, ph_l = fts.calcImpedance(Ichirp, Vchirp, tchirp[1] - tchirp[0])
    model["Features"] = {}
    (
        model["Features"]["freq"],
        model["Features"]["impedance"],
        model["Features"]["phase"],
    ) = (
        freq_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
        imp_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
        ph_l[(freq_l > 0.5) & (freq_l <= 500)].tolist(),
    )
    cutoff_freq = 50  # Frequency below which to keep the signal
    normalized_cutoff = cutoff_freq / (0.5 * len(model["Features"]["freq"]))
    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, model["Features"]["impedance"])

    ax[0].plot(cellchirpfreq, cellchirpImp, label=f'{model["Parameters"]["notes"]}', c='#1f77b4')
    ax[0].plot(model["Features"]["freq"], filtered_signal, label=f'model', c='#2ca02c')
    ax[0].legend()
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Impedance (ohms)')
    ax[0].set_title('Response to chirp injection')
    ax[0].set_xscale('log')

    plt.savefig(f'plots_imp/{model["Parameters"]["notes"]}.png', dpi=300)
    plt.close('all')

####################################################################################
## Now getting active models ########################################

