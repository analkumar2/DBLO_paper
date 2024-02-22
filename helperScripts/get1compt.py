### Here, we fit our 1compt models to exactly match the 13 exp recordings to get 13 validmodels

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4

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
from time import time
import moose

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
df_chirp = pd.read_pickle("../helperScripts/expchirp.pkl")
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
            "num_dend_segments": 0,
        },
        "Passive": {
            "Em": -82e-3,
            "sm_RM": 0.3, #Gouwens had 0.1169
            "sm_CM": 0.01, #Gouwens had 0.01
            "sm_RA": 1.18, #Gouwens had 1.18
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

# ########################################################
##### Now, fitting to -25pA response ######
import h_Chan_Hay2011_exact
moose.Neutral('library')
h_Chan = h_Chan_Hay2011_exact.h_Chan('h_Chan')
xgate = moose.element( h_Chan.path + '/gateX' )
v = np.linspace(xgate.min,xgate.max, xgate.divs+1)
h_Chan_Erev = baseModel["Parameters"]["Channels"]["h_Chan"]["Erev"]
sm_area = np.pi*baseModel["Parameters"]["Morphology"]["sm_len"]*baseModel["Parameters"]["Morphology"]["sm_diam"]

def ourfunc_helper(Em, sm_RM, sm_CM, h_Chan_Gbar):
    model = deepcopy(baseModel)

    model["Parameters"]["Passive"]["Em"] = Em
    model["Parameters"]["Passive"]["sm_RM"] = sm_RM
    model["Parameters"]["Passive"]["sm_CM"] = sm_CM
#     model["Parameters"]["Passive"]["sm_RA"] = sm_RA
#     model["Parameters"]["Passive"]["dend_RM"] = dend_RM
#     model["Parameters"]["Passive"]["dend_CM"] = dend_CM
#     model["Parameters"]["Passive"]["dend_RA"] = dend_RA

    model["Parameters"]["Channels"]["h_Chan"]["Gbar"] = h_Chan_Gbar

    return model

def ourfunc(ttt, sm_RM, sm_CM, h_Chan_Gbar):
    Em = (E_ssm25*h_Chan_Gbar*m_0*(h_Chan_Erev-E_rest) - E_rest*h_Chan_Gbar*m_m25*(h_Chan_Erev - E_ssm25) - E_rest*(-25e-12))/ (h_Chan_Gbar*m_0*(h_Chan_Erev-E_rest) - h_Chan_Gbar*m_m25*(h_Chan_Erev - E_ssm25) - (-25e-12))
    model = ourfunc_helper(Em, sm_RM, sm_CM, h_Chan_Gbar)

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
    toreturn = Vtracem25[(tm25>=0.5-stim_start_exp) & (tm25<0.5-stim_start_exp+1)]
    return toreturn

for validcell in tqdm(validcells):          
    cell = expcells.expcell(validcell, f"../expdata/Chirp/{validcell}")
    cellT, cellV = expcells.expdata(cell.preFIfile, 3, LJP=15e-3)
    cellV = cellV[(cellT>=0) & (cellT<1)]
    cellT = cellT[(cellT>=0) & (cellT<1)]

    E_rest = np.median(cellV[cellT<stim_start_exp])
    E_ssm25 = np.median(cellV[(cellT>=stim_start_exp+0.4) & (cellT<stim_end_exp)])
    m_0 = np.interp(E_rest, v, xgate.tableA/xgate.tableB)
    m_m25 = np.interp(E_ssm25, v, xgate.tableA/xgate.tableB)

    restrict = [[0.07, 0.057, 1e-9], [0.4, 0.3, 7e-8]]
    paramfitted, error = bcf.brute_scifit(
        ourfunc,
        [1,2,3],
        cellV,
        restrict,
        ntol=1000,
        returnnfactor=0.01,
        maxfev=1000,
        printerrors=True,
        parallel=20,
        savetofile=False,
    )
    # paramfitted, error = [[9.57529200e-02, 2.90396550e-01, 6.15196805e-09], 15.079058931250575]
    print(paramfitted, error)
    h_Chan_Gbar = paramfitted[2]
    Em = (E_ssm25*h_Chan_Gbar*m_0*(h_Chan_Erev-E_rest) - E_rest*h_Chan_Gbar*m_m25*(h_Chan_Erev - E_ssm25) - E_rest*(-25e-12))/ (h_Chan_Gbar*m_0*(h_Chan_Erev-E_rest) - h_Chan_Gbar*m_m25*(h_Chan_Erev - E_ssm25) - (-25e-12))
    model = ourfunc_helper(Em, *paramfitted)
    model["Parameters"]["notes"] = validcell
    with open("1compt.json", "a") as validfile:
        json.dump(model, validfile)
        validfile.write("\n")


#########################################################################

########### Checking the fits ###########################################
# Load models from the JSON file
models_list = []
file_path = "1compt.json"
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

    plt.savefig(f'plots_1compt/{model["Parameters"]["notes"]}.png', dpi=300)
    plt.close('all')

####################################################################################
## Now getting active models ########################################

