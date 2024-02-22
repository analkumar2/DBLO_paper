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
import efel

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
    # "ISIavg_1.5e-10",
    "ISImedian_1.5e-10",
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
df_expsummaryactiveF = pd.read_pickle(
    "../helperScripts/expsummaryactiveF.pkl"
)

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
    f"(t>{stim_start} & t<{stim_start+0.5}) * {150e-12} + (t>{stim_start+1} & t<{stim_start+1.5}) * {-50e-12}",
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
            "num_dend_segments": 10,
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
            "Na_P_Chan": {
                "Gbar": 1e-9,
                "Erev": 0.060,
                "Kinetics": "../Kinetics/Na_P_Chan_Migliore2018",
                "KineticVars": {"mvhalf": -52.3e-3, "mvslope": 6.8e-3},
            },
            "K_DR_Chan": {
                "Gbar": 1e-3,
                "Erev": -0.090,
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

# Load models from the JSON file
basemodels_list = []
file_path = "../helperScripts/activemodels_imp.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        basemodels_list.append(basemodel)


######################################################################################
def ourfunc(i):
    model = basemodels_list[np.random.randint(0, len(basemodels_list))]
    model["Parameters"]["Channels"]["h_Chan"]["Kinetics"] = baseModel["Parameters"]["Channels"]["h_Chan"]["Kinetics"]
    model["Parameters"]["Channels"]["Na_T_Chan"]["Kinetics"] = baseModel["Parameters"]["Channels"]["Na_T_Chan"]["Kinetics"]
    model["Parameters"]["Channels"]["K_DR_Chan"]["Kinetics"] = baseModel["Parameters"]["Channels"]["K_DR_Chan"]["Kinetics"]
    model["Parameters"]["Channels"]["Na_P_Chan"] = baseModel["Parameters"]["Channels"]["Na_P_Chan"]

    model["Parameters"]["Channels"]["Na_P_Chan"]["Gbar"] = 10 ** np.random.uniform(
        -11, -7
        # -9, -8,
    )
    model["Parameters"]["Channels"]["Na_P_Chan"]["KineticVars"][
        "mvhalf"
    ] = np.random.uniform(-80e-3, -50e-3)
    # ] = np.random.uniform(-75e-3, -70e-3)
    model["Parameters"]["Channels"]["Na_P_Chan"]["KineticVars"][
        "mvslope"
    ] = np.random.uniform(2e-3, 15e-3)
    # ] = np.random.uniform(4e-3, 5e-3)

    model["Parameters"]["Passive"]["sm_RM"] = 10 ** np.random.uniform(-1,1.3)

    ### Get bis features #####
    model["Features"] = {}
    tvec, Ivec, Vmvec, Cavec = mm.runModel(
        model,
        CurrInjection=stimlist_bis,
        vClamp=None,
        refreshKin=True,
        Truntime=2.5,
        syn=False,
        synwg=0,
        synfq=5,
    )
    plt.plot(tvec, Vmvec)
    plt.show()
    tracebis = {}
    tracebis["T"] = tvec * 1e3
    tracebis["V"] = Vmvec * 1e3

    ###### 0pA condition #########
    model["Features"]["E_rest"] = np.nanmedian(Vmvec[(tvec>=stim_start-0.1) & ((tvec<stim_start))])
    tracebis["stim_start"] = [0]
    tracebis["stim_end"] = [(stim_start) * 1e3]
    tracebis["stimulus_current"] = [0e-3]
    traces_results = efel.getFeatureValues(
        [tracebis],
        ["Spikecount_stimint"],
    )
    trace_result = traces_results[0]
    try:
        if trace_result["Spikecount_stimint"][0] > 1:  ########## Should not fire at 0pA
            return [{}]
        model["Features"]["freq_0"] = trace_result["Spikecount_stimint"][0] * 2
        # pprint(model["Features"])
    except Exception:
        return [{}]

    ####### 150pA condition ###########
    tracebis_ = {}
    tracebis_["T"] = tvec[(tvec>=stim_start) & ((tvec<stim_start+0.5))] * 1e3
    tracebis_["V"] = Vmvec[(tvec>=stim_start) & ((tvec<stim_start+0.5))] * 1e3
    tracebis_["stim_start"] = [tracebis["T"][0]]
    tracebis_["stim_end"] = [tracebis["T"][-1]]
    tracebis_["stimulus_current"] = [150e-3]
    traces_results = efel.getFeatureValues(
        [tracebis_],
        ["Spikecount_stimint", "AP1_amp", "all_ISI_values", "min_between_peaks_values"],
    )
    trace_result = traces_results[0]
    try:
        # if (trace_result["Spikecount_stimint"][0]*2<df_expsummaryactiveF.loc["freq_1.5e-10", "10th quantile"]) or (trace_result["Spikecount_stimint"][0]*2>df_expsummaryactiveF.loc["freq_1.5e-10", "90th quantile"]):
        #     return [{}]
        # if (np.nanmedian(trace_result["all_ISI_values"]) * 1e-3<df_expsummaryactiveF.loc["ISImedian_1.5e-10", "10th quantile"]) or (np.nanmedian(trace_result["all_ISI_values"]) * 1e-3>df_expsummaryactiveF.loc["ISImedian_1.5e-10", "90th quantile"]):
        #     return [{}]
        # if (trace_result["AP1_amp"][0] * 1e-3 < df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "10th quantile"]) or (trace_result["AP1_amp"][0] * 1e-3 > df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "90th quantile"]):
        #     return [{}]
        model["Features"]["freq_1.5e-10"] = trace_result["Spikecount_stimint"][0] * 2
        model["Features"]["ISImedian_1.5e-10"] = (
            np.nanmedian(trace_result["all_ISI_values"]) * 1e-3
        )
        model["Features"]["AP1_amp_1.5e-10"] = trace_result["AP1_amp"][0] * 1e-3
        model["Features"]["DBLO_1.5e-10"] = (
            np.nanmean(trace_result["min_between_peaks_values"]) * 1e-3
            - model["Features"]["E_rest"]
        )
        model["Features"]["DBL_1.5e-10"] = (
            np.nanmean(trace_result["min_between_peaks_values"]) * 1e-3
        )
        if trace_result["Spikecount_stimint"][0]*np.nanmean(trace_result["all_ISI_values"]) * 1e-3<0.45:
            return [{}]
        # pprint(model["Features"])
    except Exception:
        return [{}]

    ####### second 0pA condition ###########
    tracebis["stim_start"] = [(stim_start + 0.5) * 1e3]
    tracebis["stim_end"] = [(stim_start + 1) * 1e3]
    tracebis["stimulus_current"] = [0e-3]
    traces_results = efel.getFeatureValues(
        [tracebis],
        ["Spikecount_stimint"],
    )
    trace_result = traces_results[0]
    try:
        if trace_result["Spikecount_stimint"][0] < 2:
            return [model]
        model["Features"]["freq_s0"] = trace_result["Spikecount_stimint"][0] * 2
        pprint(model["Features"])
    except Exception:
        return [model]

    ######## -50pA condition ##############
    tracebis["stim_start"] = [(stim_start + 1) * 1e3]
    tracebis["stim_end"] = [(stim_start + 1.5) * 1e3]
    tracebis["stimulus_current"] = [-50e-3]
    traces_results = efel.getFeatureValues(
        [tracebis],
        ["Spikecount_stimint"],
    )
    trace_result = traces_results[0]
    try:
        if (
            trace_result["Spikecount_stimint"][0] > 1
        ):  ########## Should not fire at -50pA. Leeway of 2spikes due to Na resurgent firing
            return [model]
        model["Features"]["freq_m50"] = trace_result["Spikecount_stimint"][0] * 2
        pprint(model["Features"])
    except Exception:
        return [model]

    ####### third 0pA condition ###########
    tracebis["stim_start"] = [(stim_start + 1.5) * 1e3]
    tracebis["stim_end"] = [(stim_start + 2) * 1e3]
    tracebis["stimulus_current"] = [0e-3]
    traces_results = efel.getFeatureValues(
        [tracebis],
        ["Spikecount_stimint"],
    )
    trace_result = traces_results[0]
    try:
        if (
            trace_result["Spikecount_stimint"][0] > 1
        ):  ########## Should not fire at -50pA. Leeway of 1spikes due to Na resurgent firing
            return [model]
        model["Features"]["freq_t0"] = trace_result["Spikecount_stimint"][0] * 2
        pprint(model["Features"])
    except Exception:
        return [model]

    print("yoooohooooo", model["Features"]["DBLO_1.5e-10"])
    return [model]


seeed = np.random.randint(0, 2**32 - 1)
# seeed = 1964186147
print(seeed)

# for i in range(1000):
#     model = ourfunc(i)

Multiprocessthis_appendsave(
   ourfunc, range(500000), [], ["tempactivemodels.pkl"], seed=seeed, npool=100
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


with open("tempactivemodels.pkl", "rb") as f, open("activemodels.json", "a") as file:
    while True:
        model = pickle.load(f)
        # pprint(model)
        if len(model) > 0:
            json.dump(model, file, cls=NpEncoder)
            file.write("\n")
