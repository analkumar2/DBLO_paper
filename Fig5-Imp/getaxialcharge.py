### Here we take all the models and then calculate the axial charge that comes into the soma from peak to trough.

import sys

sys.path.insert(1, "../helperScripts")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# import pandas as pd
import seaborn as sns
# from matplotlib.gridspec import GridSpec
# from matplotlib.collections import LineCollection
# import scikit_posthocs as sp
# import os
# import subprocess
# from scipy import signal

# import expcells
import features as fts
import json

import pickle
import scipy.stats as scs
import MOOSEModel_ as mm
import efel
import moose
from pprint import pprint
from goMultiprocessing import Multiprocessthis_appendsave

##### Import models ##################
basemodel_imp_list = []
file_path = "../../DBLpaper_v4/helperScripts/activemodels_imp.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_imp_list.append(basemodel)

basemodel_pas_list = []
file_path = "../../DBLpaper_v4/helperScripts/activemodels_pas.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_pas_list.append(basemodel)


#################################
stim_start=0.5
stim_end=1

def ourfunc(model):
    tvec, Ivec, Vmvec,Cavec, Vmvec_dend0 = mm.runModel(model, refreshKin=False)
    # pprint(F)
    # plt.plot(tvec, Vmvec)
    # plt.show()
    trace = {}
    trace["T"] = tvec[(tvec>=stim_start) & ((tvec<stim_end))] * 1e3
    trace["V"] = Vmvec[(tvec>=stim_start) & ((tvec<stim_end))] * 1e3
    trace["stim_start"] = [trace["T"][0]]
    trace["stim_end"] = [trace["T"][-1]]
    trace["stimulus_current"] = [150e-3]
    traces_results = efel.getFeatureValues(
        [trace],
        ["peak_indices", "min_between_peaks_indices"],
    )
    troughidx = traces_results[0]["min_between_peaks_indices"][0]*2
    peak0idx = traces_results[0]["peak_indices"][0]*2
    charge = np.trapz(Vmvec_dend0[(tvec>=stim_start) & ((tvec<stim_end))][peak0idx:troughidx], trace["T"][peak0idx:troughidx]*1e-3)
    return [charge]

Multiprocessthis_appendsave(ourfunc, basemodel_imp_list, [], ["charge_impmodels.pkl"], seed=1213, npool=0.8)
Multiprocessthis_appendsave(ourfunc, basemodel_pas_list, [], ["charge_pasmodels.pkl"], seed=1213, npool=0.8)