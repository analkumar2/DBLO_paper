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
# from copy import deepcopy
from tqdm import tqdm
# import pandas as pd
from pprint import pprint
# from goMultiprocessing import Multiprocessthis_appendsave
import pickle
import json
# from scipy import signal
# import warnings
from copy import deepcopy

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
# validcells = validcells[4:5]

stim_start_exp = 0.3469
stim_end_exp = stim_start_exp + 0.5
LJP = 15e-3
stimamp = 30e-12
stim_start_chirp = 0.3
stim_end_chirp = 13.3
stim_start = 0.5
stim_end = 1
tstop = 14.5
restrict = [[0,0,0, 0,0,0], [20,0.2,20, 20,0.2,20]]

onecomptmodels_list = []
file_path = "1compt.json"
with open(file_path, "r") as file:
    for line in file:
        onecomptmodel = json.loads(line)
        onecomptmodels_list.append(onecomptmodel)

def ourfunc_helper1(basemodel, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA):
    model = deepcopy(basemodel)
    model["Parameters"]["Passive"]["sm_RM"] = sm_RM
    model["Parameters"]["Passive"]["sm_CM"] = sm_CM
    model["Parameters"]["Passive"]["sm_RA"] = sm_RA
    model["Parameters"]["Passive"]["dend_RM"] = dend_RM
    model["Parameters"]["Passive"]["dend_CM"] = dend_CM
    model["Parameters"]["Passive"]["dend_RA"] = dend_RA
    return model

def ourfunc_helper2(model):
    tm25, Im25, Vtracem25, Ca = mm.runModel(model, -25e-12, refreshKin=False)
    modelF = fts.calcRinCin(
            tm25,
            Vtracem25,
            stim_start=0.5,
            stim_end=1,
        )

    return [modelF[1]*1e-6, modelF[3]*1e4]

def ourfunc(ttt, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA):
    model = ourfunc_helper1(basemodel, sm_RM, sm_CM, sm_RA, dend_RM, dend_CM, dend_RA)
    modelF = ourfunc_helper2(model)
    return modelF

for cells in tqdm(validcells):
    print(cells)
    cell = expcells.expcell(cells, f"../expdata/Chirp/{cells}")
    texp, Vexp = expcells.expdata(cell.preFIfile, 3, LJP=15e-3)
    expF = fts.calcRinCin(
                texp,
                Vexp,
                stim_start=stim_start_exp,
                stim_end=stim_end_exp,
            )

    expF = [expF[1]*1e-6, expF[3]*1e4]
    print(expF)
    for basemodel in onecomptmodels_list:
        if basemodel["Parameters"]["notes"] == cells:
            break
    basemodel["Parameters"]["Morphology"]["num_dend_segments"] = 10

    ################################################################################
    paramfitted, error = bcf.brute_scifit(
        ourfunc,
        [1,2,3],
        expF,
        restrict,
        ntol=10000,
        returnnfactor=0.01,
        maxfev=10000,
        printerrors=False,
        parallel=100,
        savetofile=f'bnspasmodels_{cells}.pkl',
    )
    os.rename(f'sf_bnspasmodels_{cells}.pkl', f'plots_pas/sf_bnspasmodels_{cells}.pkl')
    os.rename(f'bf_bnspasmodels_{cells}.pkl', f'plots_pas/bf_bnspasmodels_{cells}.pkl')
    with open(f'plots_pas/sf_bnspasmodels_{cells}.pkl', 'rb') as f:
        paramfitted_list, error_list = pickle.load(f)
    print(paramfitted_list, error_list)
    paramfitted_list = np.array(paramfitted_list)[np.argsort(error_list)][:10]
    error_list = np.array(error_list)[np.argsort(error_list)][:10]
    # paramfitted_list, error_list = [[[0.12,  0.25]], [99999]]
    # print(paramfitted_list, error_list)

    fittedmodel_list = []
    plt.plot(texp- stim_start_exp, Vexp, label='exp')
    for paramfitted,error in zip(paramfitted_list, error_list):
        print(paramfitted, error)
        fittedmodel = ourfunc_helper1(basemodel, *paramfitted)
        fittedmodel_list.append(fittedmodel)
        fittedmodelF = ourfunc_helper2(fittedmodel)
        print(expF)
        print(fittedmodelF)
        tmodel_fitted, Imodel_fitted, Vmodel_fitted, Ca = mm.runModel(fittedmodel, -25e-12, refreshKin=False)
        plt.plot(tmodel_fitted- stim_start, Vmodel_fitted, c='C2')
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane potential (V)')
    plt.legend()
    plt.savefig(f'plots_pas/bnspas_{cells}.png')
    plt.close('all')
    # plt.show()
    # break

    validbnspas_path = "pas.json"
    with open(validbnspas_path, "a") as validbnspasfile:
        for model in fittedmodel_list:
            json.dump(model, validbnspasfile)
            validbnspasfile.write("\n")


#################################################################################


