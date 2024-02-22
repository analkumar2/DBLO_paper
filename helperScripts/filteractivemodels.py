### We generated invalid models too. Filter these bad models

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

Featurelist = [
    # "CellID",
    # "E_rest_0",
    # "Input resistance",
    # "Cell capacitance",
    # "Time constant",
    # "sagV_m50",
    # "sagrat_m50",
    "AP1_amp_1.5e-10",
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
df_expsummaryactiveF = pd.read_pickle("expsummaryactiveF.pkl")

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
##############################################################################

# Load models from the JSON file
impmodels_list = []
file_path = "activemodels_imp_.json"
with open(file_path, "r") as file:
    for line in file:
        impmodel = json.loads(line)        
        impmodels_list.append(impmodel)

bnspasmodels_list = []
file_path = "activemodels_pas_.json"
with open(file_path, "r") as file:
    for line in file:
        bnspasmodel = json.loads(line)
        bnspasmodels_list.append(bnspasmodel)

onecomptmodels_list = []
file_path = "activemodels_1compt_.json"
with open(file_path, "r") as file:
    for line in file:
        onecomptmodel = json.loads(line)
        onecomptmodels_list.append(onecomptmodel)

# ##############################
# def ourfunc(model):
#     modelF = fts.modelfeatures(
#         model, stim_start=0.5, stim_end=1, refreshKin=False
#     )
#     model["Features"] = modelF
#     return [model]

# impmodels_list_ = []
# impmodels_list_ = Multiprocessthis_appendsave(
#        ourfunc, impmodels_list, [impmodels_list_], [], seed=0, npool=100
#     )
# bnspasmodels_list_ = []
# bnspasmodels_list_ = Multiprocessthis_appendsave(
#        ourfunc, bnspasmodels_list, [bnspasmodels_list_], [], seed=0, npool=100
#     )
# onecomptmodels_list_ = []
# onecomptmodels_list_ = Multiprocessthis_appendsave(
#        ourfunc, onecomptmodels_list, [onecomptmodels_list_], [], seed=0, npool=100
#     )

# for i in tqdm(range(len(onecomptmodels_list))):
#     isvalid=True
#     # for rrow in Featurelist:
#     #     if (impmodels_list[i]["Features"][rrow]<df_expsummaryactiveF.loc[rrow, "10th quantile"]) or (impmodels_list[i]["Features"][rrow]>df_expsummaryactiveF.loc[rrow, "90th quantile"]):
#     #         isvalid=False
#     #         break
#     #     if (bnspasmodels_list[i]["Features"][rrow]<df_expsummaryactiveF.loc[rrow, "10th quantile"]) or (bnspasmodels_list[i]["Features"][rrow]>df_expsummaryactiveF.loc[rrow, "90th quantile"]):
#     #         isvalid=False
#     #         break
#     #     if (onecomptmodels_list[i]["Features"][rrow]<df_expsummaryactiveF.loc[rrow, "10th quantile"]) or (onecomptmodels_list[i]["Features"][rrow]>df_expsummaryactiveF.loc[rrow, "90th quantile"]):
#     #         isvalid=False
#     #         break

#     if isvalid:
#         with open('activemodels_imp.json', "a") as file:
#             json.dump(impmodels_list_[i], file, cls=NpEncoder)
#             file.write("\n")
#         with open('activemodels_pas.json', "a") as file:
#             json.dump(bnspasmodels_list_[i], file, cls=NpEncoder)
#             file.write("\n")
#         with open('activemodels_1compt.json', "a") as file:
#             json.dump(onecomptmodels_list_[i], file, cls=NpEncoder)
#             file.write("\n")


with open('activemodels_imp.json', "a") as file:
    for i in range(len(impmodels_list[0])):
        json.dump(impmodels_list[0][i], file, cls=NpEncoder)
        file.write("\n")
with open('activemodels_pas.json', "a") as file:
    for i in range(len(impmodels_list[0])):
        json.dump(bnspasmodels_list[0][i], file, cls=NpEncoder)
        file.write("\n")
with open('activemodels_1compt.json', "a") as file:
    for i in range(len(impmodels_list[0])):
        json.dump(onecomptmodels_list[0][i], file, cls=NpEncoder)
        file.write("\n")
