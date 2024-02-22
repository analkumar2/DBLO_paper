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
import subprocess

from matplotlib.gridspec import GridSpec
import seaborn as sns
import scipy.stats as scs

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(8, 5), constrained_layout=False)
# fig = plt.figure(constrained_layout=False)
gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.3)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
# ax5 = fig.add_subplot(gs[2, 0])
# ax6 = fig.add_subplot(gs[2, 1])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB, axC, axD]):
    x_infig, y_infig = ax.transAxes.transform([0,1])
    x_infig = x_infig - 20
    y_infig = y_infig + 20
    x_ax, y_ax = ax.transAxes.inverted().transform([x_infig,y_infig])
    ax.text(
        x_ax,
        y_ax,
        f"{chr(65+i)}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )


#####################################
if not os.path.exists('activemodels.json'):
    subprocess.call(["python3", "getbasemodels.py"])
#######################################

# Load models from the JSON file
# df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")
basemodels_list = []
file_path = "activemodels.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        # if basemodel["Features"]["ISIavg_1.5e-10"]<df_expsummaryactiveF.loc["ISIavg_1.5e-10", "10th quantile"] or basemodel["Features"]["ISIavg_1.5e-10"]>df_expsummaryactiveF.loc["ISIavg_1.5e-10", "90th quantile"]:
        #     continue
        # if basemodel["Features"]["freq_1.5e-10"]<df_expsummaryactiveF.loc["freq_1.5e-10", "10th quantile"] or basemodel["Features"]["freq_1.5e-10"]>df_expsummaryactiveF.loc["freq_1.5e-10", "90th quantile"]:
        #     continue
        basemodels_list.append(basemodel)


### Exp without lJP correction ###
LJP = 15e-3
stim_start_exp = 0.3469
stim_end_exp = stim_start_exp+0.5
cell2 = expcells.expcell('2023_01_04_cell_2', f'../expdata/Chirp/2023_01_04_cell_2')
T_300pA, Vm_300pA = expcells.expdata(cell2.preFIfile, Index=16, LJP=LJP)
T_150pA, Vm_150pA = expcells.expdata(cell2.preFIfile, Index=10, LJP=LJP)
Features = fts.expfeatures(cellpath=cell2.preFIfile, stim_start=stim_start_exp, stim_end=stim_end_exp, LJP=LJP)

DBL150pA = [Features["DBL_1.5e-10"], Features["DBLO_1.5e-10"]]
DBL300pA = [Features["DBL_3e-10"], Features["DBLO_3e-10"]]

axB.plot((T_150pA-stim_start_exp)*1e3, Vm_150pA*1e3, label='150 pA', c='C0')
axB.plot((T_300pA-stim_start_exp)*1e3, Vm_300pA*1e3, label='300 pA', c='C9', alpha=0.5)
axB.set_xlabel('Time (ms)')
axB.set_ylabel('Voltage (mV)')
axB.set_title('Representative exp')
axB.set_xlim(-0.1*1e3, 0.6*1e3)
axB.set_ylim(-0.100*1e3, 0.05*1e3)
# axB.axhline(y=Features['E_rest_300'], color='black', linestyle='--', xmin=0, xmax=1)
axB.axhline(y=DBL150pA[0]*1e3, color='C0', linestyle='--')
axB.axhline(y=DBL300pA[0]*1e3, color='C9', linestyle='--')
axB.legend(frameon=False, loc='lower center', bbox_to_anchor=[0.5,-0.05])


#############################
EK_list = []
DBLO150pA_list = []
for model in basemodels_list:
    EK_list.append(model["Parameters"]["Channels"]["K_DR_Chan"]["Erev"])
    DBLO150pA_list.append(model["Features"]["DBLO_1.5e-10"])

highDBLmodelidx = np.argmax(DBLO150pA_list)
lowDBLmodelidx = np.argmin(DBLO150pA_list)

t150, Itrace150, Vtrace150, Ca = mm.runModel(basemodels_list[lowDBLmodelidx], 150e-12, refreshKin=True)
t300, Itrace300, Vtrace300, Ca = mm.runModel(basemodels_list[lowDBLmodelidx], 300e-12, refreshKin=False)
Features = fts.modelfeatures(basemodels_list[lowDBLmodelidx], 0.5, 1, refreshKin=False)
axC.plot((t150-0.5)*1e3, (Vtrace150)*1e3, label='150 pA', c='C3')
axC.plot((t300-0.5)*1e3, (Vtrace300)*1e3, label='300 pA', c='salmon', alpha=0.5)
axC.set_xlabel('Time (ms)')
axC.set_ylabel('Voltage (mV)')
axC.set_title('Representative low DBL model')
axC.set_xlim(-0.1*1e3, 0.6*1e3)
axC.set_ylim(-0.100*1e3, 0.05*1e3)
# axC.axhline(y=Features['E_rest_300'], color='black', linestyle='--', xmin=0, xmax=1)
axC.axhline(y=Features["DBL_1.5e-10"]*1e3, color='C3', linestyle='--')
axC.axhline(y=Features["DBL_3e-10"]*1e3, color='salmon', linestyle='--')
axC.legend(frameon=False, loc='center right', bbox_to_anchor=[1.2,0.85])

t150, Itrace150, Vtrace150, Ca = mm.runModel(basemodels_list[highDBLmodelidx], 150e-12, refreshKin=False)
t300, Itrace300, Vtrace300, Ca = mm.runModel(basemodels_list[highDBLmodelidx], 300e-12, refreshKin=False)
Features = fts.modelfeatures(basemodels_list[highDBLmodelidx], 0.5, 1, refreshKin=False)
axD.plot((t150-0.5)*1e3, (Vtrace150)*1e3, label='150 pA', c='C2')
axD.plot((t300-0.5)*1e3, (Vtrace300)*1e3, label='300 pA', c='lime', alpha=0.5)
axD.set_xlabel('Time (ms)')
# axD.set_ylabel('Voltage (mV)')
axD.set_title('Representative high DBL model')
axD.set_xlim(-0.1*1e3, 0.6*1e3)
axD.set_ylim(-0.100*1e3, 0.05*1e3)
# axD.axhline(y=Features['E_rest_300'], color='black', linestyle='--', xmin=0, xmax=1)
axD.axhline(y=Features["DBL_1.5e-10"]*1e3, color='C2', linestyle='--')
axD.axhline(y=Features["DBL_3e-10"]*1e3, color='lime', linestyle='--')
axD.legend(frameon=False, ncols=2)

axA.scatter(np.array(EK_list)*1e3, np.array(DBLO150pA_list)*1e3, c='C7')
axA.set_xlabel('K_DR Erev (mV)')
axA.set_ylabel('DBL (mV)')

######################
sns.despine(fig=fig)
# plt.tight_layout()
plt.savefig('Fig6.png', dpi=300)
plt.show()


############# Stats #####################
m, b, r, pvalue, _ = scs.linregress(np.array(EK_list)*1e3, np.array(DBLO150pA_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')