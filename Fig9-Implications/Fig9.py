import sys

sys.path.insert(1, "../helperScripts")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import scikit_posthocs as sp
# import os
# import subprocess
from scipy import signal
import scipy.stats as scs

import expcells
import features as fts
import json

# import pickle
import scipy
import MOOSEModel as mm
from matplotlib.cm import viridis
from matplotlib.colors import to_rgba
from copy import deepcopy

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(7, 10))
gs = fig.add_gridspec(2,1, hspace=0.3)

gs_inner = gs[0, 0].subgridspec(1, 3, wspace=0.5, width_ratios=[6,6,1])
axA = fig.add_subplot(gs_inner[0, 0])
axB = [0]*2
axB[0] = fig.add_subplot(gs_inner[0, 1])

gs_inner = gs[1, 0].subgridspec(1, 2)
axC = [0]*2
gs_inner_inner = gs_inner[0, 0].subgridspec(2, 1, height_ratios=[4,1])
axC[0] = [0]*2
axC[0][0] = fig.add_subplot(gs_inner_inner[0, 0])
axC[0][1] = fig.add_subplot(gs_inner_inner[1, 0])

gs_inner_inner = gs_inner[0, 1].subgridspec(3, 1, height_ratios=[2,2,1])
axC[1] = [0]*3
axC[1][0] = fig.add_subplot(gs_inner_inner[0, 0])
axC[1][1] = fig.add_subplot(gs_inner_inner[1, 0])
axC[1][2] = fig.add_subplot(gs_inner_inner[2, 0])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB[0], axC[0][0]]):
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

######## Panel A ######################
df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")

basemodel_imp_list = []
file_path = "activemodels_imp_Eb2_NaTallen.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        if (basemodel["Features"]["AP1_amp_1.5e-10"]>=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "10th quantile"]) & (basemodel["Features"]["AP1_amp_1.5e-10"]<=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "90th quantile"]):
            basemodel_imp_list.append(basemodel)

DBLO_list = np.array([a["Features"]["DBLO_1.5e-10"] for a in basemodel_imp_list])
highDBLOmodel = np.array(basemodel_imp_list)[np.argsort(DBLO_list)[-1]]
stim_start = 0.5
tvec, _, Vmvec, _ = mm.runModel(highDBLOmodel, 150e-12, refreshKin=True)
axA.plot((tvec-stim_start)*1e3, Vmvec*1e3, color='C2', label='high DBLO')
axA.set_xlabel('Time (ms)')
axA.set_ylabel('Voltage (mV)')
axA.set_xlim(-0.1*1e3, 0.6*1e3)
axA.set_ylim(-100e-3*1e3, 60e-3*1e3)
axA.set_title("Representative \n high DBLO model")
print(highDBLOmodel["Features"]["DBLO_1.5e-10"], sum(DBLO_list>15.4e-3))

#######################################
################## Panel B#####################
DBLOlist = [a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodel_imp_list]
Na_T_Chan_Gbar = [a["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"]*1e6 for a in basemodel_imp_list]
K_DR_Chan_Gbar = [a["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"]*1e6 for a in basemodel_imp_list]
Gbarratio = np.array(Na_T_Chan_Gbar)/np.array(K_DR_Chan_Gbar)
spikes = [a["Features"]["freq_1.5e-10"] / 2 for a in basemodel_imp_list]
baseID = [a["Parameters"]["notes"] for a in basemodel_imp_list]


cmap = viridis
norm = plt.Normalize(0, len(set(spikes)) - 1)
colors = [cmap(norm(list(set(spikes)).index(label))) for label in spikes]
for i in np.arange(min(spikes), max(spikes)+1,1):
    x = np.array(DBLOlist)[spikes==i]
    y = np.array(Gbarratio)[spikes==i]
    c = np.array(colors)[spikes==i]
    # axB[0].scatter(x, y, label=i, color=mpl.cm.viridis((i-min(spikes))/max(spikes)))
    axB[0].scatter(x, y, label=i, cmap=cmap, c=c)
    # axB[0].scatter(x, y, label=i, color="black")
    m, b, r, pvalue, _ = scs.linregress(x, y)
    print(i, f'{r:1.2f}', f'{pvalue*len(set(spikes)):1.2e}')
    # axB[0].plot(x, m*x + b, color=mpl.cm.viridis((i-min(spikes))/max(spikes)))
    axB[0].plot(x, m*x + b, c=c[0])

m, b, r, pvalue, _ = scs.linregress(DBLOlist, Gbarratio)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
axB[0].plot(np.array(DBLOlist), m*np.array(DBLOlist) + b, color='black', linewidth=5)
axB[0].set_xlabel('DBLO (mV)')
axB[0].set_ylabel(r'$\frac{Na\_T\_Chan\_Gbar}{K\_DR\_Chan\_Gbar}$')
axB[0].legend(frameon=False, title="num of spike", loc='center left', bbox_to_anchor=(1, 0.5))
################################################################################################################
################################################################################################################

basemodel_bis_list = []
file_path = "activemodels_bis.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_bis_list.append(basemodel)

stim_start = 0.5
stim_end = 1
stimlist_bis = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.5}) * {150e-12} + (t>{stim_start+1} & t<{stim_start+1.5}) * {-50e-12}",
]
tvec, Ivec, Vmvec, Ca = mm.runModel(basemodel_bis_list[0], CurrInjection=stimlist_bis, Truntime=2.5)

axC[0][0].plot(tvec*1e3, Vmvec*1e3, c='C2')
axC[0][0].set_ylabel('Voltage (mV)')
axC[0][0].get_xaxis().set_visible(False)
axC[0][0].set_title('Representative high DBLO \n bistable model')

axC[0][1].plot(tvec*1e3, Ivec*1e12, c='black')
axC[0][1].set_xlabel('Time (ms)')
axC[0][1].set_ylabel('Current (pA)')

##########

basemodel_list = []
file_path = "activemodels_part1.json"
with open(file_path, "r") as file:
    for i,line in tqdm(enumerate(file)):
        basemodel = json.loads(line)
        basemodel_list.append(basemodel)

file_path = "activemodels_part2.json"
with open(file_path, "r") as file:
    for i,line in tqdm(enumerate(file)):
        basemodel = json.loads(line)
        basemodel_list.append(basemodel)

file_path = "activemodels_part3.json"
with open(file_path, "r") as file:
    for i,line in tqdm(enumerate(file)):
        basemodel = json.loads(line)
        basemodel_list.append(basemodel)

file_path = "activemodels_part4.json"
with open(file_path, "r") as file:
    for i,line in tqdm(enumerate(file)):
        basemodel = json.loads(line)
        basemodel_list.append(basemodel)

stim_start = 0.5
stim_end = 1
stimlist_bis = [
    "soma",
    "1",
    ".",
    "inject",
    f"(t>{stim_start} & t<{stim_start+0.5}) * {150e-12} + (t>{stim_start+1} & t<{stim_start+1.5}) * {-50e-12}",
]
tvec, Ivec, Vmvec, Ca = mm.runModel(basemodel_list[0], CurrInjection=stimlist_bis, Truntime=2.5)

axC[1][0].plot(tvec*1e3, Vmvec*1e3, c='C3')
axC[1][0].set_ylabel('Voltage (mV)')
axC[1][0].get_xaxis().set_visible(False)
axC[1][0].set_title('Representative low DBLO \n non-bistable models')


DBLO_list = [a["Features"]["DBLO_1.5e-10"] for a in basemodel_list]
nonbismodel2 = deepcopy(basemodel_list[np.argmin(DBLO_list)])
nonbismodel2["Parameters"]["Channels"]["Na_P_Chan"]["Gbar"]= 2e-9 #### Need to do this because in the original run, one of the criterion is no firing at the initial 0pA
tvec, Ivec, Vmvec, Ca = mm.runModel(nonbismodel2, CurrInjection=stimlist_bis, Truntime=2.5)

axC[1][1].plot(tvec*1e3, Vmvec*1e3, c='C3')
axC[1][1].set_ylabel('Voltage (mV)')
axC[1][1].get_xaxis().set_visible(False)

axC[1][2].plot(tvec*1e3, Ivec*1e12, c='black')
axC[1][2].set_xlabel('Time (ms)')
axC[1][2].set_ylabel('Current (pA)')

######################
sns.despine(fig=fig)
axC[0][0].spines['bottom'].set_visible(False)
axC[1][0].spines['bottom'].set_visible(False)
axC[1][1].spines['bottom'].set_visible(False)
# plt.tight_layout()
plt.savefig('Fig9.png', dpi=300)
# plt.savefig('Fig8.pdf', dpi=300)
plt.show()