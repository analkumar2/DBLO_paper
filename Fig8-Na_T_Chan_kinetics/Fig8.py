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
import moose

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(4, 10))
gs = fig.add_gridspec(3,1, hspace=0.3)

gs_inner = gs[0:2, 0].subgridspec(
    2, 1, hspace=0.05
)
axA = [0]*2
axA[0] = fig.add_subplot(gs_inner[0, 0])
axA[1] = fig.add_subplot(gs_inner[1, 0])

axB = fig.add_subplot(gs[2, 0])


# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA[0], axB]):
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

##############################
df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")

# Load models from the JSON file
basemodels_NaTallen_list = []
file_path = "activemodels_NaTallen.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        if (basemodel["Features"]["AP1_amp_1.5e-10"]>=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "10th quantile"]) & (basemodel["Features"]["AP1_amp_1.5e-10"]<=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "90th quantile"]):
            basemodels_NaTallen_list.append(basemodel)

DBLO_NaTallen_list = np.array([a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodels_NaTallen_list])
highDBLOmodel_NaTallen = np.array(basemodels_NaTallen_list)[np.argsort(DBLO_NaTallen_list)[-1]]

# Load models from the JSON file
basemodels_NaTRoyeck_list = []
file_path = "activemodels_NaTRoyeck.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        if (basemodel["Features"]["AP1_amp_1.5e-10"]>=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "10th quantile"]) & (basemodel["Features"]["AP1_amp_1.5e-10"]<=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "90th quantile"]):
            basemodels_NaTRoyeck_list.append(basemodel)

DBLO_NaTRoyeck_list = np.array([a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodels_NaTRoyeck_list])
highDBLOmodel_NaTRoyeck = np.array(basemodels_NaTRoyeck_list)[np.argsort(DBLO_NaTRoyeck_list)[-1]]

# Load models from the JSON file
basemodels_NaMig_list = []
file_path = "activemodels_NaMig.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        if (basemodel["Features"]["AP1_amp_1.5e-10"]>=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "10th quantile"]) & (basemodel["Features"]["AP1_amp_1.5e-10"]<=df_expsummaryactiveF.loc["AP1_amp_1.5e-10", "90th quantile"]):
            basemodels_NaMig_list.append(basemodel)

DBLO_NaMig_list = np.array([a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodels_NaMig_list])
highDBLOmodel_NaMig = np.array(basemodels_NaMig_list)[np.argsort(DBLO_NaMig_list)[-1]]


tvec,Ivec,Vmvec,Ca = mm.runModel(highDBLOmodel_NaTallen, refreshKin=True)
xgate = moose.element('library/Na_T_Chan/gateX')
v = np.linspace(xgate.min, xgate.max, xgate.divs+1)
axA[0].plot(v*1e3, (xgate.tableA/xgate.tableB)**moose.element('library/Na_T_Chan').Xpower, label='mInf_Gouwens', color='C2')
axA[1].plot(v*1e3, 1/xgate.tableB*1e3, label='Act_Gouwens', color='C2')
ygate = moose.element('library/Na_T_Chan/gateY')
axA[0].plot(v*1e3, (ygate.tableA/ygate.tableB)**moose.element('library/Na_T_Chan').Ypower, label='hInf_Gouwens', color='C2', linestyle='--')
axA[1].plot(v*1e3, 1/ygate.tableB*1e3, label='Inact_Gouwens', color='C2', linestyle='--')
moose.delete('library')

tvec,Ivec,Vmvec,Ca = mm.runModel(highDBLOmodel_NaTRoyeck, refreshKin=True)
xgate = moose.element('library/Na_T_Chan/gateX')
v = np.linspace(xgate.min, xgate.max, xgate.divs+1)
axA[0].plot(v*1e3, (xgate.tableA/xgate.tableB)**moose.element('library/Na_T_Chan').Xpower, label='mInf_Royeck', color='C8')
axA[1].plot(v*1e3, 1/xgate.tableB*1e3, label='Act_Royeck', color='C8')
ygate = moose.element('library/Na_T_Chan/gateY')
axA[0].plot(v*1e3, (ygate.tableA/ygate.tableB)**moose.element('library/Na_T_Chan').Ypower, label='hInf_Royeck', color='C8', linestyle='--')
axA[1].plot(v*1e3, 1/ygate.tableB*1e3, label='Inact_Royeck', color='C8', linestyle='--')
moose.delete('library')

tvec,Ivec,Vmvec,Ca = mm.runModel(highDBLOmodel_NaMig, refreshKin=True)
xgate = moose.element('library/Na_Chan/gateX')
v = np.linspace(xgate.min, xgate.max, xgate.divs+1)
axA[0].plot(v*1e3, (xgate.tableA/xgate.tableB)**moose.element('library/Na_Chan').Xpower, label='mInf_Migliore', color='C3')
axA[1].plot(v*1e3, 1/xgate.tableB*1e3, label='Act_Migliore', color='C3')
ygate = moose.element('library/Na_Chan/gateY')
axA[0].plot(v*1e3, (ygate.tableA/ygate.tableB)**moose.element('library/Na_Chan').Ypower, label='hInf_Migliore', color='C3', linestyle='--')
axA[1].plot(v*1e3, 1/ygate.tableB*1e3, label='Inact_Migliore', color='C3', linestyle='--')
moose.delete('library')

# axA[0].set_xlabel('Voltage (mV)')
axA[0].set_ylabel('Steady state (1)')
# axA[0].legend(frameon=False, loc='center right', bbox_to_anchor=[1.4,0.4])
axA[0].set_xlim(-100, 0)
axA[0].tick_params(bottom=False, labelbottom=False)

axA[1].set_xlabel('Voltage (mV)')
axA[1].set_ylabel('Tau (ms)')
axA[1].legend(frameon=False)
axA[1].set_xlim(-100, 0)


##########################################

######### Panel C ########################
df = pd.DataFrame(columns=["type", "DBLO150 (mV)"])
modeltype = (
    ["Gouwens"] * len(basemodels_NaTallen_list)
    + ["Royeck"] * len(basemodels_NaTRoyeck_list)
    + ["Migliore"] * len(basemodels_NaMig_list)
)
DBLO150 = np.concatenate((DBLO_NaTallen_list,DBLO_NaTRoyeck_list,DBLO_NaMig_list))
df.loc[:, "type"] = modeltype
df.loc[:, "DBLO150 (mV)"] = DBLO150
df = df.convert_dtypes()

GouwensvsRoyeckvsMigliore_kruskal = scipy.stats.kruskal(
    list(df[df["type"] == "Gouwens"].loc[:, "DBLO150 (mV)"]),
    list(df[df["type"] == "Royeck"].loc[:, "DBLO150 (mV)"]),
    list(df[df["type"] == "Migliore"].loc[:, "DBLO150 (mV)"]),
)
print('Kruskal-Wallis H-test ', GouwensvsRoyeckvsMigliore_kruskal)

GouwensvsRoyeckvsMigliore_dunn = sp.posthoc_dunn(df, val_col="DBLO150 (mV)", group_col="type", p_adjust='bonferroni')

print('Dunn’s test\n', GouwensvsRoyeckvsMigliore_dunn)
print('Mean DBLO at 150pA\n', df.groupby('type').mean()['DBLO150 (mV)'], df.groupby('type').std()['DBLO150 (mV)'])

def statannotator(ax, xpair_list, y, d, pvalues_list):
    d_ = 0
    for xpair, pvalue in zip(xpair_list, pvalues_list):
        ax.plot(xpair, [y + d_, y + d_], c="black")
        ax.text(np.mean(xpair), y + d_, pvalue, ha="center", va="bottom", c="black")
        d_ = d_ + d

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


order = ["Gouwens", "Royeck", "Migliore"]
palette = ["C2", "C8", "C3"]
ax = sns.boxplot(
    ax=axB,
    data=df,
    x="type",
    y="DBLO150 (mV)",
    order=order,
    palette=palette,
    showfliers=False,
    zorder=2
)
# ax = sns.violinplot(ax=axD, data=df, x='type', y='DBLO150 (mV)', order=order, palette=palette)
sns.stripplot(
    ax=ax,
    data=df,
    x="type",
    y="DBLO150 (mV)",
    order=order,
    palette=palette,
    size=2,
    zorder=3
)
statannotator(
    axB,
    [[0, 1], [0, 2]],
    df["DBLO150 (mV)"].max()+0.5,
    3,
    [
        convert_pvalue_to_asterisks(a)
        for a in [GouwensvsRoyeckvsMigliore_dunn.loc["Gouwens", "Royeck"], GouwensvsRoyeckvsMigliore_dunn.loc["Gouwens", "Migliore"]]
    ],
)
statannotator(
    axB,
    [[1, 2]],
    df[df["type"]=="Royeck"]["DBLO150 (mV)"].max(),
    0.01,
    [
        convert_pvalue_to_asterisks(a)
        for a in [GouwensvsRoyeckvsMigliore_dunn.loc["Royeck", "Migliore"]]
    ],
)


######################
sns.despine(fig=fig)
axA[0].spines["bottom"].set_visible(False)
plt.subplots_adjust(left=0.150)
# plt.tight_layout()
plt.savefig('Fig8.png', dpi=300)
# plt.savefig('Fig8.pdf', dpi=300)
plt.show()