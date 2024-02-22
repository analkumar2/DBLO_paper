import sys

sys.path.insert(1, "../helperScripts")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import scikit_posthocs as sp
import os
# import subprocess
from scipy import signal

import expcells
import features as fts
import json
import moose
import pickle
import scipy
import MOOSEModel as mm
import MOOSEModel_ as mm_
import pingouin as pg
import efel
from goMultiprocessing import Multiprocessthis_appendsave

sns.set(style="ticks")
sns.set_context("paper")

# fig = plt.figure(figsize=(7, 10), constrained_layout=True)
fig = plt.figure(constrained_layout=False, dpi=100, figsize=[8, 10])
gs_outer = GridSpec(4, 2, figure=fig, height_ratios=[2, 1, 1, 1], wspace=0.5, hspace=0.5)
axA = fig.add_subplot(gs_outer[0, 0])
gs_inner = gs_outer[0, 1].subgridspec(
    2, 1, height_ratios=[2, 1], wspace=0.1, hspace=0.5
)
gs_inner_inner = gs_inner[0].subgridspec(
    2, 2, width_ratios=[9, 1], wspace=0.1, hspace=0.1
)
axB = [0] * 4
axB[0] = fig.add_subplot(gs_inner_inner[0, 0])
axB[1] = fig.add_subplot(gs_inner_inner[1, 0])
axB[2] = fig.add_subplot(gs_inner[1, 0])
axB[3] = fig.add_subplot(gs_inner_inner[:, 1])
gs_inner = gs_outer[1, 0].subgridspec(1, 2, wspace=0.5, hspace=0.2)
axC = [0] * 4
axC[0] = fig.add_subplot(gs_inner[0, 0])
axC[1] = fig.add_subplot(gs_inner[0, 1])
axD = fig.add_subplot(gs_outer[1:3, 1])
axE = fig.add_subplot(gs_outer[2, 0])
axF = fig.add_subplot(gs_outer[3, 0])
axG = fig.add_subplot(gs_outer[3, 1])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB[0], axC[0], axD, axE, axF, axG]):
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

############ Panel A #######################
image = plt.imread('ballandstick.png')
position = axA.get_position()
Aposition = axA.transAxes.transform(axA.get_children()[0].get_position())
Aposition[1] += 50
axA.imshow(image, aspect='equal')
left, bottom, width, height = position.x0, position.y0, position.width, position.height
zoom = 1.8
axA.set_position((left-0.08, bottom+height-zoom*height+0.06, width*image.shape[0]/image.shape[1]*zoom, height*zoom ))
axA.get_children()[0].set_position(axA.transAxes.inverted().transform(Aposition)) ##Panel label postition
axA.axis('off')
############################################

############ Panel B #######################
cellname = '2023_01_04_cell_3'
cell1 = expcells.expcell(cellname, f'../expdata/Chirp/{cellname}')
cell1.ampphase_freq()
cell1.chirpresponse(normalize=False)
ti=1
stimamp = 30e-12
t = np.arange(0, 13, cell1.chirpdt[ti])
chirp = np.zeros(int(300e-3/cell1.chirpdt[ti]))
chirp = np.concatenate([chirp, stimamp*np.sin(2*np.pi*t*t**2)])
chirp = np.concatenate([chirp, np.zeros(len(cell1.chirpV[ti]) - len(chirp))]) #Its in pA
t = np.arange(0, len(chirp)*cell1.chirpdt[ti], cell1.chirpdt[ti])

analytic_signal = signal.hilbert(chirp)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * 20000)

segs_c = [[(t[j], chirp[j]*1e12), (t[j+1], chirp[j+1]*1e12)] for j in range(len(t)-1)]
segs_v = [[(cell1.chirpT[ti][j], cell1.chirpV[ti][j]*1e3), (cell1.chirpT[ti][j+1], cell1.chirpV[ti][j+1]*1e3)] for j in range(len(cell1.chirpT[ti])-1)]
line_segments_c = LineCollection(segs_c, cmap='viridis', norm=plt.Normalize(0, max(instantaneous_frequency)), array=instantaneous_frequency)
line_segments_v = LineCollection(segs_v, cmap='viridis', norm=plt.Normalize(0, max(instantaneous_frequency)), array=instantaneous_frequency)
axB[0].add_collection(line_segments_c)
axB[1].add_collection(line_segments_v)

plt.colorbar(line_segments_c, label='Frequency (Hz)', cax=axB[3])

# axB[0].plot(t, chirp*1e12)
# axB[0].set_xlabel('Time (s)')
axB[0].set_ylabel('Current\n(pA)')
axB[0].set_xlim(min(t), max(t))
axB[0].set_ylim(min(chirp*1e12), max(chirp*1e12))
axB[0].tick_params(bottom=False, labelbottom=False)

# axB[1].plot(cell1.chirpT[i], cell1.chirpV[i]*1e3)
axB[1].set_xlabel('Time (s)')
axB[1].set_ylabel('Voltage\n(mV)')
axB[1].set_xlim(min(cell1.chirpT[ti]), max(cell1.chirpT[ti]))
axB[1].set_ylim(min(cell1.chirpV[ti]*1e3), max(cell1.chirpV[ti]*1e3))

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

if os.path.exists("Impedance_exp.pkl"):
    freqdf = pd.read_pickle("Impedance_exp.pkl")
else:    
    expcell_list = []
    for cell in tqdm(validcells):
        expcell_list.append(expcells.expcell(cell, f'../expdata/Chirp/{cell}'))

    for cell in tqdm(expcell_list):
        cell.ampphase_freq(True)

    freq_list = []
    impedance_list = []
    phase_list = []

    for cell in tqdm(expcell_list):
        freq_list.append(np.round(np.mean([cell.freq[i] for i in range(len(cell.freq))], axis=0), 5))
        impedance_list.append(np.mean([cell.impedance[i] for i in range(len(cell.freq))], axis=0))
        phase_list.append(np.mean([cell.phase[i] for i in range(len(cell.freq))], axis=0))

    freqdf = pd.DataFrame({"Frequency (Hz)": np.ravel(freq_list),
                       'Impedance ($\mathrm{M\Omega}$)': np.ravel(impedance_list)*1e-6,
                       'repeat': np.repeat(np.arange(0,len(freq_list)), len(freq_list[0]))})

    freqdf.to_pickle("Impedance_exp.pkl")

sns.lineplot(data=freqdf, x="Frequency (Hz)", y='Impedance ($\mathrm{M\Omega}$)', hue='repeat', errorbar=None, ax=axB[2], legend=False, palette=['grey'], alpha=0.2)
sns.lineplot(data=freqdf, x="Frequency (Hz)", y='Impedance ($\mathrm{M\Omega}$)', errorbar='se', ax=axB[2], legend=False, color='black')
axB[2].set_xscale('log')
axB[2].set_xlim(1,500)
#################################################################

############ Panel C #######################

# Load model from the JSON file
file_path = "../helperScripts/imp.json"
with open(file_path, "r") as file:
    for line in file:
        model_ = json.loads(line)
        if model_["Parameters"]["notes"] == cellname:
            model = model_
            break

####
LJP = 15e-3
samplingrate = 20000
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

axC[0].plot(np.mean([cell1.freq[i] for i in range(len(cell1.freq))],axis=0), np.mean([cell1.impedance[i] for i in range(len(cell1.impedance))],axis=0)*1e-6, label=cellname, c='C0')
axC[0].plot(model["Features"]["freq"], filtered_signal*1e-6, label=f'model', c='C2')
axC[0].set_xlabel('Frequency (Hz)')
axC[0].set_ylabel('Impedance\n($\mathrm{M\Omega}$)')
# axC[0].legend(frameon=False)
axC[0].set_xlim(1,500)
axC[0].set_xscale('log')

###########

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

cellT, cellV = expcells.expdata(cell1.preFIfile, 3)
cellV = cellV[(cellT>=0) & (cellT<1)]
cellT = cellT[(cellT>=0) & (cellT<1)]

axC[1].plot((cellT-stim_start_exp+0.5)*1e3, (cellV-LJP)*1e3, label='exp', c='C0')
axC[1].plot(modelT*1e3, modelV*1e3, label=f'model', c='C2')
axC[1].legend(frameon=False, loc='center left', bbox_to_anchor=(0.8, 0.5))
# axC[1].legend(frameon=False)
axC[1].set_xlabel('Time (ms)')
axC[1].set_ylabel('Voltage\n(mV)')

############################################################

############# Panel D and E ###############################################
basemodel_1compt_list = []
file_path = "../helperScripts/activemodels_1compt.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_1compt_list.append(basemodel)


basemodel_imp_list = []
file_path = "../helperScripts/activemodels_imp.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_imp_list.append(basemodel)

basemodel_pas_list = []
file_path = "../helperScripts/activemodels_pas.json"
with open(file_path, "r") as file:
    for line in tqdm(file):
        basemodel = json.loads(line)
        basemodel_pas_list.append(basemodel)

df = pd.DataFrame(columns=["modelID", "type", "DBLO150\n(mV)", "spikes"])
# df.loc[:,'notes'] = [a["Parameters"]["notes"] for a in basemodels_list]
modelID = np.tile(np.arange(0,len(basemodel_1compt_list)), 3)
modeltype = (
    ["1compt"] * len(basemodel_1compt_list)
    + ["imp"] * len(basemodel_imp_list)
    + ["pas"] * len(basemodel_pas_list)
)
DBLO150_1compt = [a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodel_1compt_list]
DBLO150_imp = [a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodel_imp_list]
DBLO150_pas = [a["Features"]["DBLO_1.5e-10"]*1e3 for a in basemodel_pas_list]
DBLO150 = (DBLO150_1compt + DBLO150_imp + DBLO150_pas)
spikes = (
    [a["Features"]["freq_1.5e-10"] / 2 for a in basemodel_1compt_list]
    + [a["Features"]["freq_1.5e-10"] / 2 for a in basemodel_imp_list]
    + [a["Features"]["freq_1.5e-10"] / 2 for a in basemodel_pas_list]
)

df.loc[:, "modelID"] = modelID
df.loc[:, "type"] = modeltype
df.loc[:, "DBLO150\n(mV)"] = DBLO150
df.loc[:, "spikes"] = spikes
df = df.convert_dtypes()

### Stats ###
impvs1comptvspas_anovaRM = pg.rm_anova(df, dv="DBLO150\n(mV)", within="modelID", subject="type")

print('Repeated Measures ANOVA ', impvs1comptvspas_anovaRM)

impvs1compt = pg.ttest(x=list(df[df["type"] == "1compt"].loc[:, "DBLO150\n(mV)"]), y=list(df[df["type"] == "imp"].loc[:, "DBLO150\n(mV)"]), paired=True)
impvspas = pg.ttest(x=list(df[df["type"] == "pas"].loc[:, "DBLO150\n(mV)"]), y=list(df[df["type"] == "imp"].loc[:, "DBLO150\n(mV)"]), paired=True)
pasvs1compt = pg.ttest(x=list(df[df["type"] == "1compt"].loc[:, "DBLO150\n(mV)"]), y=list(df[df["type"] == "pas"].loc[:, "DBLO150\n(mV)"]), paired=True)

print("paired ttest", "Imp vs 1compt", impvs1compt["p-val"].values[0])
print("paired ttest", "Imp vs pas", impvspas["p-val"].values[0])
print("paired ttest", "pas vs 1compt", pasvs1compt["p-val"].values[0])
print('Mean DBLO at 150pA\n', df.groupby('type').mean()['DBLO150\n(mV)'])

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


order = ["1compt", "pas", "imp"]
palette = ["C3", "C8", "C2"]
ax = sns.boxplot(
    ax=axD,
    data=df,
    x="type",
    y="DBLO150\n(mV)",
    order=order,
    palette=palette,
    showfliers=False,
    zorder=2
)
sns.stripplot(
    ax=ax,
    data=df,
    x="type",
    y="DBLO150\n(mV)",
    order=order,
    palette=palette,
    size=2,
    alpha=0.4,
    zorder=3
)

df_ = df[df["modelID"].isin(np.random.randint(0,len(basemodel_1compt_list), 100))]
sns.lineplot(
    ax=ax,
    data=df_, x="type", y="DBLO150\n(mV)", units="modelID",
    color=".7", estimator=None,
    zorder=0,
    alpha=0.2,
    linewidth = 1
)


statannotator(
    axD,
    [[0, 2], [0, 1], [1, 2]],
    df["DBLO150\n(mV)"].max(),
    2.5,
    [
        convert_pvalue_to_asterisks(a)
        for a in [impvs1compt["p-val"].values[0], pasvs1compt["p-val"].values[0], impvspas["p-val"].values[0]]
    ],
)

sns.lineplot(
    ax=axE,
    data=df,
    x="spikes",
    y="DBLO150\n(mV)",
    hue="type",
    hue_order=order,
    palette=palette,
    style="type",
    markers="o",
    markersize=5,
    err_style="bars",
    errorbar=("se", 1),
)
axE.legend(frameon=False, loc='upper right')

########### Panel F #################################
def ourfunc(model):
    tvec, Ivec, Vmvec,Cavec, Vmvec_dend0 = mm_.runModel(model, refreshKin=False)
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
    Vdiff = Vmvec - Vmvec_dend0
    Iaxial_trace = Vdiff/((moose.element('model/elec/soma').Ra + moose.element('model/elec/dend0').Ra)/2)
    charge = np.trapz(Iaxial_trace[(tvec>=stim_start) & ((tvec<stim_end))][peak0idx:troughidx], trace["T"][peak0idx:troughidx]*1e-3)
    return [charge, charge]

if os.path.exists("charge_impmodels.pkl"):
    charge_impmodels = []
    with open("charge_impmodels.pkl", 'rb') as f:
        while True:
            try:
                charge_impmodels.append(pickle.load(f))
            except Exception:
                break
    charge_impmodels = np.array(charge_impmodels)

    charge_pasmodels = []
    with open("charge_pasmodels.pkl", 'rb') as f:
        while True:
            try:
                charge_pasmodels.append(pickle.load(f))
            except Exception:
                break
    charge_pasmodels = np.array(charge_pasmodels)
else:
    charge_impmodels = []
    charge_pasmodels = []
    charge_impmodels = np.array(Multiprocessthis_appendsave(ourfunc, basemodel_imp_list, [charge_impmodels], ["charge_impmodels.pkl"], seed=1213, npool=0.99))
    charge_pasmodels = np.array(Multiprocessthis_appendsave(ourfunc, basemodel_pas_list, [charge_pasmodels], ["charge_pasmodels.pkl"], seed=1213, npool=0.99))

df["Backprop charge \n(pF)"] = np.concatenate([np.repeat(None, len(basemodel_1compt_list)),-1*charge_impmodels,-1*charge_pasmodels])
df_charge = df[df["type"] != "1compt"]
df_charge.loc[:,"Backprop charge \n(pF)"] = df_charge.loc[:,"Backprop charge \n(pF)"]*1e12

order = ["pas", "imp"]
palette = ["C8", "C2"]
ax = sns.boxplot(
    ax=axF,
    data=df_charge,
    x="type",
    y="Backprop charge \n(pF)",
    order=order,
    palette=palette,
    showfliers=False,
    zorder=2
)
sns.stripplot(
    ax=ax,
    data=df_charge,
    x="type",
    y="Backprop charge \n(pF)",
    order=order,
    palette=palette,
    size=2,
    alpha=0.4,
    zorder=3
)

df_charge_ = df_charge[df_charge["modelID"].isin(np.random.randint(0,len(basemodel_pas_list), 100))]
sns.lineplot(
    ax=ax,
    data=df_charge_, x="type", y="Backprop charge \n(pF)", units="modelID",
    color=".7", estimator=None,
    zorder=0,
    alpha=0.2,
    linewidth = 1
)

### Stats
impvspas = pg.ttest(x=list(df_charge[df_charge["type"] == "pas"].loc[:, "Backprop charge \n(pF)"]), y=list(df_charge[df_charge["type"] == "imp"].loc[:, "Backprop charge \n(pF)"]), paired=True)

print("paired ttest", "Imp vs pas", impvspas["p-val"].values[0])
print('Mean Axial Charge', df_charge.groupby('type').mean()["Backprop charge \n(pF)"])
print('std Axial Charge', df_charge.groupby('type').std()["Backprop charge \n(pF)"])


statannotator(
    axF,
    [[0, 1]],
    df_charge["Backprop charge \n(pF)"].max(),
    0,
    [
        convert_pvalue_to_asterisks(a)
        for a in [impvs1compt["p-val"].values[0], pasvs1compt["p-val"].values[0], pasvs1compt["p-val"].values[0]]
    ],
)


################### Panel G ##########################
highestDBLOdiffidx = np.argmax(np.array(DBLO150_imp)- np.array(DBLO150_pas))
tvec_imp, Ivec_imp, Vmvec_imp,Cavec_imp, Vmvec_dend0_imp = mm_.runModel(basemodel_imp_list[highestDBLOdiffidx], refreshKin=False)
Vdiff_imp = Vmvec_imp - Vmvec_dend0_imp
Iaxial_imp_trace = Vdiff_imp/((moose.element('model/elec/soma').Ra + moose.element('model/elec/dend0').Ra)/2)
axG.plot(tvec_imp*1e3, Vmvec_imp*1e3, label='imp', color="C2")

tvec_pas, Ivec_pas, Vmvec_pas,Cavec_pas, Vmvec_dend0_pas = mm_.runModel(basemodel_pas_list[highestDBLOdiffidx], refreshKin=False)
Vdiff_pas = Vmvec_pas - Vmvec_dend0_pas
Iaxial_pas_trace = Vdiff_pas/((moose.element('model/elec/soma').Ra + moose.element('model/elec/dend0').Ra)/2)
axG.plot(tvec_pas*1e3, Vmvec_pas*1e3, label='pas', color="C8")

axG.set_xlim(520,580)
axG.set_ylim(-85,-60)
axG.set_xlabel('Time (ms)')
axG.set_ylabel('Voltage\n(mV)')
axG.legend(frameon=False, loc='upper center')

axG_ = axG.twinx()
axG_.plot(tvec_imp*1e3, Iaxial_imp_trace*-1e9, label='imp', color="C2", linestyle='--')
axG_.plot(tvec_imp*1e3, Iaxial_pas_trace*-1e9, label='pas', color="C8", linestyle='--')

axG_.set_ylabel('Axial current (nA)')
axG_.set_ylim(-1,1)
#################################################################


## show plot ##
sns.despine(fig=fig)
axB[0].spines["bottom"].set_visible(False)
axG.spines["right"].set_visible(True)
plt.savefig('Fig5.png', dpi=300)
# plt.savefig('Fig5.svg', dpi=300)
# plt.savefig('Fig5.pdf', dpi=300)
plt.show()


########################### Some analysis ######################
print('1compt', np.sum(df[df["type"]=="1compt"]["DBLO150\n(mV)"]<10), len(df[df["type"]=="1compt"]["DBLO150\n(mV)"]) - np.sum(df[df["type"]=="1compt"]["DBLO150\n(mV)"]<10) - np.sum(df[df["type"]=="1compt"]["DBLO150\n(mV)"]>=15.4), np.sum(df[df["type"]=="1compt"]["DBLO150\n(mV)"]>=15.4) )
print('pas', np.sum(df[df["type"]=="pas"]["DBLO150\n(mV)"]<10), len(df[df["type"]=="pas"]["DBLO150\n(mV)"]) - np.sum(df[df["type"]=="pas"]["DBLO150\n(mV)"]<10) - np.sum(df[df["type"]=="pas"]["DBLO150\n(mV)"]>=15.4), np.sum(df[df["type"]=="pas"]["DBLO150\n(mV)"]>=15.4) )
print('imp', np.sum(df[df["type"]=="imp"]["DBLO150\n(mV)"]<10), len(df[df["type"]=="imp"]["DBLO150\n(mV)"]) - np.sum(df[df["type"]=="imp"]["DBLO150\n(mV)"]<10) - np.sum(df[df["type"]=="imp"]["DBLO150\n(mV)"]>=15.4), np.sum(df[df["type"]=="imp"]["DBLO150\n(mV)"]>=15.4) )

print(np.mean(np.array(df[df["type"]=="imp"]["DBLO150\n(mV)"]) - np.array(df[df["type"]=="1compt"]["DBLO150\n(mV)"])))