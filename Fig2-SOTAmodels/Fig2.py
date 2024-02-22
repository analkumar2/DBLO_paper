import sys 
sys.path.insert(1, "../helperScripts")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import features as fts
from matplotlib.gridspec import GridSpec
from neuron import h,gui
import os
import subprocess
from scipy.signal import butter, filtfilt

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(7, 8), constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])
# gs_inner = gs[3, :].subgridspec(1, 4)
# ax7 = [0]*4
# ax7[0] = fig.add_subplot(gs_inner[0])
# ax7[1] = fig.add_subplot(gs_inner[1])
# ax7[2] = fig.add_subplot(gs_inner[2])
# ax7[3] = fig.add_subplot(gs_inner[3])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB, axC, axD, axE, axF]):
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

stim_start = 0.200
stim_end = stim_start+0.5
LJP = 15e-3 ### Delete this. Not needed in this particular script

df_features = pd.DataFrame()

#### First Migliore 2018 without LJP correction ###
model_dir = 'MiglioreEtAl2018PLOSCompBiol2018_o'
# result1 = subprocess.run(['nrnivmodl'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
# result1 = subprocess.run(['python3', 'Migliore2018CA1pyrIclamp.py'], cwd=model_dir, capture_output=True, text=True) ### RUn this if redoing simulations
# print(result1)
output = np.load(f'{model_dir}/output.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axA.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C3')
line, = axB.plot([], alpha = 0)
axA.set_xlim(-0.1*1e3, 0.6*1e3)
axA.set_title(f'Migliore et. al., 2018\n {amp*1e12:.0f}pA current injected')
axA.set_ylim(-0.095*1e3, 0.05*1e3)
axA.set_xlabel('Time (ms)')
axA.set_ylabel('Voltage (mV)')
# axA.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axA.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axA.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Migliore LJP not corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

### Migliore 2018 LJP corrected ###
model_dir = 'MiglioreEtAl2018PLOSCompBiol2018_o'
output = np.load(f'{model_dir}/output_LJP.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axB.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C2')
line, = axB.plot([], alpha = 0)
axB.set_xlim(-0.1*1e3, 0.6*1e3)
axB.set_title(f'Migliore et. al., 2018 LJP corrected\n {amp*1e12:.0f}pA current injected')
axB.set_ylim(-0.095*1e3, 0.05*1e3)
axB.set_xlabel('Time (ms)')
axB.set_ylabel('Voltage (mV)')
# axB.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axB.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axB.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Migliore LJP corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

## Second Turi et al ###
model_dir = 'Turi_et_al_2018_o'
# result1 = subprocess.run(['nrnivmodl', 'mechanisms'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
# result1 = subprocess.run(['python3', 'Turi2019CA1pyrIclamp.py'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
output = np.load(f'{model_dir}/output.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axC.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C3')
line, = axB.plot([], alpha = 0)
axC.set_xlim(-0.1*1e3, 0.6*1e3)
axC.set_title(f'Turi et. al., 2019\n {amp*1e12:.0f}pA current injected')
axC.set_ylim(-0.095*1e3, 0.05*1e3)
axC.set_xlabel('Time (ms)')
axC.set_ylabel('Voltage (mV)')
# axC.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axC.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axC.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Turi LJP not corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

### Turi et al LJP corrected ####
output = np.load(f'{model_dir}/output_LJP.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axD.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C8')
line, = axB.plot([], alpha = 0)
axD.set_xlim(-0.1*1e3, 0.6*1e3)
axD.set_title(f'Turi et. al., 2019 LJP corrected\n {amp*1e12:.0f}pA current injected')
axD.set_ylim(-0.095*1e3, 0.05*1e3)
axD.set_xlabel('Time (ms)')
axD.set_ylabel('Voltage (mV)')
# axD.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axD.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axD.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Turi LJP corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

## Tomko et al 2021 #######################################
model_dir = 'TomkoEtAl2021_o'
# result1 = subprocess.run(['nrnivmodl', 'Mods'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
# result1 = subprocess.run(['python3', 'bAP_modified.py'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
output = np.load(f'{model_dir}/output.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axE.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C8')
line, = axB.plot([], alpha = 0)
axE.set_xlim(-0.1*1e3, 0.6*1e3)
axE.set_title(f'Tomko et. al., 2021\n {amp*1e12:.0f}pA current injected')
axE.set_ylim(-0.095*1e3, 0.05*1e3)
axE.set_xlabel('Time (ms)')
axE.set_ylabel('Voltage (mV)')
# axE.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axE.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axE.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Tomko LJP not corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

## Gouwens et al 2018 #######################################
model_dir = 'Gouwens_neuronal_model_488083972'
# result1 = subprocess.run(['nrnivmodl', 'modfiles'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
# result1 = subprocess.run(['python3', 'allenmodel_488083972_fullmorpho.py'], cwd=model_dir, capture_output=True, text=True)
# print(result1)
output = np.load(f'{model_dir}/output.npz')
delay, amp, t_vec, v_vec = output['delay'], output['amp'], output['t_vec'], output['v_vec']
# print(delay, amp, t_vec, v_vec)

features_ = {}
features_ = fts.ftscalc_helper(
    features_,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec*0,
    t_vec,
    v_vec,
    t_vec,
    v_vec,
    delay,
    delay+0.5,
)

axF.plot((t_vec-0.2)*1e3,v_vec*1e3, c='C2')
line, = axB.plot([], alpha = 0)
axF.set_xlim(-0.1*1e3, 0.6*1e3)
axF.set_title(f'Gouwens et. al., 2018\n {amp*1e12:.0f}pA current injected')
axF.set_ylim(-0.095*1e3, 0.05*1e3)
axF.set_xlabel('Time (ms)')
axF.set_ylabel('Voltage (mV)')
# axF.legend([line], [f'DBL = {DBL[1]:.1e}V'], loc='best', frameon=False)
axF.axhline(y=features_['E_rest_150']*1e3, color='black', linestyle='solid')
axF.axhline(y=features_['DBL_1.5e-10']*1e3, color='black', linestyle='--')

print("Gouwens LJP not manually corrected", features_['DBL_1.5e-10'], features_['DBLO_1.5e-10'])

## show plot ##
sns.despine(fig=fig)
# plt.show()
plt.savefig('Fig2.png', dpi=300)

