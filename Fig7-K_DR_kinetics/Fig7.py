import sys

sys.path.insert(1, "../helperScripts")
sys.path.insert(1, "../Kinetics")

import matplotlib.pyplot as plt
import numpy as np
import features as fts
import expcells
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from pprint import pprint
import os
import pickle
import json
from goMultiprocessing import Multiprocessthis_appendsave
from copy import deepcopy
from pprint import pprint
import MOOSEModel as mm
import os
import subprocess
import scipy.stats as scs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(8, 10), constrained_layout=False)
# fig = plt.figure(constrained_layout=True, dpi=300)
gs = GridSpec(6, 6, figure=fig, wspace=1, hspace=1, height_ratios=[3,2,2,2,2,2])
axA = fig.add_subplot(gs[0, :3])
axB = fig.add_subplot(gs[0, 3:])
axC = fig.add_subplot(gs[1, :2])
axD = fig.add_subplot(gs[1, 2:4])
axE = fig.add_subplot(gs[1, 4:])
axF = fig.add_subplot(gs[2, :2])
axG = fig.add_subplot(gs[2, 2:4])
axH = fig.add_subplot(gs[2, 4:])
axI = fig.add_subplot(gs[3, :2])
axJ = fig.add_subplot(gs[3, 2:4])
axK = fig.add_subplot(gs[3, 4:])
axL = fig.add_subplot(gs[4, :2])
axM = fig.add_subplot(gs[4, 2:4])
axN = fig.add_subplot(gs[4, 4:])
gs_inner = gs[5, :].subgridspec(1, 3, wspace=0.5, width_ratios=[3,2,2])
axO = fig.add_subplot(gs_inner[0, 0])
axP = fig.add_subplot(gs_inner[0, 1:])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB, axC, axD, axE, axF, axG, axH, axI, axJ, axK, axL, axM, axN, axO, axP]):
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
basemodels_list = []
file_path = "activemodels.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        basemodels_list.append(basemodel)

df_expsummaryactiveF = pd.read_pickle("../helperScripts/expsummaryactiveF.pkl")
########################################################################
image = plt.imread('single compt.png')
position = axA.get_position()
Aposition = axA.transAxes.transform(axA.get_children()[0].get_position())
Aposition[1] += 30
axA.imshow(image, aspect='equal')
left, bottom, width, height = position.x0, position.y0, position.width, position.height
zoom = 2.5
axA.set_position((left-0.07, bottom+height-zoom*height+0.06, width*image.shape[0]/image.shape[1]*zoom, height*zoom ))
axA.get_children()[0].set_position(axA.transAxes.inverted().transform(Aposition))
axA.axis('off')
########################################################################
### getting the kinetics ###
import moose
import K_DR_Chan_Custom3
moose.Neutral('library')
K_DR = K_DR_Chan_Custom3.K_DR_Chan("K_DR_Chan")
xgate = moose.element( K_DR.path + '/gateX' )
v = np.linspace(xgate.min, xgate.max, xgate.divs+1)
Inf = xgate.tableA/xgate.tableB
Tau = 1/xgate.tableB

axB.plot(v*1e3, Tau*1e3, color='black')
firstargmin = np.argmin(Tau[:100])
lastargmin = len(Tau) - 100 + np.argmin(Tau[-100:])
argmax = np.argmax(Tau)
axB.plot([v[firstargmin]*1e3 - 10, v[firstargmin]*1e3 + 50], np.repeat(Tau[firstargmin]*1e3,2), color='black', linestyle='--', alpha=0.5)
axB.text(x=v[firstargmin]*1e3, y=Tau[firstargmin]*1e3+3, s='D')
axB.plot([v[lastargmin]*1e3 - 50, v[lastargmin]*1e3 + 10], np.repeat(Tau[lastargmin]*1e3,2), color='black', linestyle='--', alpha=0.5)
axB.text(x=v[lastargmin]*1e3, y=Tau[lastargmin]*1e3+3, s='C')
axB.plot([v[argmax]*1e3 - 10, v[argmax]*1e3 + 40], np.repeat(Tau[argmax]*1e3,2), color='black', linestyle='--', alpha=0.5)
axB.text(x=v[argmax]*1e3 + 40, y=Tau[argmax]*1e3, s='F')
axB.plot(np.repeat(v[argmax]*1e3,2), [0, Tau[argmax]*1e3 + 6], color='black', linestyle='--', alpha=0.5)
axB.text(x=v[argmax]*1e3, y=Tau[argmax]*1e3 + 6, s='A')
Xlarg = np.argmax(np.diff(Tau[:int(len(Tau)/2)], 2))
Xrarg = int(len(Tau)/2 + np.argmax(np.diff(Tau[int(len(Tau)/2):], 2)))
ml = np.diff(Tau)[Xlarg]/(v[1]-v[0])
mr = np.diff(Tau)[Xrarg]/(v[1]-v[0])
axB.plot([v[Xlarg]*1e3 - 20, v[Xlarg]*1e3 + 20], [ml*-20 + Tau[Xlarg]*1e3, ml*20 + Tau[Xlarg]*1e3], color='black', linestyle='--', alpha=0.5)
axB.text(x=v[Xlarg]*1e3+5, y=Tau[Xlarg]*1e3-5, s='E')
axB.plot([v[Xrarg]*1e3 - 15, v[Xrarg]*1e3 + 15], [mr*-15 + Tau[Xrarg]*1e3, mr*15 + Tau[Xrarg]*1e3], color='black', linestyle='--', alpha=0.5)
axB.text(x=v[Xrarg]*1e3-8, y=Tau[Xrarg]*1e3-5, s='B')
# X = v[[int(len(Tau)/4)-20, int(len(Tau)/4)+20]]
# axB.plot(X*1e3, np.gradient(Tau*1e3)[int(len(Tau)/4)]*X*1e3 + 15)
# axB.plot(v*1e3, Tau2*1e3)
axB.set_xlabel('Voltage (mV)')
axB.set_ylabel(r'$\tau$ (ms)')
axB.set_ylim(0,40)

##################################################################################
def ChanGate(v,vhalf_inf, slope_inf, A, B, C, D, E, F):
    # alge model
    Inf = 1/(1+np.exp((v-vhalf_inf)/-slope_inf))
    yl = (v-A)/-B
    yr = (v-A)/E
    Tau = (C + (1 + yl/(np.sqrt(1+yl**2)))/2) * (D + (1 + yr/(np.sqrt(1+yr**2)))/2) * F
    # Tau[Tau<0.00002] = 0.00002
    return [Inf,Tau]

Em_list = []
Rm_list = []
Cm_list = []
h_Chan_Gbar_list = []
K_DR_Chan_Gbar_list = []
Na_T_Chan_Gbar_list = []
A_list = []
B_list = []
C_list = []
D_list = []
E_list = []
F_list = []
DBLO_list = []
Taum65_list = []
maxtau_list = []
for i in range(len(basemodels_list)):
    sm_area = np.pi*basemodels_list[i]["Parameters"]["Morphology"]["sm_len"]*basemodels_list[i]["Parameters"]["Morphology"]["sm_diam"]
    Em_list.append(basemodels_list[i]["Parameters"]["Passive"]["Em"])
    Rm_list.append(basemodels_list[i]["Parameters"]["Passive"]["sm_RM"]/sm_area)
    Cm_list.append(basemodels_list[i]["Parameters"]["Passive"]["sm_CM"]*sm_area)
    h_Chan_Gbar_list.append(basemodels_list[i]["Parameters"]["Channels"]["h_Chan"]["Gbar"])
    K_DR_Chan_Gbar_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["Gbar"])
    Na_T_Chan_Gbar_list.append(basemodels_list[i]["Parameters"]["Channels"]["Na_T_Chan"]["Gbar"])
    A_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_A"])
    B_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_B"])
    C_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_C"])
    D_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_D"])
    E_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_E"])
    F_list.append(basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"]["n_F"])
    DBLO_list.append(basemodels_list[i]["Features"]["DBLO_1.5e-10"])
    Inf, Tau = ChanGate(-65e-3, *basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"].values())
    Taum65_list.append(Tau)
    Inf, Tau = ChanGate(np.linspace(-0.1, 0.1, 1000), *basemodels_list[i]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"].values())
    maxtau_list.append(max(Tau))

#####################
axC.scatter(np.array(Em_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axC.scatter(np.array(Em_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axC.scatter(np.array(Em_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axC.set_xlabel('Em (mV)')
axC.set_ylabel('DBLO (mV)')

axD.scatter(np.array(Rm_list)*1e-6, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axD.scatter(np.array(Rm_list)[np.argsort(DBLO_list)[0]]*1e-6, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axD.scatter(np.array(Rm_list)[np.argsort(DBLO_list)[-1]]*1e-6, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axD.set_xlabel(r'Rm (M$\Omega$)')
# axD.set_ylabel('DBLO (mV)')

axE.scatter(np.array(Cm_list)*1e12, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axE.scatter(np.array(Cm_list)[np.argsort(DBLO_list)[0]]*1e12, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axE.scatter(np.array(Cm_list)[np.argsort(DBLO_list)[-1]]*1e12, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axE.set_xlabel('Cm (pF)')
# axE.set_ylabel('DBLO (mV)')

##############################
axF.scatter(np.array(h_Chan_Gbar_list)*1e6, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axF.scatter(np.array(h_Chan_Gbar_list)[np.argsort(DBLO_list)[0]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axF.scatter(np.array(h_Chan_Gbar_list)[np.argsort(DBLO_list)[-1]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axF.set_xlabel(r'h_Chan_$\overline{G}$ ($\mu$S)')
axF.set_ylabel('DBLO (mV)')

axG.scatter(np.array(K_DR_Chan_Gbar_list)*1e6, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axG.scatter(np.array(K_DR_Chan_Gbar_list)[np.argsort(DBLO_list)[0]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axG.scatter(np.array(K_DR_Chan_Gbar_list)[np.argsort(DBLO_list)[-1]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axG.set_xlabel(r'K_DR_Chan_$\overline{G}$ ($\mu$S)')
# axG.set_ylabel('DBLO (mV)')
axG.set_xscale('log')

axH.scatter(np.array(Na_T_Chan_Gbar_list)*1e6, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axH.scatter(np.array(Na_T_Chan_Gbar_list)[np.argsort(DBLO_list)[0]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axH.scatter(np.array(Na_T_Chan_Gbar_list)[np.argsort(DBLO_list)[-1]]*1e6, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axH.set_xlabel(r'Na_T_Chan_$\overline{G}$ ($\mu$S)')
# axH.set_ylabel('DBLO (mV)')
axH.set_xscale('log')

##############################
axI.scatter(np.array(A_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axI.scatter(np.array(A_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axI.scatter(np.array(A_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axI.set_xlabel('$K^+$ n_A')
axI.set_ylabel('DBLO (mV)')

axJ.scatter(np.array(B_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axJ.scatter(np.array(B_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axJ.scatter(np.array(B_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axJ.set_xlabel('$K^+$ n_B')
# axJ.set_ylabel('DBLO (mV)')

axK.scatter(np.array(C_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axK.scatter(np.array(C_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axK.scatter(np.array(C_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axK.set_xlabel('$K^+$ n_C')
# axK.set_ylabel('DBLO (mV)')

###############################
axL.scatter(np.array(D_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axL.scatter(np.array(D_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axL.scatter(np.array(D_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axL.set_xlabel('$K^+$ n_D')
axL.set_ylabel('DBLO (mV)')

axM.scatter(np.array(E_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axM.scatter(np.array(E_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axM.scatter(np.array(E_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axM.set_xlabel('$K^+$ n_E')
# axM.set_ylabel('DBLO (mV)')

axN.scatter(np.array(F_list), np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axN.scatter(np.array(F_list)[np.argsort(DBLO_list)[0]], np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axN.scatter(np.array(F_list)[np.argsort(DBLO_list)[-1]], np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axN.set_xlabel('$K^+$ n_F')
# axN.set_ylabel('DBLO (mV)')

################################
axO.scatter(np.array(Taum65_list)*1e3, np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
axO.scatter(np.array(Taum65_list)[np.argsort(DBLO_list)[0]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
axO.scatter(np.array(Taum65_list)[np.argsort(DBLO_list)[-1]]*1e3, np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
axO.set_xlabel(r'$\tau$ at -65mV (ms)')
axO.set_ylabel('DBLO (mV)')

# axP.scatter(np.array(Taum65_list)*1e3/(np.array(maxtau_list)*1e3), np.array(DBLO_list)*1e3, s=4, c='C7', alpha=0.4)
# axP.scatter(np.array(Taum65_list)[np.argsort(DBLO_list)[0]]*1e3/(np.array(maxtau_list)[np.argsort(DBLO_list)[0]]*1e3), np.array(DBLO_list)[np.argsort(DBLO_list)[0]]*1e3, s=15, c='C3')
# axP.scatter(np.array(Taum65_list)[np.argsort(DBLO_list)[-1]]*1e3/(np.array(maxtau_list)[np.argsort(DBLO_list)[-1]]*1e3), np.array(DBLO_list)[np.argsort(DBLO_list)[-1]]*1e3, s=15, c='C2')
# axP.set_xlabel('ratio of tau at -65 and max tau')
# axP.set_ylabel('DBLO (mV)')

#### Representative extreme models #####
stim_start = 0.5
tvec, _, Vmvec, _ = mm.runModel(basemodels_list[np.argsort(DBLO_list)[0]], 150e-12, refreshKin=True)
axP.plot((tvec-stim_start)*1e3, Vmvec*1e3, c='C3', label=f'{basemodels_list[np.argsort(DBLO_list)[0]]["Features"]["DBLO_1.5e-10"]*1e3:1.1f} mV DBLO')
tvec, _, Vmvec, _ = mm.runModel(basemodels_list[np.argsort(DBLO_list)[-1]], 150e-12, refreshKin=True)
axP.plot((tvec-stim_start)*1e3, Vmvec*1e3, c='C2', label=f'{basemodels_list[np.argsort(DBLO_list)[-1]]["Features"]["DBLO_1.5e-10"]*1e3:1.1f} mV DBLO')

# axP.legend(frameon=False, loc='center right', bbox_to_anchor=(1.1,1),handlelength=1)
axP.legend(frameon=False, ncols=2,handlelength=1, loc='upper center', bbox_to_anchor=(0.5,1.3))
axP.set_xlabel('Time (ms)')
axP.set_ylabel('Voltage (mV)')
# axP.legend(frameon=False)
axP.set_xlim(-0.1*1e3, 0.6*1e3)
axP.set_ylim(-100e-3*1e3, 60e-3*1e3)

Inf, Tau = ChanGate(v, *basemodels_list[np.argsort(DBLO_list)[0]]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"].values())
axB.plot(v*1e3, Tau*1e3, c='C3', alpha=0.5, linewidth=2)
Inf, Tau = ChanGate(v, *basemodels_list[np.argsort(DBLO_list)[-1]]["Parameters"]["Channels"]["K_DR_Chan"]["KineticVars"].values())
axB.plot(v*1e3, Tau*1e3, c='C2', alpha=0.5, linewidth=2)
######################
sns.despine(fig=fig)
plt.savefig('Fig7.png', dpi=300)
plt.show()


###### Some stats ##############
m, b, r, pvalue, _ = scs.linregress(np.array(K_DR_Chan_Gbar_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(Na_T_Chan_Gbar_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(A_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(B_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(C_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(D_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(E_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(F_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(np.array(Taum65_list), np.array(DBLO_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')