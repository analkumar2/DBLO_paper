import matplotlib.pyplot as plt
import numpy as np
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
import MOOSEModel_ as mm
import os
import subprocess
import scipy.stats as scs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set(style="ticks")
sns.set_context("paper")

fig = plt.figure(figsize=(7, 6), constrained_layout=False)
# fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axD_inset = inset_axes(axD, width="30%", height="30%", loc='center right')

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

#######################################

#########################################
image = plt.imread('AISsoma_Anzal.png')
position = axA.get_position()
Aposition = axA.transAxes.transform(axA.get_children()[0].get_position())
axA.imshow(image, aspect='auto')
left, bottom, width, height = position.x0, position.y0, position.width, position.height
zoom = 1
axA.set_position((left, bottom+height-zoom*height, width*image.shape[0]/image.shape[1]*zoom, height*zoom ))
axA.get_children()[0].set_position(axA.transAxes.inverted().transform(Aposition))
axA.axis('off')
#######################################

# Load models from the JSON file
basemodels_list = []
file_path = "activemodels.json"
with open(file_path, "r") as file:
    for line in file:
        basemodel = json.loads(line)
        basemodels_list.append(basemodel)

DBLO_150pA_list = []
AP1amp_150pA_list = []
RA_list = []
for model in basemodels_list:
    DBLO_150pA_list.append(model["Features"]["DBLO_1.5e-10"])
    AP1amp_150pA_list.append(model["Features"]["AP1_amp_1.5e-10"])
    RA_list.append(model["Parameters"]["Passive"]["sm_RA"])

axB.scatter(RA_list, np.array(DBLO_150pA_list)*1e3, c='C7', s=4)
axB.set_xlabel(r'RA ($\Omega$m)')
axB.set_ylabel('DBLO (mV)')
axB.set_xscale('log')

axC.scatter(RA_list, np.array(AP1amp_150pA_list)*1e3, c='C7', s=4)
axC.set_ylabel(r'Spike height (mV)')
axC.set_xlabel(r'RA ($\Omega$m)')
axC.set_xscale('log')


highRAmodelidx = np.argsort(RA_list)[-1]
lowRAmodelidx = np.argsort(RA_list)[0]
highRAmodel = basemodels_list[highRAmodelidx]
lowRAmodel = basemodels_list[lowRAmodelidx]

# pprint(highRAmodel)
t150, Itrace150, Vtrace150, Ca = mm.runModel(highRAmodel, 150e-12, refreshKin=True)
axD.plot(np.array(t150)*1e3, np.array(Vtrace150)*1e3, label='high RA', c='C2')
axD_inset.plot(np.array(t150)*1e3, np.array(Vtrace150)*1e3, label='high RA', c='C2')
axD_inset.set_xlim(500,1000)
axD_inset.set_ylim(np.max(Vtrace150)*1e3-5,np.max(Vtrace150)*1e3)
print('inset limits', (np.max(Vtrace150)*1e3-5,np.max(Vtrace150)*1e3), (500,1000))
t150, Itrace150, Vtrace150, Ca = mm.runModel(lowRAmodel, 150e-12, refreshKin=True)
axD.plot(np.array(t150)*1e3, np.array(Vtrace150)*1e3, label='low RA', c='C3')
# axD_inset.plot(np.array(t150)*1e3, np.array(Vtrace150)*1e3, label='low RA', c='C3')
axD.set_xlabel("Time (ms)")
axD.set_ylabel("Voltage (mV)")
axD.set_title("Representative models")
axD.legend(frameon=False)

axD_inset.tick_params(labelleft=False, labelbottom=False, bottom = False, left=False)

axB.scatter(RA_list[highRAmodelidx], np.array(DBLO_150pA_list)[highRAmodelidx]*1e3, c='C2', s=15)
axB.scatter(RA_list[lowRAmodelidx], np.array(DBLO_150pA_list)[lowRAmodelidx]*1e3, c='C3', s=15)
axC.scatter(RA_list[highRAmodelidx], np.array(AP1amp_150pA_list)[highRAmodelidx]*1e3, c='C2', s=15)
axC.scatter(RA_list[lowRAmodelidx], np.array(AP1amp_150pA_list)[lowRAmodelidx]*1e3, c='C3', s=15)

######################
sns.despine(fig=fig)
plt.tight_layout()
plt.savefig('Fig4.png', dpi=300)
# plt.savefig('Fig4.svg', dpi=300)
plt.show()


############# Stats #####################
m, b, r, pvalue, _ = scs.linregress(RA_list, np.array(DBLO_150pA_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')
m, b, r, pvalue, _ = scs.linregress(RA_list, np.array(AP1amp_150pA_list)*1e3)
print(f'{r:1.2f}', f'{pvalue:1.2e}')