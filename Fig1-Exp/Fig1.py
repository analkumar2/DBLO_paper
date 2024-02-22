import sys 
sys.path.insert(1, "../helperScripts")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import features as fts
import expcells
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
import os
from pprint import pprint
import scipy.stats as scs


sns.set(style="ticks")
sns.set_context("paper")

# Create two subplots side by side
fig = plt.figure(figsize=(8, 6), constrained_layout=False)
# fig = plt.figure(figsize=(4.69, 3.135), constrained_layout=True)
gs = GridSpec(2, 6, figure=fig)
axA = fig.add_subplot(gs[0, 0:3])
axB = fig.add_subplot(gs[0, 3:])
axC = fig.add_subplot(gs[1, 0:2])
axD = fig.add_subplot(gs[1, 2:4])
axE = fig.add_subplot(gs[1, 4:])

# add a, b, c text to each subplot axis
fig.transFigure.inverted().transform([0.5,0.5])
for i, ax in enumerate([axA, axB, axC, axD, axE]):
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

def optimal_position_text(ax, text):
    line, = axD.plot([], [], label=text, color='black')
    text_legend = ax.legend(handlelength=0, handles=[line], loc='best', frameon=False)
    ax.add_artist(text_legend)

cell2 = expcells.expcell('2023_01_04_cell_2', '../expdata/Chirp/2023_01_04_cell_2') #Representative cell
T_150pA, Vm_150pA = expcells.expdata(cell2.preFIfile, Index=10) #get the curent response at 150pA. Index 0 = -100pA, Index 1 = -75pA, and so on
T_300pA, Vm_300pA = expcells.expdata(cell2.preFIfile, Index=16) #get the curent response at 300pA. Index 0 = -100pA, Index 1 = -75pA, and so on
stim_start = 0.3469
stim_end = stim_start+0.5
LJP = 15e-3
Features = fts.expfeatures(cellpath=cell2.preFIfile, stim_start=stim_start, stim_end=stim_end, LJP=15e-3)
print(Features['DBLO_1.5e-10'], Features['DBLO_3e-10'], Features['DBL_1.5e-10'], Features['DBL_3e-10'])

# Plot something on the first subplot
axA.plot((T_150pA-stim_start)*1e3, (Vm_150pA-LJP)*1e3, c='C0')
axA.set_xlabel('Time (ms)')
axA.set_ylabel('Voltage (mV)')
axA.set_title('150pA')
axA.set_xlim(-0.1*1e3, 0.6*1e3)
axA.set_ylim(-90e-3*1e3, 40e-3*1e3)
axA.axhline(y=Features['E_rest_150']*1e3, color='black', linestyle='dotted', xmin=0, xmax=1)
axA.axhline(y=Features['DBL_1.5e-10']*1e3, color='black', linestyle='--')


# Plot something on the second subplot
axB.plot((T_300pA-stim_start)*1e3, (Vm_300pA-LJP)*1e3, c='C0')
axB.set_xlabel('Time (ms)')
axB.set_title('300pA')
axB.set_xlim(-0.1*1e3, 0.6*1e3)
axB.set_ylim(-90e-3*1e3, 40e-3*1e3)
axB.tick_params(left=False, labelleft=False)
axB.axhline(y=Features['E_rest_300']*1e3, color='black', linestyle='dotted')
axB.axhline(y=Features['DBL_3e-10']*1e3, color='black', linestyle='--')

# Add text between the two subplots
fig.text(0.535, 0.625, 'DBLO', ha='center', fontsize=14)
# Add a vertical arrow below the text
arrow_style = dict(arrowstyle='<->', color='black')
axA.annotate('',
             xy=(0.59*1e3,Features['E_rest_150']*1e3),  # starting point of the arrow
             xytext=(0.59*1e3, Features['DBL_1.5e-10']*1e3),  # end point of the arrow
             arrowprops=arrow_style,
             xycoords='data')
axB.annotate('',
             xy=(-0.09*1e3,Features['E_rest_300']*1e3),  # starting point of the arrow
             xytext=(-0.09*1e3, Features['DBL_3e-10']*1e3),  # end point of the arrow
             arrowprops=arrow_style,
             xycoords='data')


#### DBL, Rin and Cin #####

if not os.path.exists('expfeatures.npz'):
    validcells = [
            "2023_01_04_cell_1",
            "2023_01_04_cell_2",
            "2023_01_04_cell_3",
            "2023_01_04_cell_4",
            "2023_01_04_cell_5",
            "2023_01_04_cell_6",
            # "2023_01_20_cell_1", #Invalid exp
            "2023_01_20_cell_2",
            "2023_01_20_cell_3",
            "2023_01_20_cell_4",
            "2023_02_13_cell_1",
            "2023_02_13_cell_2",
            "2023_02_13_cell_3",
            "2023_02_13_cell_4",
        ]
    expcell_list = []
    Features_list = []
    DBL_list_list = []
    Rin_list = []
    Cin_list = []
    for cells in tqdm(validcells):
        cell = expcells.expcell(cells, f'../expdata/Chirp/{cells}')
        expcell_list.append(cell)
        Features = fts.expfeatures(cellpath=cell.preFIfile, stim_start=stim_start, stim_end=stim_end, LJP=15e-3)
        Features_list.append(Features)
        Rin_list.append(Features['Input resistance'])
        Cin_list.append(Features['Cell capacitance'])
        DBL_list = []
        for i in range(21):
            t, V = expcells.expdata(cell.preFIfile, i, LJP=15e-3)
            features_ = {}
            features_ = fts.ftscalc_helper(
                features_,
                t,
                V*0,
                t,
                V*0,
                t,
                V*0,
                t,
                V,
                t,
                V,
                stim_start,
                stim_end,
            )
            DBL = [features_["DBLO_1.5e-10"], features_["DBL_1.5e-10"]] #We have used placeholder V traces while calculating the features
            DBL_list.append(DBL)
        DBL_list_list.append(DBL_list)

    np.savez('expfeatures.npz', Features_list=Features_list, DBL_list_list=DBL_list_list, Rin_list=Rin_list, Cin_list=Cin_list)
else:
    _ = np.load('expfeatures.npz', allow_pickle=True)
    Features_list = _['Features_list']
    DBL_list_list = _['DBL_list_list']
    Rin_list = _['Rin_list']
    Cin_list = _['Cin_list']

# Features_df = pd.DataFrame(Features_list)

## first DBL vs current##
DBL_list_list = np.array(DBL_list_list, dtype=float)
# Features_df_melted = Features_df.melt(value_vars=['offset_1.5e-10', 'offset_3e-10'], var_name='Injected current (A)', value_name='DBL (V)')
# Features_df_melted['Injected current (A)'] = Features_df_melted['Injected current (A)'].replace({'offset_1.5e-10': '150e-12', 'offset_3e-10': '300e-12'})
# # sns.boxplot(x='Injected current (A)', y='DBL (V)', data=Features_df_melted, ax=axC, showfliers=False)
# sns.stripplot(x='Injected current (A)', y='DBL (V)', data=Features_df_melted, ax=axC, color=".25")
# Plot each element of DBL_list separately
Inj = np.arange(-100e-12, 410e-12, 25e-12)
# Compute mean and std deviation
meann = np.nanmean(DBL_list_list[:,:,0], axis=0)
# stdn = np.nanstd(DBL_list_list[:,:,1], axis=0)
# sten  =  stdn/np.sqrt(np.sum(~np.isnan(DBL_list_list[:,:,1])))
sten = scs.sem(DBL_list_list[:,:,0], axis=0)
axC.scatter([Inj*1e12]*len(DBL_list_list[:,:,0]), DBL_list_list[:,:,0]*1e3, c='C7', alpha=0.5)
# Plot the mean line with error bars representing 1 standard deviation
axC.errorbar(Inj*1e12, meann*1e3, yerr=sten*1e3, fmt='-o', c='black')
axC.set_xlabel(r'$I_{inj}$ (pA)')
axC.set_ylabel('DBLO (mV)')
# axC.set_xticks(Inj)

## next Rin ##
Rin_list = np.array(Rin_list)
# Features_df_melted = Features_df.melt(value_vars=['Input resistance'], var_name='CA1 pyr', value_name='Input resistance (ohms)')
# # Features_df_melted['CA1 pyr'] = Features_df_melted['CA1 pyr'].replace({'offset_1.5e-10': '150e-12', 'offset_3e-10': '300e-12'})
# # sns.boxplot(x='CA1 pyr', y='Input resistance (ohms)', data=Features_df_melted, ax=axD, showfliers=False)
# sns.stripplot(x='CA1 pyr', y='Input resistance (ohms)', data=Features_df_melted, ax=axD, color=".25")
# axD.set(xticklabels=[])
axD.scatter(Rin_list*1e-6, DBL_list_list[:,10,0]*1e3, color='C7')
m, b, r, pvalue, _ = scs.linregress(Rin_list, DBL_list_list[:,10,0])
r, pvalue = scs.spearmanr(Rin_list, DBL_list_list[:,10,0])
axD.plot(Rin_list*1e-6, m*Rin_list*1e3 + b*1e3, color='black')
axD.set_xlabel(r'$R_{in}$ (M$\Omega$)')
axD.set_ylabel('DBLO (mV)')
axD.set_ylim([0,max(DBL_list_list[:,10,0]*1e3)+5])
optimal_position_text(axD, ' '+'_'*20 + '\n' + ' '*14 + 'ns' + '\n' + f"Spearman's r = {r:.2f}" )
# axD.tick_params(left=False, labelleft=False)
# axD.set_title('Input resistance vs DBL at 150pA')
print(f'Rin vs DBLO: {m=}, {b=}, {r=}, {pvalue=}')

# ## next Cin ##
Cin_list = np.array(Cin_list)
# Features_df_melted = Features_df.melt(value_vars=['Cell capacitance'], var_name='CA1 pyr', value_name='Cell capacitance (F)')
# # Features_df_melted['CA1 pyr'] = Features_df_melted['CA1 pyr'].replace({'offset_1.5e-10': '150e-12', 'offset_3e-10': '300e-12'})
# # sns.boxplot(x='CA1 pyr', y='Cell capacitance (F)', data=Features_df_melted, ax=axE, showfliers=False)
# sns.stripplot(x='CA1 pyr', y='Cell capacitance (F)', data=Features_df_melted, ax=axE, color=".25")
# axE.set(xticklabels=[])
axE.scatter(Cin_list*1e12, DBL_list_list[:,10,0]*1e3, color='C7')
m, b, r, pvalue, _ = scs.linregress(Cin_list, DBL_list_list[:,10,0])
r, pvalue = scs.spearmanr(Cin_list, DBL_list_list[:,10,0])
axE.plot(Cin_list*1e12, m*Cin_list*1e3 + b*1e3, color='black')
axE.set_xlabel(r'$C_{in}$ (pF)')
axE.set_ylabel('DBLO (mV)')
axE.set_ylim([0,max(DBL_list_list[:,10,0]*1e3)+5])
optimal_position_text(axE, ' '+'_'*20 + '\n' + ' '*14 + r'$\star$' + '\n' + f"Spearman's r = {r:.2f}" )
# axE.tick_params(left=False, labelleft=False)
# axE.set_title('Cell capacitance vs DBL at 150pA')
axE.legend(frameon=False)
print(f'Cin vs DBLO: {m=}, {b=}, {r=}, {pvalue=}')


# Show the plots
sns.despine(fig=fig)
axB.spines['left'].set_visible(False)
# axD.spines['left'].set_visible(False)
# axE.spines['left'].set_visible(False)
plt.savefig('Fig1.png', dpi=300)
plt.show()
