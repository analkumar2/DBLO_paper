## Here we just csv and pkl of exp features. passive, passive+active and chirp ones

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4

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
from time import time

### get exp cell features ###
LJP = 15e-3
samplingrate = 20000
stim_start_exp = 0.3469
stim_end_exp = stim_start_exp + 0.5

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


########## Chirp features #####################################
def getexpchirp():
    if os.path.exists("expchirp.pkl"):
        df = pd.read_pickle("expchirp.pkl")
        return df
    else:
        freq_list = []
        impedance_list = []
        phase_list = []

        for cells in tqdm(validcells):
            cell = expcells.expcell(cells, f"../expdata/Chirp/{cells}")
            cell.ampphase_freq(True)

            freq_l = []
            imp_l = []
            ph_l = []
            for i in range(len(cell.freq)):
                freq_l.append(cell.freq[i])
                imp_l.append(cell.impedance[i])
                ph_l.append(cell.phase[i])
            freq_list.append(np.mean(freq_l, axis=0))
            impedance_list.append(np.mean(imp_l, axis=0))
            phase_list.append(np.mean(ph_l, axis=0))

        df = pd.DataFrame(
            {
                "cellID": validcells,
                "freq": freq_list,
                "impedance": impedance_list,
                "phase": phase_list,
            }
        )
        df.to_pickle("expchirp.pkl")
        df.to_csv("expchirp.csv", index=True)
        return df

df_expchirp = getexpchirp()


######## Active features + passive #################################
def getexpactiveF():
    if os.path.exists("expactiveF.pkl"):
        df = pd.read_pickle("expactiveF.pkl")
        return df
    else:
        df = pd.DataFrame()
        for cells in tqdm(validcells):
            cell = expcells.expcell(cells, f"../expdata/Chirp/{cells}")
            print(cells)
            cellF = fts.expfeatures(
                cell.preFIfile, stim_start_exp, stim_end_exp, LJP=LJP
            )
            cellF["CellID"] = cells
            # pprint(cellF)
            df = pd.concat(
                [df, pd.DataFrame(cellF, index=[0])],
                ignore_index=True,
            )

        df.to_pickle("expactiveF.pkl")
        df.to_csv("expactiveF.csv", index=True)

        df = df.drop(columns=["CellID", 'Cell name'])
        print(df.columns)
        summary_df = pd.DataFrame(
            {
                "Median": df.median(),
                "10th quantile": df.quantile(0.1),
                "90th quantile": df.quantile(0.9),
            }
        )
        summary_df.to_pickle("expsummaryactiveF.pkl")
        summary_df.to_csv("expsummaryactiveF.csv", index=True)

        return df


df_expactiveF = getexpactiveF()


############ Just passives #################################
def getexppasF():
    if os.path.exists("exppasF.pkl"):
        df = pd.read_pickle("exppasF.pkl")
        return df
    else:
        df = pd.DataFrame()
        for cells in tqdm(validcells):
            print(cells)
            cell = expcells.expcell(cells, f"../expdata/Chirp/{cells}")
            tm25, Vtracem25 = expcells.expdata(cell.preFIfile, 3, LJP=LJP)
            E_rest, Rin, Cin, tau, sag_rat = fts.calcRinCin(
                    tm25,
                    Vtracem25,
                    stim_start_exp,
                    stim_end_exp,
                )
            features = {}
            features["cellID"] = cells
            features["E_rest"] = E_rest
            features["Rin"] = Rin
            features["Cin"] = Cin
            features["tau"] = tau
            features["sag_rat"] = sag_rat
            df = pd.concat([df, pd.DataFrame(features, index=[0])], ignore_index=True)

        df.to_pickle("exppasF.pkl")
        df.to_csv("exppasF.csv", index=True)
        return df

df_exppasF = getexppasF()