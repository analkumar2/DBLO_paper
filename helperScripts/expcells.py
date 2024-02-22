import numpy as np
import matplotlib.pyplot as plt
import features as fts
from glob import glob
from neo.io import AxonIO
from goMultiprocessing import Multiprocessthis_appendsave
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import os


validcells = [
    "2023_01_04_cell_1",
    "2023_01_04_cell_2",
    "2023_01_04_cell_3",
    "2023_01_04_cell_4",
    "2023_01_04_cell_5", ## Anomalous firing at 300pA
    "2023_01_04_cell_6",
    # "2023_01_20_cell_1", ## Invalid cell
    "2023_01_20_cell_2",
    "2023_01_20_cell_3",
    "2023_01_20_cell_4",
    "2023_02_13_cell_1",
    "2023_02_13_cell_2",
    "2023_02_13_cell_3",
    "2023_02_13_cell_4",
]


class expcell:
    def __init__(self, ID, IDpath):
        self.ID = ID
        self.IDpath = IDpath
        self.listabf()
        self.findpreFIabf()

    def listabf(self):
        abffilelist = glob(f"{self.IDpath}/**/*.abf", recursive=True)
        # print(self.ID, abffilelist)
        self.abffilelist = abffilelist

    def findpreFIabf(self):
        FIfiles = []
        chirpfiles = []
        for file in self.abffilelist:
            reader = AxonIO(filename=file)
            # print(file, len(reader.read_block(signal_group_mode='split-all').segments))
            if (
                len(reader.read_block(signal_group_mode="split-all").segments) == 21
                or len(reader.read_block(signal_group_mode="split-all").segments) == 17
            ):
                FIfiles.append(file)
            if len(reader.read_block(signal_group_mode="split-all").segments) == 5:
                chirpfiles.append(file)

        FIfiles = np.sort(FIfiles)
        chirpfiles = np.sort(chirpfiles)
        self.preFIfile = FIfiles[0]
        self.postFIfile = FIfiles[-1]
        self.chirpfiles = chirpfiles

    def chirpresponse(self, normalize=True):
        self.chirpT = []
        self.chirpV = []
        self.chirpdt = []
        self.sProtocolPath = []
        self.Injcurr = []
        for file in self.chirpfiles:
            reader = AxonIO(filename=file)
            Injcurr = (
                int(
                    str(reader._axon_info["sProtocolPath"])
                    .split(".")[0]
                    .split("_")[-1][:2]
                )
                * 1e-12
            )
            for seg in reader.read_block(signal_group_mode="split-all").segments:
                self.Injcurr.append(Injcurr)
                Tdur = np.array(seg.t_stop - seg.t_start)
                V = np.array(np.ravel(seg.analogsignals[0])) * 1e-3
                T = np.linspace(0, Tdur + 0, len(V))
                self.chirpdt.append(1 / reader.get_signal_sampling_rate())
                self.sProtocolPath.append(reader._axon_info["sProtocolPath"])
                if normalize:
                    self.chirpT.append(T)
                    self.chirpV.append((V - np.median(V)) / Injcurr)
                else:
                    self.chirpT.append(T)
                    self.chirpV.append(V)

    def ampphase_freq(self, windowed=False):
        self.chirpresponse(normalize=True)
        self.freq = []
        self.impedance = []
        self.phase = []
        stimamp = 1
        for i in range(len(self.chirpT)):
            t = np.arange(0, 13, self.chirpdt[i])
            chirp = np.zeros(int(300e-3 / self.chirpdt[i]))
            chirp = np.concatenate([chirp, stimamp * np.sin(2 * np.pi * t * t**2)])
            # chirp = np.concatenate([chirp, stimamp*np.sin(2*np.pi*t*np.e**t)])
            chirp = np.concatenate(
                [chirp, np.zeros(len(self.chirpV[i]) - len(chirp))]
            )  # Its in pA
            t = np.arange(0, len(chirp) * self.chirpdt[i], self.chirpdt[i])
            # plt.plot(t, chirp)
            # plt.show()
            # plt.plot(self.chirpT[i], self.chirpV[i])
            # plt.show()
            freq_exp, Impedance_exp, phase_exp = fts.calcImpedance(
                chirp, self.chirpV[i], self.chirpdt[i]
            )
            # plt.plot(freq_exp[(freq_exp>0.5) & (freq_exp<=500)], Impedance_exp[(freq_exp>0.5) & (freq_exp<=500)])
            # plt.plot(freq_exp, Impedance_exp)
            # plt.show()
            self.freq.append(freq_exp[(freq_exp > 0.5) & (freq_exp <= 500)])
            self.impedance.append(Impedance_exp[(freq_exp > 0.5) & (freq_exp <= 500)])
            self.phase.append(phase_exp[(freq_exp > 0.5) & (freq_exp <= 500)])

def expdata(Address, Index=0, mode='Iclamp', LJP=0):
    '''
    mode: Either Iclamp or Vclamp.
    '''
    if os.path.splitext(Address)[1] == '.abf':
        reader = AxonIO(filename=Address)
        Samprate = reader.get_signal_sampling_rate()
        seg = reader.read_block(signal_group_mode='split-all').segments[Index]
        Tdur = np.array(seg.t_stop - seg.t_start)
        if mode=='Iclamp':
            V = np.array(np.ravel(seg.analogsignals[0]))*1e-3 - LJP
            return [np.linspace(0,Tdur+0,len(V)), V]
        elif mode=='Vclamp':
            I = np.array(np.ravel(seg.analogsignals[0]))*1e-12
            return [np.linspace(0,Tdur+0,len(I)), I]
    elif os.path.splitext(Address)[1] == '.dat':
        a = np.loadtxt(Address)
        b = np.transpose(a)
        T = b[0]
        if mode=='Iclamp':
            V = b[1:][Index]*1e-3 - LJP
            return [T,V]
        elif mode=='Vclamp':
            I = b[1:][Index]*1e-12
            return [T,I]

def plotexp(Address, Index=0, mode='Iclamp', Title='Current clamp at 150pA', LJP=0):
    '''
    mode: Either Iclamp or Vclamp.
    '''
    if mode=='Iclamp':
        T, V = expdata(Address, Index, mode, LJP=LJP)
        plt.plot(T,V, label=Address)
        plt.legend()
        plt.title(Title)
        plt.xlabel('Time (s)')
        plt.ylabel('Membrane potential (V)')
        plt.show()
    elif mode=='Vclamp':
        T, I = expdata(Address, Index, mode)
        plt.plot(T,I, label=Address)
        plt.legend()
        plt.title(Title)
        plt.xlabel('Time (s)')
        plt.ylabel('Holding current (A)')
        plt.show()


if __name__ == "__main__":
    expcell_list = []
    for cell in tqdm(validcells):
        expcell_list.append(expcell(cell, f"../expdata/Chirp/{cell}"))

    for cell in expcell_list:
        print(cell.ID)
        T, V = expdata(cell.preFIfile, 10, mode='Iclamp', LJP=15e-3)
        plt.plot(T,V)
        T, V = expdata(cell.preFIfile, 16, mode='Iclamp', LJP=15e-3)
        plt.plot(T,V)
        plt.show()

    # for cell in expcell_list:
    #     print(cell.ID, cell.chirpfiles, cell.Injcurr, sep='\n')
    #     print('################################################')

    # for cell in tqdm(expcell_list):
    #     cell.ampphase_freq(True)

    # for cell in tqdm(expcell_list):
    #     for i in range(len(cell.chirpT)):
    #         print(cell.ID, i, cell.sProtocolPath[i])
    #         plt.plot(cell.chirpT[i], cell.chirpV[i])

    # plt.show()

    # for cell in tqdm(expcell_list):
    #     cell.ampphase_freq()
    #     print(cell.ID, cell.chirpfiles, cell.Injcurr, sep="\n")
    #     for i in range(len(cell.chirpT)):
    #         fig, ax = plt.subplots(2, 1, figsize=[12.8, 9.6])
    #         ax[0].plot(cell.chirpT[i], cell.chirpV[i], label=f"{cell.ID}, repeat {i}")
    #         ax[0].legend()
    #         ax[0].set_xlabel("Frequency (Hz)")
    #         ax[0].set_ylabel("Normalized Vm ((raw Vm - median Vm)/chirp amplitude)")
    #         ax[1].plot(cell.freq[i], cell.impedance[i], label=f"{cell.ID}, repeat {i}")
    #         ax[1].set_xscale("log")
    #         ax[1].set_xlabel("Frequency (Hz)")
    #         ax[1].set_ylabel("Impedance (ohms)")
    #         ax[1].legend()
    #         plt.savefig(f"allchirpfigures/{cell.ID}_{i}.png")
    #         # plt.show()
    #         plt.close("all")

    # freq_list = []
    # impedance_list = []
    # phase_list = []
    # for cell in tqdm(expcell_list):
    #     for i in range(len(cell.freq)):
    #         print(cell.ID, i, cell.sProtocolPath[i])
    #         freq_list.append(cell.freq[i])
    #         impedance_list.append(cell.impedance[i])
    #         phase_list.append(cell.phase[i])

    # freqdf = pd.DataFrame({'freq': np.ravel(freq_list),
    #                    'impedance': np.ravel(impedance_list)})
    # g = sns.relplot(data=freqdf, kind='line', x='freq', y='impedance', errorbar="sd")
    # g.set_axis_labels("Frequency (Hz)", "Impedance (ohms)")
    # plt.show()
