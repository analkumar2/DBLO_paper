from neo.io import AxonIO
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import argparse
import tkinter as tk
from tkinter import filedialog
import gc


# @profile
def main():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()

    # def absoluteFilePaths(directory):
    #     for dirpath,_,filenames in os.walk(directory):
    #         for f in filenames:
    #             yield os.path.abspath(os.path.join(dirpath, f))

    filegenerator = []
    for dirpath,_,filenames in os.walk(folder_path):
        for f in filenames:
            tempfilename = os.path.abspath(os.path.join(dirpath, f))
            # if os.path.splitext(tempfilename)[1] == '.dat':
            if os.path.splitext(tempfilename)[1] == '.abf' or os.path.splitext(tempfilename)[1] == '.dat':
                filegenerator.append(tempfilename)
    # absoluteFilePaths(folder_path)

    for file in filegenerator:
        print(file)
        splitted = os.path.splitext(file)
        if os.path.splitext(file)[1] == '.abf':
            if os.path.isfile(file[:-4]+'_Anal.png') or os.path.isfile(file[:-4]+'_Anal.jpg') or os.path.isfile(file[:-4]+'_Anal.jpeg'):
                print(file[:-4]+'_Anal.png already exists')
                continue
            # print(file)
            fig,axs = plt.subplots(1,1,figsize=(19.20,10.80))
            reader = AxonIO(filename=file)
            Samprate = reader.get_signal_sampling_rate()
            for seg in reader.read_block(signal_group_mode='split-all').segments:
                Tdur = np.array(seg.t_stop - seg.t_start)
                V = np.array(np.ravel(seg.analogsignals[0]))
                axs.plot(np.linspace(0,Tdur+0,len(V)), V)
            fig.suptitle(os.path.splitext(os.path.basename(file))[0])
            # axs.set_ylim(-0.1,0.1)
            # axs.set_ylim(-100e-12, 500e-12)
            # axs.set_xlim(0, 5)
            # plt.show()
            # print(splitted[0]+'_Anal.png')
            fig.savefig(splitted[0]+'_Anal.png')
            plt.close()

        if os.path.splitext(file)[1] == '.dat':
            if os.path.isfile(file[:-4]+'_Anal.png') or os.path.isfile(file[:-4]+'_Anal.jpg') or os.path.isfile(file[:-4]+'_Anal.jpeg'):
                print(file[:-4]+'_Anal.png already exists')
                continue
            try:
                fig,axs = plt.subplots(1,1,figsize=(19.20,10.80))
                a = np.loadtxt(file)
                b = np.transpose(a)
                axs.plot(b[0], np.transpose(b[1:]))
                fig.suptitle(os.path.splitext(os.path.basename(file))[0])
                # axs.set_ylim(-0.1,0.1)
                # axs.set_ylim(-100e-12, 500e-12)
                # axs.set_xlim(0, 5)
                # plt.show()
                # print(splitted[0]+'_Anal.png')
                fig.savefig(splitted[0]+'_Anal.png')
                plt.close()
            except:
                print(f'######## \n {file} skipped\n #########') 



if __name__ == '__main__':
    main()
