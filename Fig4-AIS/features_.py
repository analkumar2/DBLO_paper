import numpy as np

# import numpy.random as nr
# import quantities as pq
import matplotlib.pyplot as plt

# from neo.io import AxonIO
import expcells

import MOOSEModel_ as mm

# import moose
# import os
# import csv
# import scipy.signal as scs
# import scipy.interpolate as sciip
# import scipy.optimize as scioz
# import scipy.stats as scst
# import pandas as pd
import brute_curvefit as bcf
from pprint import pprint

# from copy import deepcopy
# import argparse
# import pickle
# from scipy.signal import butter, filtfilt
# from scipy.signal import hilbert, chirp
from scipy.fft import fft, fftfreq, fftshift

# import multiprocessing
# from multiprocessing import Pool
# import time
# import argparse
# from pprint import pprint
# import pickle
# from copy import deepcopy
import warnings
import efel


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=RuntimeError)
# warnings.filterwarnings("error", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# from sklearn.linear_model import LinearRegression


def calcRinCin(
    tm25,
    Vtracem25,
    stim_start,
    stim_end,
):
    E_rest = np.median(Vtracem25[(tm25 >= stim_start-0.1) & (tm25 <= stim_start)])
    E = np.median(Vtracem25[(tm25 >= stim_start-0.001) & (tm25 <= stim_start)])

    def chargingm25(t, R, tau):
        return E - R * 25e-12 * (1 - np.exp(-t / tau))

    tillt_idx = np.argmin(Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_end)])
    tillt = tm25[(tm25 >= stim_start) & (tm25 <= stim_end)][tillt_idx] - stim_start
    tempv = Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_start + tillt)]
    tempt = np.linspace(0, tillt, len(tempv))
    RCfitted_chm25, errorm25 = bcf.brute_scifit(
        chargingm25,
        tempt,
        tempv,
        restrict=[[5e6, 0], [1000e6, 0.150]],
        ntol=1000,
        printerrors=False,
        parallel=False,
    )
    # print(RCfitted_chm25, errorm25)
    # plt.plot(tm25-stim_start, Vtracem25)
    # plt.plot(np.linspace(0, 0.5, 10000), chargingm25(np.linspace(0, 0.5, 10000), *RCfitted_chm25))
    # plt.show()
    Rin = RCfitted_chm25[0]
    tau = RCfitted_chm25[1]
    Cin = tau / Rin
    Vss = np.median(Vtracem25[(tm25 >= stim_start + 0.4) & (tm25 < stim_start + 0.5)])
    sag_rat = (Vss - (E_rest - 25e-12 * Rin)) / (E_rest - Vss)
    return [E_rest, Rin, Cin, tau, sag_rat]


def calcImpedance(Itrace, Vtrace, dt):
    vvv = Vtrace - np.mean(Vtrace)
    Vmsp = fftshift(fft(vvv))
    Isp = fftshift(fft(Itrace))
    freq = fftshift(fftfreq(vvv.shape[-1], d=dt))
    Vmsp_by_Isp = Vmsp / Isp

    # plt.plot(freq[freq>=0], np.abs(Isp)[freq>=0])
    # plt.plot(freq[freq>=0], np.abs(Vmsp)[freq>=0])
    # plt.show()
    # Impedance = np.abs(Vmsp)/np.abs(Isp)
    Impedance = np.abs(Vmsp_by_Isp)
    # phase = np.arctan(np.imag(Vmsp_by_Isp)/np.imag(Vmsp_by_Isp))
    phase = np.angle(Vmsp_by_Isp)

    return [freq[freq >= 0], Impedance[freq >= 0], phase[freq >= 0]]


def ftscalcpas_helper(
    features,
    t0,
    Vtrace0,
    tm50,
    Vtracem50,
    tm25,
    Vtracem25,
    stim_start,
    stim_end,
):

    featurekeys_list = [
        "Sampling rate",
        "stim_start",
        "stim_end",
        "E_rest_0",
        "E_rest_m50",
        "E_rest_m25",
        "Input resistance",
        "Cell capacitance",
        "Time constant",
        "Equalization time",
        "sagSS_m50",
        "sagV_m50",
        "sagrat_m50",
    ]
    for fkey in featurekeys_list:
        if fkey not in features.keys():
            features[fkey] = None
    features["Sampling rate"] = len(tm25) / tm25[-1]
    Samprate = features["Sampling rate"]
    features["stim_start"] = stim_start
    features["stim_end"] = stim_end

    ## Not processing if it fires even at -25pA current injection
    trace1 = {}
    trace1["T"] = tm25 * 1e3
    trace1["V"] = Vtracem25 * 1e3
    trace1["stim_start"] = [stim_start * 1e3]
    trace1["stim_end"] = [stim_end * 1e3]
    traces = [trace1]
    traces_results = efel.getFeatureValues(traces, ["Spikecount_stimint"])
    # print(traces_results)
    if traces_results[0]["Spikecount_stimint"][0] >= 4:
        # plt.plot(tt,vv)
        # plt.show()
        return features

    ## Erest
    features[f"E_rest_0"] = np.median(Vtrace0)
    if stim_start < 0.2:
        features[f"E_rest_m50"] = np.median(Vtracem50[tm50 < stim_start])
        features[f"E_rest_m25"] = np.median(Vtracem25[tm25 < stim_start])
    else:
        features[f"E_rest_m50"] = np.median(
            Vtracem50[(tm50 < stim_start) & (tm50 > stim_start - 0.1)]
        )
        features[f"E_rest_m25"] = np.median(
            Vtracem25[(tm25 < stim_start) & (tm25 > stim_start - 0.1)]
        )

    ## Input resistance and Cell capacitance
    def chargingm25(t, R1, R2, tau1, tau2):
        return (
            Vtracem25[np.argmin(np.abs(tm25 - stim_start))]
            - R1 * 25e-12 * (1 - np.exp(-t / tau1))
            - R2 * 25e-12 * (1 - np.exp(-t / tau2))
        )

    tillt_idx = np.argmin(Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_end)])
    tillt = tm25[(tm25 >= stim_start) & (tm25 <= stim_end)][tillt_idx] - stim_start
    tempv = Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_start + tillt)]
    tempt = np.linspace(0, tillt, len(tempv))
    RCfitted_chm25, errorm25 = bcf.brute_scifit(
        chargingm25,
        tempt,
        tempv,
        restrict=[[0, 0, 0, 0], [250e6, 50e6, 0.050, 0.002]],
        ntol=1000,
        printerrors=False,
    )
    # print(RCfitted_chm25, errorm25)
    # tempvv = Vtracem25[(tm25 >= stim_start) & (tm25 <= stim_start+0.6)]
    # plt.plot(np.linspace(0, 0.6, len(tempvv)), tempvv)
    # plt.plot(np.linspace(0, 0.6, len(tempvv)), chargingm25(np.linspace(0, 0.6, len(tempvv)), *RCfitted_chm25))
    # plt.show()
    features[f"Input resistance"] = RCfitted_chm25[0] + RCfitted_chm25[1]
    if RCfitted_chm25[2] > RCfitted_chm25[3]:
        features[f"Cell capacitance"] = RCfitted_chm25[2] / RCfitted_chm25[0]
    else:
        features[f"Cell capacitance"] = RCfitted_chm25[3] / RCfitted_chm25[1]

    features[f"Equalization time"] = min(RCfitted_chm25[2], RCfitted_chm25[3])
    features[f"Time constant"] = max(RCfitted_chm25[2], RCfitted_chm25[3])

    features[f"sagSS_m50"] = np.median(
        Vtracem50[(tm50 < stim_end) & (tm50 > stim_end - 0.1)]
    )
    features[f"sagV_m50"] = features[f"sagSS_m50"] - (
        features[f"E_rest_m50"] - 50e-12 * features[f"Input resistance"]
    )
    features[f"sagrat_m50"] = features[f"sagV_m50"] / (
        features[f"E_rest_m50"] - features[f"sagSS_m50"]
    )

    return features


def ftscalc_helper(
    features,
    t0,
    Vtrace0,
    tm50,
    Vtracem50,
    tm25,
    Vtracem25,
    t150,
    Vtrace150,
    t300,
    Vtrace300,
    stim_start,
    stim_end,
):
    featurekeys_list = [
        "Sampling rate",
        "stim_start",
        "stim_end",
        "E_rest_0",
        "E_rest_m50",
        "E_rest_m25",
        "E_rest_150",
        "E_rest_300",
        "Input resistance",
        "Cell capacitance",
        "Time constant",
        "Equalization time",
        "sagSS_m50",
        "sagV_m50",
        "sagrat_m50",
        "freq_0",
        "AP1_amp_1.5e-10",
        "APp_amp_1.5e-10",
        "AP1_time_1.5e-10",
        "APp_time_1.5e-10",
        "APavgpratio_amp_1.5e-10",
        "AP1_width_1.5e-10",
        "APp_width_1.5e-10",
        "AP1_thresh_1.5e-10",
        "APp_thresh_1.5e-10",
        "AP1_lat_1.5e-10",
        "ISI1_1.5e-10",
        "ISIl_1.5e-10",
        "ISIavg_1.5e-10",
        "ISImedian_1.5e-10",
        "freq_1.5e-10",
        "Adptn_id_1.5e-10",
        "fAHP_AP1_amp_1.5e-10" "mAHP_stimend_amp_1.5e-10",
        "sAHP_stimend_amp_1.5e-10",
        "AHP_AP1_amp_1.5e-10",
        "AHP_APp_amp_1.5e-10",
        "AHP_AP1_time_1.5e-10",
        "AHP_APp_time_1.5e-10",
        "Upstroke_AP1_time_1.5e-10",
        "Upstroke_APp_time_1.5e-10",
        "Upstroke_AP1_amp_1.5e-10",
        "Upstroke_APp_amp_1.5e-10",
        "Upstroke_AP1_value_1.5e-10",
        "Upstroke_APp_value_1.5e-10",
        "Downstroke_AP1_time_1.5e-10",
        "Downstroke_APp_time_1.5e-10",
        "Downstroke_AP1_amp_1.5e-10",
        "Downstroke_APp_amp_1.5e-10",
        "Downstroke_AP1_value_1.5e-10",
        "Downstroke_APp_value_1.5e-10",
        "UpDn_AP1_ratio_1.5e-10",
        "UpDn_APp_ratio_1.5e-10",
        "UpThr_AP1_diff_1.5e-10",
        "UpThr_APp_diff_1.5e-10",
        "DBLO_1.5e-10",
        "DBL_1.5e-10",
        "AP1_amp_3e-10",
        "APp_amp_3e-10",
        "AP1_time_3e-10",
        "APp_time_3e-10",
        "APavgpratio_amp_3e-10",
        "AP1_width_3e-10",
        "APp_width_3e-10",
        "AP1_thresh_3e-10",
        "APp_thresh_3e-10",
        "AP1_lat_3e-10",
        "ISI1_3e-10",
        "ISIl_3e-10",
        "ISIavg_3e-10",
        "ISImedian_3e-10",
        "freq_3e-10",
        "Adptn_id_3e-10",
        "fAHP_AP1_amp_3e-10" "mAHP_stimend_amp_3e-10",
        "sAHP_stimend_amp_3e-10",
        "AHP_AP1_amp_3e-10",
        "AHP_APp_amp_3e-10",
        "AHP_AP1_time_3e-10",
        "AHP_APp_time_3e-10",
        "Upstroke_AP1_time_3e-10",
        "Upstroke_APp_time_3e-10",
        "Upstroke_AP1_amp_3e-10",
        "Upstroke_APp_amp_3e-10",
        "Upstroke_AP1_value_3e-10",
        "Upstroke_APp_value_3e-10",
        "Downstroke_AP1_time_3e-10",
        "Downstroke_APp_time_3e-10",
        "Downstroke_AP1_amp_3e-10",
        "Downstroke_APp_amp_3e-10",
        "Downstroke_AP1_value_3e-10",
        "Downstroke_APp_value_3e-10",
        "UpDn_AP1_ratio_3e-10",
        "UpDn_APp_ratio_3e-10",
        "UpThr_AP1_diff_3e-10",
        "UpThr_APp_diff_3e-10",
        "DBLO_3e-10",
        "DBL_3e-10",
        "freq300to150ratio",
    ]
    efelFkeyslist = [
        "peak_voltage",
        "AP1_amp",
        "AP_amplitude",
        "time_to_first_spike",
        "mean_AP_amplitude",
        "AP_duration_half_width",
        "all_ISI_values",
        "Spikecount_stimint",
        "adaptation_index",
        "AP_begin_time",
        "AP_begin_voltage",
        "peak_time",
        "min_between_peaks_values",
        "fast_AHP",
    ]
    for f in featurekeys_list:
        features[f] = None
        
    features = ftscalcpas_helper(
        features, t0, Vtrace0, tm50, Vtracem50, tm25, Vtracem25, stim_start, stim_end
    )
    Samprate = features["Sampling rate"]

    ## Erest
    tracem50, tracem25, trace0, trace150, trace300 = {}, {}, {}, {}, {}
    tracem50["T"], tracem25["T"], trace0["T"], trace150["T"], trace300["T"] = (
        tm50 * 1e3,
        tm25 * 1e3,
        t0 * 1e3,
        t150 * 1e3,
        t300 * 1e3,
    )
    tracem50["V"], tracem25["V"], trace0["V"], trace150["V"], trace300["V"] = (
        Vtracem50 * 1e3,
        Vtracem25 * 1e3,
        Vtrace0 * 1e3,
        Vtrace150 * 1e3,
        Vtrace300 * 1e3,
    )
    (
        tracem50["stim_start"],
        tracem25["stim_start"],
        trace0["stim_start"],
        trace150["stim_start"],
        trace300["stim_start"],
    ) = [[stim_start * 1e3]] * 5
    (
        tracem50["stim_end"],
        tracem25["stim_end"],
        trace0["stim_end"],
        trace150["stim_end"],
        trace300["stim_end"],
    ) = [[stim_end * 1e3]] * 5
    (
        tracem50["stimulus_current"],
        tracem25["stimulus_current"],
        trace0["stimulus_current"],
        trace150["stimulus_current"],
        trace300["stimulus_current"],
    ) = ([-50e-3], [-25e-3], [0e-3], [150e-3], [300e-3])

    #### freq at 0pA ############
    traces_results = efel.getFeatureValues(
        [trace0],
        ["Spikecount_stimint"],
    )
    trace_result = traces_results[0]
    try:
        features["freq_0"] = trace_result["Spikecount_stimint"][0] * 2
    except Exception:
        features["freq_0"] = None
    ####################################

    traces_results = efel.getFeatureValues(
        [trace150, trace300],
        ["voltage_base"],
    )

    features[f"E_rest_150"] = traces_results[0]["voltage_base"][0] * 1e-3
    features[f"E_rest_300"] = traces_results[1]["voltage_base"][0] * 1e-3

    currlist = [150e-12, 300e-12]
    for I in currlist:
        if I == 150e-12:
            trace150["T"] = t150[(t150>=stim_start) & ((t150<stim_end))] * 1e3
            trace150["V"] = Vtrace150[(t150>=stim_start) & ((t150<stim_end))] * 1e3
            trace150["stim_start"] = [trace150["T"][0]]
            trace150["stim_end"] = [trace150["T"][-1]]
            trace150["stimulus_current"] = [150e-3]
            traces_results = efel.getFeatureValues(
                [trace150],
                efelFkeyslist,
            )
            trace_result = traces_results[0]
            # pprint(trace_result)
        elif I == 300e-12:
            trace300["T"] = t300[(t300>=stim_start) & ((t300<stim_end))] * 1e3
            trace300["V"] = Vtrace300[(t300>=stim_start) & ((t300<stim_end))] * 1e3
            trace300["stim_start"] = [trace300["T"][0]]
            trace300["stim_end"] = [trace300["T"][-1]]
            trace300["stimulus_current"] = [300e-3]
            traces_results = efel.getFeatureValues(
                [trace300],
                efelFkeyslist,
            )
            trace_result = traces_results[0]
            # pprint(trace_result)

        # AP1_amp
        try:
            features[f"AP1_amp_{I}"] = trace_result["AP1_amp"][0] * 1e-3
        except Exception:
            features[f"AP1_amp_{I}"] = None

        # APp_amp
        try:
            features[f"APp_amp_{I}"] = trace_result["AP_amplitude"][-2] * 1e-3
        except Exception:
            features[f"APp_amp_{I}"] = None

        # AP1_time
        try:
            features[f"AP1_time_{I}"] = trace_result["time_to_first_spike"][0] * 1e-3
        except Exception:
            features[f"AP1_time_{I}"] = None

        # APp_time
        try:
            features[f"APp_time_{I}"] = trace_result["peak_time"][-2] * 1e-3 - stim_start
        except Exception:
            features[f"APp_time_{I}"] = None

        # APavgpratio_amp
        try:
            features[f"APavgpratio_amp_{I}"] = (
                trace_result["AP_amplitude"][-2] / trace_result["mean_AP_amplitude"][0]
            )
        except Exception:
            features[f"APavgpratio_amp_{I}"] = None

        # AP1_width
        try:
            features[f"AP1_width_{I}"] = trace_result["AP_duration_half_width"][0] * 1e-3
        except Exception:
            features[f"AP1_width_{I}"] = None

        # APp_width
        try:
            features[f"APp_width_{I}"] = trace_result["AP_duration_half_width"][-2] * 1e-3
        except Exception:
            features[f"APp_width_{I}"] = None

        # AP1_thresh
        try:
            features[f"AP1_thresh_{I}"] = trace_result["AP_begin_voltage"][0] * 1e-3
        except Exception:
            features[f"AP1_thresh_{I}"] = None

        # APp_thresh
        try:
            features[f"APp_thresh_{I}"] = trace_result["AP_begin_voltage"][-2] * 1e-3
        except Exception:
            features[f"APp_thresh_{I}"] = None

        # AP1_lat
        try:
            features[f"AP1_lat_{I}"] = trace_result["AP_begin_time"][0] * 1e-3 - stim_start
        except Exception:
            features[f"AP1_lat_{I}"] = None

        # ISI1
        try:
            features[f"ISI1_{I}"] = trace_result["all_ISI_values"][0] * 1e-3
        except Exception:
            features[f"ISI1_{I}"] = None

        # ISIl
        try:
            features[f"ISIl_{I}"] = trace_result["all_ISI_values"][-1] * 1e-3
        except Exception:
            features[f"ISIl_{I}"] = None

        # ISIavg
        try:
            features[f"ISIavg_{I}"] = np.nanmean(trace_result["all_ISI_values"]) * 1e-3
        except Exception:
            features[f"ISIavg_{I}"] = None

        # ISImedian
        try:
            features[f"ISImedian_{I}"] = np.nanmedian(trace_result["all_ISI_values"]) * 1e-3
        except Exception:
            features[f"ISImedian_{I}"] = None

        # freq
        try:
            features[f"freq_{I}"] = trace_result["Spikecount_stimint"][0] * 2
        except Exception:
            features[f"freq_{I}"] = None

        # Adptn_id
        try:
            features[f"Adptn_id_{I}"] = trace_result["adaptation_index"][0]
        except Exception:
            features[f"Adptn_id_{I}"] = None

        # fAHP_AP1_amp
        try:
            features[f"fAHP_AP1_amp_{I}"] = trace_result["fast_AHP"][0] * 1e-3
        except Exception:
            features[f"fAHP_AP1_amp_{I}"] = None

        # fAHP_APp_amp
        try:
            features[f"fAHP_APp_amp_{I}"] = trace_result["fast_AHP"][-2] * 1e-3
        except Exception:
            features[f"fAHP_APp_amp_{I}"] = None

        # # mAHP_AP1_amp
        # features[f"mAHP_AP1_amp_{I}"] = trace_result['threshold']

        # # mAHP_APp_amp
        # features[f"mAHP_APp_amp_{I}"] = trace_result['threshold']

        # # mAHP_AP1_time
        # features[f"mAHP_AP1_time_{I}"] = trace_result['threshold']

        # # mAHP_APp_time = mAHP of second last spike (penultimate)
        # features[f"mAHP_APp_time_{I}"] = trace_result['threshold']

        # # ADP_AP1_amp
        # features[f"ADP_AP1_amp_{I}"] = trace_result['threshold']

        # # ADP_APp_amp
        # features[f"ADP_APp_amp_{I}"] = trace_result['threshold']

        # # mAHP_stimend_amp = within 50ms
        # features[f"mAHP_stimend_amp_{I}"] = trace_result['threshold']

        # # sAHP_stimend_amp = within 200ms
        # features[f"sAHP_stimend_amp_{I}"] = trace_result['threshold']

        # # AHP_AP1_amp = Minimum between the first two spikes
        # features[f"AHP_AP1_amp_{I}"] = trace_result['threshold']

        # # AHP_APp_amp = Minimum between the last two spikes
        # features[f"AHP_APp_amp_{I}"] = trace_result['threshold']

        # # AHP_AP1_time = Minimum between the first two spikes
        # features[f"AHP_AP1_time_{I}"] = trace_result['threshold']

        # # AHP_APp_time = Minimum between the last two spikes
        # features[f"AHP_APp_time_{I}"] = trace_result['threshold']

        # # Upstroke_AP1_time = Upstroke time of first spike
        # features[f"Upstroke_AP1_time_{I}"] = trace_result['threshold']

        # # Upstroke_APp_time = Upstroke time of penultimate spike
        # features[f"Upstroke_APp_time_{I}"] = trace_result['threshold']

        # # Upstroke_AP1_amp = Upstroke amp of first spike
        # features[f"Upstroke_AP1_amp_{I}"] = trace_result['threshold']

        # # Upstroke_APp_amp = Upstroke amp of penultimate spike
        # features[f"Upstroke_APp_amp_{I}"] = trace_result['threshold']

        # # Upstroke_AP1_value = Upstroke value of first spike
        # features[f"Upstroke_AP1_value_{I}"] = trace_result['threshold']

        # # Upstroke_APp_value = Upstroke value of penultimate spike
        # features[f"Upstroke_APp_value_{I}"] = trace_result['threshold']

        # # Downstroke_AP1_time = Downstroke time of first spike
        # features[f"Downstroke_AP1_time_{I}"] = trace_result['threshold']

        # # Downstroke_APp_time = Downstroke time of penultimate spike
        # features[f"Downstroke_APp_time_{I}"] = trace_result['threshold']

        # # Downstroke_AP1_amp = Downstroke amp of first spike
        # features[f"Downstroke_AP1_amp_{I}"] = trace_result['threshold']

        # # Downstroke_APp_amp = Downstroke amp of penultimate spike
        # features[f"Downstroke_APp_amp_{I}"] = trace_result['threshold']

        # # Downstroke_AP1_value = Downstroke value of first spike
        # features[f"Downstroke_AP1_value_{I}"] = trace_result['threshold']

        # # Downstroke_APp_value = Downstroke value of penultimate spike
        # features[f"Downstroke_APp_value_{I}"] = trace_result['threshold']

        # # UpDn_AP1_ratio = Upstroke/Downstroke ratio of first spike
        # features[f"UpDn_AP1_ratio_{I}"] = trace_result['threshold']

        # # UpDn_APp_ratio = Upstroke/Downstroke ratio of penultimate spike
        # features[f"UpDn_APp_ratio_{I}"] = trace_result['threshold']

        # # UpThr_AP1_diff = Upstroke_v - threshold_v of first spike
        # features[f"UpThr_AP1_diff_{I}"] = trace_result['threshold']

        # # UpThr_APp_diff = Upstroke_v - threshold_v of penultimate spike
        # features[f"UpThr_APp_diff_{I}"] = trace_result['threshold']

        # DBLO
        try:
            features[f"DBLO_{I}"] = (
                np.nanmean(trace_result["min_between_peaks_values"][:-1]) * 1e-3 #Last ISI ignored because the efel package appends for some reason the last voltage point to the min_between_peaks_values
                - features[f"E_rest_150"]
            )
        except Exception:
            features[f"DBLO_{I}"] = None

        # DBL
        try:
            features[f"DBL_{I}"] = (
                np.nanmean(trace_result["min_between_peaks_values"][:-1]) * 1e-3 #Last ISI ignored because the efel package appends for some reason the last voltage point to the min_between_peaks_values
            )
        except Exception:
            features[f"DBL_{I}"] = None

    # frequency 300pA to frequency 150pA ratio
    try:
        features[f"freq300to150ratio"] = features[f"freq_3e-10"] / features[f"freq_1.5e-10"]
    except Exception:
        features[f"freq300to150ratio"] = None

    return features


def expfeatures(cellpath, stim_start, stim_end, LJP=0):
    t0, Vtrace0 = expcells.expdata(cellpath, 4, LJP=LJP)
    tm50, Vtracem50 = expcells.expdata(cellpath, 2, LJP=LJP)
    tm25, Vtracem25 = expcells.expdata(cellpath, 3, LJP=LJP)
    t150, Vtrace150 = expcells.expdata(cellpath, 10, LJP=LJP)
    t300, Vtrace300 = expcells.expdata(cellpath, 16, LJP=LJP)

    features = {}
    features["Cell name"] = cellpath.split("/")[-1]
    features = ftscalc_helper(
        features,
        t0,
        Vtrace0,
        tm50,
        Vtracem50,
        tm25,
        Vtracem25,
        t150,
        Vtrace150,
        t300,
        Vtrace300,
        stim_start,
        stim_end,
    )

    return features


def modelfeatures(
    modeldict, stim_start=1, stim_end=1.5, refreshKin=True
):
    t0, Itrace0, Vtrace0, Ca = mm.runModel(modeldict, 0e-12, refreshKin=refreshKin)
    tm50, Itrace50, Vtracem50, Ca = mm.runModel(
        modeldict, -50e-12, refreshKin=False
    )
    tm25, Itrace25, Vtracem25, Ca = mm.runModel(
        modeldict, -25e-12, refreshKin=False
    )
    t150, Itrace150, Vtrace150, Ca = mm.runModel(
        modeldict, 150e-12, refreshKin=False
    )
    t300, Itrace300, Vtrace300, Ca = mm.runModel(
        modeldict, 300e-12, refreshKin=False
    )
    # plt.plot(t150, Vtrace150)

    features = {}
    # features['Model name'] = cellpath.split('/')[-1]
    features = ftscalc_helper(
        features,
        t0,
        Vtrace0,
        tm50,
        Vtracem50,
        tm25,
        Vtracem25,
        t150,
        Vtrace150,
        t300,
        Vtrace300,
        stim_start,
        stim_end,
    )

    return features
