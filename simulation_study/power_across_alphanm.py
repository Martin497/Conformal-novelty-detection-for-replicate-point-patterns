#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:33:51 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Script to import p-values, run a multiple testing procedure, and plot the
resulting empirical metrics (TDR), for varying significance levels, alpha,
varying null sample sizes, n, and varying number of test points, m.

Reproduces Figure 8.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds

import multiple_testing_module as mtm


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha = 0.2
    lambda_ = 0.5
    use_m0_est = True

    null_model_list = ["Poisson_Mrkvicka"]
    test_model_list = ["Strauss_Mrkvicka"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    # =============================================================================
    # Conformal
    # =============================================================================
    method_list = ["CMMCTest"]
    folder_append = ["_14"]

    n_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    m_list = [6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    TDR_tot = np.zeros((len(method_list), len(n_list), len(m_list), null_number_models, number_models), dtype=np.float64)
    FDR_tot = np.zeros((len(method_list), len(n_list), len(m_list), null_number_models, number_models), dtype=np.float64)
    for method_idx, (method, fapp) in enumerate(zip(method_list, folder_append)):
        folder = f"p_values/{method}{fapp}"
        for idx1 in range(null_number_models):
            for idx2 in range(number_models):
                name = f"{null_model_list[idx1]}_{test_model_list[idx2]}"

                try:
                    u_marg = pd.read_csv(f"{folder}/{name}.csv", index_col=(0)).to_numpy()
                except FileNotFoundError:
                    res = read_rds(f"{folder}/{name}.rds")
                    u_marg = res["data"].reshape(res["attributes"]["dim"]["data"], order="F")
                if "Estimate" in fapp:
                    u_marg = u_marg[0]
                data_sims = u_marg.shape[0]
                lenN = u_marg.shape[1]

                for n_idx in range(len(n_list)):
                    for m_idx in range(len(m_list)):
                        u_marg_ = np.concatenate((u_marg[:, n_idx, :int(m_list[m_idx]/2)], u_marg[:, n_idx, -int(m_list[m_idx]/2):]), axis=-1)
                        rejectBool = mtm.testing_wrapper(u_marg_, alpha, int(m_list[m_idx]/2), m_list[m_idx], data_sims,
                                                         "BH", null_model_list[idx1], test_model_list[idx2],
                                                         lambda_, use_m0_est)

                        TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_list[m_idx]/2), m_list[m_idx], data_sims)
                        TDR_tot[method_idx, n_idx, m_idx, idx1, idx2] = TDR

                        FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_list[m_idx]/2), data_sims)
                        FDR_tot[method_idx, n_idx, m_idx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)

    # =============================================================================
    # Monte Carlo
    # =============================================================================

    MCfolder_append = ["_06", "_03", "_04", "_05"]
    MCmethod_list = ["MMCTest"]*len(MCfolder_append)
    MCn_lists = [[240, 540, 780, 1020, 1380, 1800, 2220, 2520],
                 [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],
                 [300, 400, 700, 1000, 1500, 2000, 2500],
                 [300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1800, 2100, 2250, 2520]]
    MCm_list = [6, 10, 20, 30]
    TDR_MCtot = list()
    FDR_MCtot = list()
    for method_idx, (method, fapp, n_list_iter, m_iter) in enumerate(zip(MCmethod_list, MCfolder_append, MCn_lists, MCm_list)):
        TDR_MCtot.append(np.zeros((len(n_list_iter), null_number_models, number_models), dtype=np.float64))
        FDR_MCtot.append(np.zeros((len(n_list_iter), null_number_models, number_models), dtype=np.float64))
        folder = f"p_values/{method}{fapp}"
        for idx1 in range(null_number_models):
            for idx2 in range(number_models):
                name = f"{null_model_list[idx1]}_{test_model_list[idx2]}"

                try:
                    u_marg = pd.read_csv(f"{folder}/{name}.csv", index_col=(0)).to_numpy()
                except FileNotFoundError:
                    res = read_rds(f"{folder}/{name}.rds")
                    u_marg = res["data"].reshape(res["attributes"]["dim"]["data"], order="F")
                if "Estimate" in fapp:
                    u_marg = u_marg[0]
                data_sims = u_marg.shape[0]
                lenN = u_marg.shape[1]

                for n_idx in range(len(n_list_iter)):
                    u_marg_ = u_marg[:, n_idx, :]
                    rejectBool = mtm.testing_wrapper(u_marg_, alpha, int(m_iter/2), m_iter, data_sims,
                                                     "BH", null_model_list[idx1], test_model_list[idx2],
                                                     lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_iter/2), m_iter, data_sims)
                    TDR_MCtot[method_idx][n_idx, idx1, idx2] = TDR

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_iter/2), data_sims)
                    FDR_MCtot[method_idx][n_idx, idx1, idx2] = FDR

    # =============================================================================
    # Together
    # =============================================================================
    mark_list = ["s", "*", "d", "p"]
    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                for i in range(len(TDR_MCtot)):
                    plt.plot(MCn_lists[i], TDR_MCtot[i][:, null_idx, alt_idx], color="tab:orange", marker=mark_list[i], label=f"MMCTest, m={MCm_list[i]}")
                for m_idx in range(len(TDR_MCtot)):
                    plt.plot(n_list, TDR_tot[0, :, m_idx, null_idx, alt_idx], color="tab:purple", marker=mark_list[m_idx], label=f"CMMCTest, m={m_list[m_idx]}")
                plt.xlabel("n")
                plt.ylabel("TDR")
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.legend()
                plt.show()
