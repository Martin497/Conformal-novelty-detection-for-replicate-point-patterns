#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:36:10 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Script to run a multiple testing procedure on the sweat gland data p-values.

Reproduces Figure 12.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Storeys_correction(lambda_, p_hat, alpha, m):
    """
    Estimating the proportion of true nulls, pi0, using Storey's estimate.
    Then make the test sequence to be used for the Benjamini-Hochberg procedure.
    """
    pi0_hat = min(1, (1 + np.sum(p_hat > lambda_)) / (m * (1 - lambda_)))
    m0_hat = np.ceil(pi0_hat * m)
    return pi0_hat, m0_hat

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")

    # test_ids = [10, 20, 40, 61, 71, 96, 149, 203, 205]
    test_ids = [10, 20, 40, 61, 71, 23, 36, 42, 50, 73]
    m = len(test_ids)
    alpha = 0.05
    lambda_ = 0.25
    u_margMC = pd.read_csv("p_values/gm_control_MonteCarlo_n5000.csv", index_col=(0)).to_numpy()[:, 0]
    u_marg = pd.read_csv("p_values/gm_control_Conformal_n5000.csv", index_col=(0)).to_numpy()[:, 0]

    pi0_hat, m0_hat = Storeys_correction(lambda_, u_marg, alpha, m)

    test_sequence = np.arange(1, m+1)/m0_hat * alpha
    sort_ = np.argsort(u_marg).astype(np.int16)
    u_marg_sorted = u_marg[sort_]
    rejectBoolBH = np.zeros(m, dtype=bool)
    iter_ = m-1
    while iter_ >= 0:
        if u_marg_sorted[iter_] <= test_sequence[iter_]:
            rejectBoolBH[sort_[iter_]] = True
            break
        else:
            rejectBoolBH[sort_[iter_]] = False
        iter_ -= 1

    legend_bool = np.ones(4, dtype=bool)
    for idx in range(m):
        if idx < 5 and rejectBoolBH[idx] == True:
            color = "tab:orange"
            marker = "x"
            legend = "Rejected MNA suspected"
            if legend_bool[0] == True:
                plt.plot(idx, u_marg[idx], marker, color=color, label=legend)
            else:
                plt.plot(idx, u_marg[idx], marker, color=color)
            legend_bool[0] = False
        elif idx < 5 and rejectBoolBH[idx] == False:
            color = "tab:orange"
            marker = "o"
            legend = "Not-rejected MNA suspected"
            if legend_bool[1] == True:
                plt.plot(idx, u_marg[idx], marker, color=color, label=legend)
            else:
                plt.plot(idx, u_marg[idx], marker, color=color)
            legend_bool[1] = False
        elif idx >= 5 and rejectBoolBH[idx] == True:
            color = "tab:green"
            marker = "x"
            legend = "Rejected MNA diagnosed"
            if legend_bool[2] == True:
                plt.plot(idx, u_marg[idx], marker, color=color, label=legend)
            else:
                plt.plot(idx, u_marg[idx], marker, color=color)
            legend_bool[2] = False
        elif idx >= 5 and rejectBoolBH[idx] == False:
            color = "tab:green"
            marker = "o"
            legend = "Not-rejected MNA diagnosed"
            if legend_bool[3] == True:
                plt.plot(idx, u_marg[idx], marker, color=color, label=legend)
            else:
                plt.plot(idx, u_marg[idx], marker, color=color)
            legend_bool[3] = False
    plt.yscale("log")
    plt.xticks(np.arange(m), test_ids)
    plt.ylabel("p-value")
    plt.xlabel("Test subject id")
    for iter_ in range(m):
        plt.plot(sort_[iter_], test_sequence[iter_], "_", color="tab:purple")
    plt.show()

    # =============================================================================
    # Power comparison
    # =============================================================================
    alpha_arr = np.logspace(-4, 0, 1000)
    rejectBoolBH_CMMCTest = np.zeros((len(alpha_arr), m), dtype=bool)
    for alpha_idx, alpha in enumerate(alpha_arr):
        test_sequence = np.arange(1, m+1)/m0_hat * alpha
        sort_ = np.argsort(u_marg).astype(np.int16)
        u_marg_sorted = u_marg[sort_]
        iter_ = 0
        while iter_ < m:
            if u_marg_sorted[iter_] > test_sequence[iter_]:
                rejectBoolBH_CMMCTest[alpha_idx, sort_[iter_]] = False
            else:
                rejectBoolBH_CMMCTest[alpha_idx, sort_[iter_]] = True
            iter_ += 1
    
    rejectBoolBH_MMCTest = np.zeros((len(alpha_arr), m), dtype=bool)
    for alpha_idx, alpha in enumerate(alpha_arr):
        test_sequence = np.arange(1, m+1)/m0_hat * alpha
        sort_ = np.argsort(u_margMC).astype(np.int16)
        u_marg_sorted = u_margMC[sort_]
        iter_ = 0
        while iter_ < m:
            if u_marg_sorted[iter_] > test_sequence[iter_]:
                rejectBoolBH_MMCTest[alpha_idx, sort_[iter_]] = False
            else:
                rejectBoolBH_MMCTest[alpha_idx, sort_[iter_]] = True
            iter_ += 1

    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    plt.plot(alpha_arr, np.sum(rejectBoolBH_MMCTest[:, 5:], axis=1), color="tab:orange", label="MMCTest control")
    plt.plot(alpha_arr, np.sum(rejectBoolBH_CMMCTest[:, 5:], axis=1), color="tab:purple", label="CMMCTest control")
    plt.plot(alpha_arr, np.sum(rejectBoolBH_MMCTest[:, :5], axis=1), linestyle="dashed", color="tab:orange", label="MMCTest suspected")
    plt.plot(alpha_arr, np.sum(rejectBoolBH_CMMCTest[:, :5], axis=1), linestyle="dashed", color="tab:purple", label="CMMCTest suspected")
    plt.xscale("log")
    plt.ylabel("Rejections")
    plt.xlabel("$\\alpha$")
    plt.legend()
    plt.show()
