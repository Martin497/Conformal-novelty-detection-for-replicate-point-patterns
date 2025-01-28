#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:12:21 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Script to import p-values, run a multiple testing procedure, and plot the
resulting empirical metrics (FDR, FWER, TDR, etc...),
for varying conformal scores.

Reproduces Figure 6.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds

import multiple_testing_module as mtm


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha_arr = np.linspace(1e-03, 0.5, 25)
    m = 10
    m0 = 5
    lambda_ = 0.5
    use_m0_est = True
    save_res = False

    folder_append = ["_01", "_03", "_04", "_05", "_06", "_07", "_08", "_09", "_10", "_11", "_12", "_13"]
    method_list = ["CMMCTest"]*len(folder_append)

    null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    TDR_tot = np.zeros((len(alpha_arr), len(method_list), null_number_models, number_models), dtype=np.float64)
    FDR_tot = np.zeros((len(alpha_arr), len(method_list), null_number_models, number_models), dtype=np.float64)
    for alpha_idx, alpha in enumerate(alpha_arr):
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
                    if (len(u_marg.shape) == 3) and (u_marg.shape[1] == 1):
                        u_marg = u_marg[:, 0, :]

                    data_sims = u_marg.shape[0]
                    assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                    rejectBool = mtm.testing_wrapper(u_marg, alpha, m0, m, data_sims,
                                                     "BH", null_model_list[idx1], test_model_list[idx2],
                                                     lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, m, data_sims)
                    TDR_tot[alpha_idx, method_idx, idx1, idx2] = TDR

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FDR_tot[alpha_idx, method_idx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)
    savename = "m05"
    # idx_list = [0, 3, 4, 2, 6, 5, 1, 7, 8, 9, 10, 11]

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(alpha_arr, TDR_tot[:, 0, null_idx, alt_idx], color="tab:orange", label="par-L-ERL")
                plt.plot(alpha_arr, TDR_tot[:, 3, null_idx, alt_idx], color="tab:orange", linestyle="dashed", label="par-L-cont")
                plt.plot(alpha_arr, TDR_tot[:, 4, null_idx, alt_idx], color="tab:orange", linestyle="dotted", label="par-L-area")
                plt.plot(alpha_arr, TDR_tot[:, 2, null_idx, alt_idx], color="tab:green", label="par-J-ERL")
                plt.plot(alpha_arr, TDR_tot[:, 6, null_idx, alt_idx], color="tab:green", linestyle="dashed", label="par-J-cont")
                plt.plot(alpha_arr, TDR_tot[:, 5, null_idx, alt_idx], color="tab:green", linestyle="dotted", label="par-J-area")
                plt.plot(alpha_arr, TDR_tot[:, 1, null_idx, alt_idx], color="tab:purple", label="joint-L-ERL")
                plt.plot(alpha_arr, TDR_tot[:, 7, null_idx, alt_idx], color="tab:purple", linestyle="dashed", label="joint-L-cont")
                plt.plot(alpha_arr, TDR_tot[:, 8, null_idx, alt_idx], color="tab:purple", linestyle="dotted", label="joint-L-area")
                plt.plot(alpha_arr, TDR_tot[:, 9, null_idx, alt_idx], color="tab:blue", linestyle="dashed", label="joint-J-erl")
                plt.plot(alpha_arr, TDR_tot[:, 10, null_idx, alt_idx], color="tab:blue", linestyle="dashed", label="joint-J-cont")
                plt.plot(alpha_arr, TDR_tot[:, 11, null_idx, alt_idx], color="tab:blue", linestyle="dotted", label="joint-J-area")
                plt.xlabel("$\\alpha$")
                plt.ylabel("TDR")
                plt.legend()
                plt.title(f"{null_model_list[null_idx]} vs {test_model_list[alt_idx]}")
                if save_res is True:
                    plt.savefig(f"p_values/test_stat_plots/TDR_{null_idx}_{alt_idx}_{savename}.png", dpi=500, bbox_inches="tight")
                plt.show()

        plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        plt.plot(alpha_arr, alpha_arr, color="k")
        plt.plot(alpha_arr, FDR_tot_mean[:, 0, null_idx], color="tab:orange", label="par-L-ERL")
        plt.plot(alpha_arr, FDR_tot_mean[:, 3, null_idx], color="tab:orange", linestyle="dashed", label="par-L-cont")
        plt.plot(alpha_arr, FDR_tot_mean[:, 4, null_idx], color="tab:orange", linestyle="dotted", label="par-L-area")
        plt.plot(alpha_arr, FDR_tot_mean[:, 2, null_idx], color="tab:green", label="par-J-ERL")
        plt.plot(alpha_arr, FDR_tot_mean[:, 6, null_idx], color="tab:green", linestyle="dashed", label="par-J-cont")
        plt.plot(alpha_arr, FDR_tot_mean[:, 5, null_idx], color="tab:green", linestyle="dotted", label="par-J-area")
        plt.plot(alpha_arr, FDR_tot_mean[:, 1, null_idx], color="tab:purple", label="joint-L-ERL")
        plt.plot(alpha_arr, FDR_tot_mean[:, 7, null_idx], color="tab:purple", linestyle="dashed", label="joint-L-cont")
        plt.plot(alpha_arr, FDR_tot_mean[:, 8, null_idx], color="tab:purple", linestyle="dotted", label="joint-L-area")
        plt.plot(alpha_arr, FDR_tot_mean[:, 9, null_idx], color="tab:blue", linestyle="dashed", label="joint-J-erl")
        plt.plot(alpha_arr, FDR_tot_mean[:, 10, null_idx], color="tab:blue", linestyle="dashed", label="joint-J-cont")
        plt.plot(alpha_arr, FDR_tot_mean[:, 11, null_idx], color="tab:blue", linestyle="dotted", label="joint-J-area")
        plt.xlabel("$\\alpha$")
        plt.ylabel("FDR")
        plt.legend()
        plt.title(f"{null_model_list[null_idx]}")
        if save_res is True:
            plt.savefig(f"p_values/test_stat_plots/FDR_{null_idx}_{savename}.png", dpi=500, bbox_inches="tight")
        plt.show()
