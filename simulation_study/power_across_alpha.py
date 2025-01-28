#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:06:57 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Script to import p-values, run a multiple testing procedure, and plot the
resulting empirical metrics (FDR, FWER, TDR, etc...),
for varying significance levels, alpha.

Reproduces Figures 3, 4, 9, and 10.
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
    testing_procedure = "BH" # options: "BH", "Hochberg", and more

    method_list = ["MMCTest", "CMMCTest"]
    folder_append = ["_01", "_01"]

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
                                                     testing_procedure, null_model_list[idx1], test_model_list[idx2],
                                                     lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, m, data_sims)
                    TDR_tot[alpha_idx, method_idx, idx1, idx2] = TDR

                    if testing_procedure == "BH":
                        FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    else:
                        FDR = mtm.compute_FWER(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FDR_tot[alpha_idx, method_idx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)
    savename = f"m05_{testing_procedure}"

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(alpha_arr, TDR_tot[:, 0, null_idx, alt_idx], color="tab:orange", label="MMCTest")
                plt.plot(alpha_arr, TDR_tot[:, 1, null_idx, alt_idx], color="tab:purple", label="CMMCTest")
                plt.xlabel("$\\alpha$")
                plt.ylabel("TDR")
                plt.legend()
                plt.title(f"{null_model_list[null_idx]} vs {test_model_list[alt_idx]}")
                if save_res is True:
                    plt.savefig(f"p_values/alpha_plots/alpha_plot_TDR_{null_idx}_{alt_idx}_{savename}.png", dpi=500, bbox_inches="tight")
                plt.show()

                if save_res is True:
                    with open(f"p_values/alpha_plots/alpha_plot_TDR_{null_idx}_{alt_idx}_{savename}.txt", "w") as file:
                        file.write("\\addplot[semithick, mark=diamond, color4]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 0, null_idx, alt_idx]):
                            file.write(f"{x} {y}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=diamond, color2]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 1, null_idx, alt_idx]):
                            file.write(f"{x} {y}\n")
                        file.write("};\n")

        plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        plt.plot(alpha_arr, alpha_arr, color="k")
        plt.plot(alpha_arr, FDR_tot_mean[:, 0, null_idx], color="tab:orange", label="MMCTest")
        plt.plot(alpha_arr, FDR_tot_mean[:, 1, null_idx], color="tab:purple", label="CMMCTest")
        plt.xlabel("$\\alpha$")
        if testing_procedure == "BH":
            plt.ylabel("FDR")
        else:
            plt.ylabel("FWER")
        plt.legend()
        plt.title(f"{null_model_list[null_idx]}")
        if save_res is True:
            plt.savefig(f"p_values/alpha_plots/alpha_plot_FDR_{null_idx}_{savename}.png", dpi=500, bbox_inches="tight")
        plt.show()

        if save_res is True:
            with open(f"p_values/alpha_plots/alpha_plot_FDR_{null_idx}_{savename}.txt", "w") as file:
                file.write("\\addplot[semithick, mark=diamond, color4]\ntable{%\n")
                for x, y in zip(alpha_arr, FDR_tot_mean[:, 0, null_idx]):
                    file.write(f"{x} {y}\n")
                file.write("};\n")
                file.write("\\addplot[semithick, mark=diamond, color2]\ntable{%\n")
                for x, y in zip(alpha_arr, FDR_tot_mean[:, 1, null_idx]):
                    file.write(f"{x} {y}\n")
                file.write("};\n")


