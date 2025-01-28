#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:04:23 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Script to import p-values, run a multiple testing procedure, and plot the
resulting empirical metrics (FDR, FWER, TDR, etc...),
for varying values of Storey's hyperparameter, lambda.

Reproduces Figure 5.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds

import multiple_testing_module as mtm


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha = 0.05
    m = 10
    m0 = 5
    lambda_arr = np.linspace(0, 1, 100, endpoint=False)
    use_m0_est = True
    save_res = False

    method_list = ["MMCTest", "CMMCTest"]
    folder_append = ["_01", "_01"]

    null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    TDR_tot = np.zeros((len(lambda_arr), 2, null_number_models, number_models), dtype=np.float64)
    FDR_tot = np.zeros((len(lambda_arr), 2, null_number_models, number_models), dtype=np.float64)
    for lambda_idx, lambda_ in enumerate(lambda_arr):
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
                    assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                    rejectBool = mtm.testing_wrapper(u_marg, alpha, m0, m, data_sims,
                                                     "BH", null_model_list[idx1], test_model_list[idx2],
                                                     lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, m, data_sims)
                    TDR_tot[lambda_idx, method_idx, idx1, idx2] = TDR

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FDR_tot[lambda_idx, method_idx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_idx != alt_idx:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(lambda_arr, TDR_tot[:, 0, null_idx, alt_idx], color="tab:purple", label="MMCTest")
                plt.plot(lambda_arr, TDR_tot[:, 1, null_idx, alt_idx], color="tab:orange", label="CMMCTest")
                plt.xlabel("$\\lambda$")
                plt.ylabel("TDR")
                plt.legend()
                plt.title(f"{null_idx}_{alt_idx}")
                if save_res is True:
                    plt.savefig(f"p_values/lambda_plots/lambda_plot_TDR_{null_idx}_{alt_idx}_m09.png", dpi=500, bbox_inches="tight")
                plt.show()

                if save_res is True:
                    with open(f"p_values/lambda_plots/lambda_plot_TDR_{null_idx}_{alt_idx}_m09.txt", "w") as file:
                        file.write("\\addplot[semithick, mark=diamond, color1]\ntable{%\n")
                        for x, y in zip(lambda_arr, TDR_tot[:, 0, null_idx, alt_idx]):
                            file.write(f"{x} {y}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=diamond, color2]\ntable{%\n")
                        for x, y in zip(lambda_arr, TDR_tot[:, 1, null_idx, alt_idx]):
                            file.write(f"{x} {y}\n")
                        file.write("};\n")

        plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        plt.axhline(alpha, color="black")
        plt.plot(lambda_arr, FDR_tot_mean[:, 0, null_idx], color="tab:purple", label="MMCTest")
        plt.plot(lambda_arr, FDR_tot_mean[:, 1, null_idx], color="tab:orange", label="CMMCTest")
        plt.xlabel("$\\lambda$")
        plt.ylabel("FDR")
        plt.legend()
        plt.title(f"{null_idx}")
        if save_res is True:
            plt.savefig(f"p_values/lambda_plots/lambda_plot_FDR_{null_idx}_m09.png", dpi=500, bbox_inches="tight")
        plt.show()

        if save_res is True:
            with open(f"p_values/lambda_plots/lambda_plot_FDR_{null_idx}_m09.txt", "w") as file:
                file.write("\\addplot[semithick, mark=diamond, color1]\ntable{%\n")
                for x, y in zip(lambda_arr, FDR_tot_mean[:, 0, null_idx]):
                    file.write(f"{x} {y}\n")
                file.write("};\n")
                file.write("\\addplot[semithick, mark=diamond, color2]\ntable{%\n")
                for x, y in zip(lambda_arr, FDR_tot_mean[:, 1, null_idx]):
                    file.write(f"{x} {y}\n")
                file.write("};\n")



