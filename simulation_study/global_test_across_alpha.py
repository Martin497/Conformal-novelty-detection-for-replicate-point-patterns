#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 07:22:36 2025

@author: martin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds
from scipy.stats import chi2

from MC_CP_comparison import testing_wrapper

def Corrected_Fisher_combination(p_values, gamma):
    """
    """
    m = len(p_values)
    sum_ = np.sum(np.log(p_values))
    p = (-2*sum_ + 2*m*(np.sqrt(1+gamma) - 1))/np.sqrt(1+gamma)
    return p

def Fisher_combination(p_values):
    """
    """
    sum_ = np.sum(np.log(p_values))
    p = -2*sum_
    return p

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha_arr = np.linspace(1e-03, 0.5, 100)
    m = 10
    m0 = 5
    use_m0_est = True
    save_res = True
    testing_procedure = "Fisher_combination"

    method_list = ["MonteCarlo", "Proposed", "GET", "MonteCarlo", "Proposed"]
    technique_list = ["Fisher", "Fisher", "th", "Hochberg", "Hochberg"]
    folder_append = ["_03", "_03big", "_04", "_03", "_03big"]

    null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "LGCP"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    TDR_tot = np.zeros((len(alpha_arr), len(method_list), null_number_models, number_models), dtype=np.float64)
    FWER_tot = np.zeros((len(alpha_arr), len(method_list), null_number_models, number_models), dtype=np.float64)
    for alpha_idx, alpha in enumerate(alpha_arr):
        chisq_quantile = chi2.ppf(1-alpha, 2*m)
        for method_idx, (method, fapp, technique) in enumerate(zip(method_list, folder_append, technique_list)):
            if method == "MonteCarlo" or method == "GET":
                n = 250
            elif method == "Proposed":
                n = 2500
            gamma = m/n

            folder = f"{method}{fapp}"
            for idx1 in range(null_number_models):
                for idx2 in range(number_models):
                    false_rejection_counter = 0
                    true_rejection_counter = 0
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
                    if method == "GET":
                        u_marg = u_marg[:, 0]
                    else:
                        assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                    if method == "Proposed" or method == "MonteCarlo":
                        if technique == "Fisher":
                            rejectBool = np.zeros(data_sims, dtype=bool)
                            for sim in range(data_sims):
                                if method == "Proposed":
                                    pval = Corrected_Fisher_combination(u_marg[sim, :], gamma)
                                    if pval >= chisq_quantile:
                                        rejectBool[sim] = True
                                elif method == "MonteCarlo":
                                    pval = Fisher_combination(u_marg[sim, :])
                                    if pval >= chisq_quantile:
                                        rejectBool[sim] = True
                            if null_model_list[idx1] != test_model_list[idx2]:
                                true_rejection_counter += np.sum(rejectBool)
                            else:
                                false_rejection_counter += np.sum(rejectBool)
                        elif technique == "Hochberg":
                            rejectBool = testing_wrapper(u_marg, alpha, m0, m, data_sims,
                                                         technique, null_model_list[idx1], test_model_list[idx2],
                                                         None, use_m0_est)
                            if null_model_list[idx1] != test_model_list[idx2]:
                                true_rejection_counter += np.sum(np.any(rejectBool, axis=1))
                            else:
                                false_rejection_counter += np.sum(np.any(rejectBool, axis=1))
                    elif method == "GET":
                        rejectBool = np.zeros(data_sims, dtype=bool)
                        for sim in range(data_sims):
                            pval = u_marg[sim]
                            if pval <= alpha:
                                rejectBool[sim] = True
                        if null_model_list[idx1] != test_model_list[idx2]:
                            true_rejection_counter += np.sum(rejectBool)
                        else:
                            false_rejection_counter += np.sum(rejectBool)
                    TDR_tot[alpha_idx, method_idx, idx1, idx2] = true_rejection_counter/data_sims
                    FWER_tot[alpha_idx, method_idx, idx1, idx2] = false_rejection_counter/data_sims

    savename = "m05"

    for idx1 in range(null_number_models):
        for idx2 in range(number_models):
            if null_model_list[idx1] != test_model_list[idx2]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(alpha_arr, TDR_tot[:, 0, idx1, idx2], color="tab:orange", label="MMCTest Fisher combination")
                plt.plot(alpha_arr, TDR_tot[:, 1, idx1, idx2], color="tab:purple", label="CMMCTest Fisher combination")
                plt.plot(alpha_arr, TDR_tot[:, 2, idx1, idx2], color="tab:green", label="Multiple GET")
                plt.plot(alpha_arr, TDR_tot[:, 3, idx1, idx2], color="tab:orange", linestyle="dashed", label="MMCTest Hochberg")
                plt.plot(alpha_arr, TDR_tot[:, 4, idx1, idx2], color="tab:purple", linestyle="dashed", label="CMMCTest Hochberg")

                plt.xlabel("$\\alpha$")
                plt.ylabel("TDR")
                plt.legend()
                # plt.xscale("log")
                plt.title(f"{null_model_list[idx1]} vs {test_model_list[idx2]}")
                if save_res is True:
                    plt.savefig(f"global_plots/alpha_plot_TDR_{idx1}_{idx2}_{savename}.png", dpi=500, bbox_inches="tight")
                plt.show()

                if save_res is True:
                    with open(f"global_plots/alpha_plot_TDR_{idx1}_{idx2}_{savename}.txt", "w") as file:
                        file.write("\\addplot[semithick, mark=square, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 0, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:MMCTest_m05_globalFisher}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=square, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 1, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:CMMCTest_m05_globalFisher}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=diamond, color11, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 2, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:GET_m05_global}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=triangle, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 3, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:MMCTest_m05_globalHoch}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=triangle, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, TDR_tot[:, 4, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:CMMCTest_m05_globalHoch}\n")
                        file.write("};\n")

            if null_model_list[idx1] == test_model_list[idx2]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(alpha_arr, alpha_arr, color="k")
                plt.plot(alpha_arr, FWER_tot[:, 0, idx1, idx2], color="tab:orange", label="MMCTest Fisher combination")
                plt.plot(alpha_arr, FWER_tot[:, 1, idx1, idx2], color="tab:purple", label="CMMCTest Fisher combination")
                plt.plot(alpha_arr, FWER_tot[:, 2, idx1, idx2], color="tab:green", label="Multiple GET")
                plt.plot(alpha_arr, FWER_tot[:, 3, idx1, idx2], color="tab:orange", linestyle="dashed", label="MMCTest Hochberg")
                plt.plot(alpha_arr, FWER_tot[:, 4, idx1, idx2], color="tab:purple", linestyle="dashed", label="CMMCTest Hochberg")
                plt.xlabel("$\\alpha$")
                plt.ylabel("FWER")
                plt.legend()
                # plt.xscale("log")
                plt.title(f"{null_model_list[idx1]}")
                if save_res is True:
                    plt.savefig(f"global_plots/alpha_plot_FDR_{idx1}_{savename}.png", dpi=500, bbox_inches="tight")
                plt.show()

                if save_res is True:
                    with open(f"global_plots/alpha_plot_FWER_{idx1}_{savename}.txt", "w") as file:
                        file.write("\\addplot[semithick, mark=square, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, FWER_tot[:, 0, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:MMCTest_m05_globalFisher}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=square, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, FWER_tot[:, 1, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:CMMCTest_m05_globalFisher}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=diamond, color11, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, FWER_tot[:, 2, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:GET_m05_global}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=triangle, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, FWER_tot[:, 3, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:MMCTest_m05_globalHoch}\n")
                        file.write("};\n")
                        file.write("\\addplot[semithick, mark=triangle, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                        for x, y in zip(alpha_arr, FWER_tot[:, 4, idx1, idx2]):
                            file.write(f"{x} {y}\n")
                        # file.write("};\n\\label{plot:CMMCTest_m05_globalHoch}\n")
                        file.write("};\n")




