#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:28:38 2024

@author: martin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds

import multiple_testing_module as mtm
from MC_CP_comparison import testing_wrapper


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    alpha = 0.1
    m = 10
    m0 = 5
    lambda_ = 0.5
    use_m0_est = True

    method_list = ["Proposed"]
    folder_append = ["_10"]

    # null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # null_model_list = ["Strauss_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    null_model_list = ["Poisson_Mrkvicka"]
    test_model_list = ["Strauss_Mrkvicka"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    n_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]

    TDR_tot = np.zeros((len(method_list), len(n_list), null_number_models, number_models), dtype=np.float64)
    FDR_tot = np.zeros((len(method_list), len(n_list), null_number_models, number_models), dtype=np.float64)
    for method_idx, (method, fapp) in enumerate(zip(method_list, folder_append)):
        folder = f"{method}{fapp}"
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
                u_marg = np.concatenate((u_marg[:, :, :m0], u_marg[:, :, -(m-m0):]), axis=-1)

                data_sims = u_marg.shape[0]
                lenN = u_marg.shape[1]
                assert m == u_marg.shape[2], "The number of test points, m, is mis-specified."

                for nidx in range(lenN):
                    rejectBool = testing_wrapper(u_marg[:, nidx, :], alpha, m0, m, data_sims,
                                                 "BH", null_model_list[idx1], test_model_list[idx2],
                                                 lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, m, data_sims)
                    TDR_tot[method_idx, nidx, idx1, idx2] = TDR

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FDR_tot[method_idx, nidx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                plt.plot(n_list, TDR_tot[0, :, null_idx, alt_idx], color="tab:orange")
                # plt.plot(n_list, TDR_tot[1, :, null_idx, alt_idx], color="tab:purple")
                plt.xlabel("n")
                plt.ylabel("TDR")
                # plt.legend()
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.show()
                # plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                # plt.plot([2500, 3500], TDR_tot[:, null_idx, alt_idx], color="tab:orange")
                # plt.xlabel("n")
                # plt.ylabel("TDR")
                # # plt.legend()
                # plt.title(f"{null_idx}_{alt_idx}")
                # plt.show()

        plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        plt.plot(n_list, FDR_tot_mean[0, :, null_idx], color="tab:orange")
        # plt.plot(n_list, FDR_tot_mean[1, :, null_idx], color="tab:purple")
        plt.xlabel("n")
        plt.ylabel("FDR")
        # plt.legend()
        plt.title(f"{null_model_list[null_idx]}")
        plt.show()

        # plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        # plt.axhline(alpha, color="black")
        # plt.plot([2500, 3500], FDR_tot_mean[:, null_idx], color="tab:orange")
        # plt.xlabel("n")
        # plt.ylabel("FDR")
        # # plt.legend()
        # plt.title(f"{null_model_list[null_idx]}")
        # plt.show()




