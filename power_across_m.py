#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:33:51 2024

@author: martin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds
from scipy.interpolate import LinearNDInterpolator

import multiple_testing_module as mtm
from MC_CP_comparison import testing_wrapper


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    # alpha = 0.03
    alpha = 0.2
    # m = 10
    # m0 = 5
    lambda_ = 0.5
    use_m0_est = True

    # null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # null_model_list = ["Strauss_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    # test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    null_model_list = ["Poisson_Mrkvicka"]
    test_model_list = ["Strauss_Mrkvicka"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    # =============================================================================
    # Conformal
    # =============================================================================
    method_list = ["Proposed"]
    folder_append = ["_10"]

    n_list = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    m_list = [6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    TDR_tot = np.zeros((len(method_list), len(n_list), len(m_list), null_number_models, number_models), dtype=np.float64)
    FDR_tot = np.zeros((len(method_list), len(n_list), len(m_list), null_number_models, number_models), dtype=np.float64)
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
                data_sims = u_marg.shape[0]
                lenN = u_marg.shape[1]
                # assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                for n_idx in range(len(n_list)):
                    for m_idx in range(len(m_list)):
                        u_marg_ = np.concatenate((u_marg[:, n_idx, :int(m_list[m_idx]/2)], u_marg[:, n_idx, -int(m_list[m_idx]/2):]), axis=-1)
                        rejectBool = testing_wrapper(u_marg_, alpha, int(m_list[m_idx]/2), m_list[m_idx], data_sims,
                                                     "BH", null_model_list[idx1], test_model_list[idx2],
                                                     lambda_, use_m0_est)

                        TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_list[m_idx]/2), m_list[m_idx], data_sims)
                        TDR_tot[method_idx, n_idx, m_idx, idx1, idx2] = TDR

                        FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_list[m_idx]/2), data_sims)
                        FDR_tot[method_idx, n_idx, m_idx, idx1, idx2] = FDR

    FDR_tot_mean = np.mean(FDR_tot, axis=-1)

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                for n_idx in range(len(n_list)):
                    plt.plot(m_list, TDR_tot[0, n_idx, :, null_idx, alt_idx], label=f"n{n_list[n_idx]}")
                plt.xlabel("m")
                plt.ylabel("TDR")
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.legend()
                plt.show()

                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                for m_idx in range(len(m_list)):
                    plt.plot(n_list, TDR_tot[0, :, m_idx, null_idx, alt_idx], label=f"m{m_list[m_idx]}")
                plt.xlabel("n")
                plt.ylabel("TDR")
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.legend()
                plt.show()

                spec = plt.pcolormesh(m_list, n_list, TDR_tot[0, :, :, null_idx, alt_idx], cmap="cool", shading="gouraud")
                cb = plt.colorbar(spec)
                cb.set_label(label="TDR")
                plt.xlabel("m")
                plt.ylabel("n")
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.show()

        plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        for n_idx in range(len(n_list)):
            plt.plot(m_list, FDR_tot_mean[0, n_idx, :, null_idx], label=f"n{n_list[n_idx]}")
        plt.xlabel("m")
        plt.ylabel("FDR")
        plt.legend()
        plt.title(f"{null_model_list[null_idx]}")
        plt.show()

        spec = plt.pcolormesh(m_list, n_list, FDR_tot_mean[0, :, :, null_idx], cmap="cool", shading="gouraud")
        cb = plt.colorbar(spec)
        cb.set_label(label="FDR")
        plt.xlabel("m")
        plt.ylabel("n")
        plt.title(f"{null_model_list[null_idx]}")
        plt.show()

    # =============================================================================
    # Monte Carlo
    # =============================================================================

    MCfolder_append = ["_26", "_23", "_24", "_25"]
    MCmethod_list = ["MonteCarlo"]*len(MCfolder_append)
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
                data_sims = u_marg.shape[0]
                lenN = u_marg.shape[1]
                # assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                for n_idx in range(len(n_list_iter)):
                    u_marg_ = u_marg[:, n_idx, :]
                    rejectBool = testing_wrapper(u_marg_, alpha, int(m_iter/2), m_iter, data_sims,
                                                 "BH", null_model_list[idx1], test_model_list[idx2],
                                                 lambda_, use_m0_est)

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_iter/2), m_iter, data_sims)
                    TDR_MCtot[method_idx][n_idx, idx1, idx2] = TDR

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], int(m_iter/2), data_sims)
                    FDR_MCtot[method_idx][n_idx, idx1, idx2] = FDR

    FDR_MCtot_mean = [np.mean(a, axis=-1) for a in FDR_MCtot]
    nGrid, mGrid = np.meshgrid(np.linspace(300, 2500, 100, endpoint=True), np.linspace(6, 30, 50, endpoint=True))
    MCn_arr = np.concatenate(MCn_lists)
    MCm_arr = np.array([MCm_list[0]]*len(MCn_lists[0]) + [MCm_list[1]]*len(MCn_lists[1]) + [MCm_list[2]]*len(MCn_lists[2]) + [MCm_list[3]]*len(MCn_lists[3]))
    xy = np.stack((MCn_arr, MCm_arr), axis=-1)
    TDR_MCtot_arr = np.concatenate(TDR_MCtot)

    for null_idx in range(null_number_models):
        for alt_idx in range(number_models):
            if null_model_list[null_idx] != test_model_list[alt_idx]:
                plt.figure(figsize=(6.4*1.5, 4.8*1.2))
                for i in range(len(TDR_MCtot)):
                    plt.plot(MCn_lists[i], TDR_MCtot[i][:, null_idx, alt_idx], label=f"m{MCm_list[i]}")
                plt.xlabel("n")
                plt.ylabel("TDR")
                plt.title(f"{null_model_list[null_idx]}_{test_model_list[alt_idx]}")
                plt.legend()
                plt.show()

                z = TDR_MCtot_arr[:, null_idx, alt_idx]
                interp = LinearNDInterpolator(xy, z)
                Z = interp(nGrid, mGrid).T
                spec = plt.pcolormesh(np.linspace(6, 30, 50, endpoint=True), np.linspace(300, 2500, 100, endpoint=True),
                                      Z, cmap="cool", shading="nearest")
                cb = plt.colorbar(spec)
                cb.set_label(label="TDR")
                plt.xlabel("m")
                plt.ylabel("n")
                plt.title(f"{null_model_list[null_idx]}")
                plt.show()

        # plt.figure(figsize=(6.4*1.5, 4.8*1.2))
        # for n_idx in range(len(n_list)):
        #     plt.plot(m_list, FDR_tot_mean[0, n_idx, :, null_idx], label=f"n{n_list[n_idx]}")
        # plt.xlabel("m")
        # plt.ylabel("FDR")
        # plt.legend()
        # plt.title(f"{null_model_list[null_idx]}")
        # plt.show()

        # spec = plt.pcolormesh(m_list, n_list, FDR_tot_mean[0, :, :, null_idx], cmap="cool", shading="gouraud")
        # cb = plt.colorbar(spec)
        # cb.set_label(label="FDR")
        # plt.xlabel("m")
        # plt.ylabel("n")
        # plt.title(f"{null_model_list[null_idx]}")
        # plt.show()

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

            with open(f"knm_phase/TDR_{null_idx}_{alt_idx}.txt", "w") as file:
                for i in range(len(TDR_MCtot)):
                    if i == 0:
                        file.write("\\addplot[semithick, mark=square, solid, mark options=solid, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif i == 1:
                        file.write("\\addplot[semithick, mark=triangle, solid, mark options=solid, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif i == 2:
                        file.write("\\addplot[semithick, mark=diamond, solid, mark options=solid, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif i == 3:
                        file.write("\\addplot[semithick, mark=pentagon, solid, mark options=solid, color10, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    for x, y in zip(MCn_lists[i], TDR_MCtot[i][:, null_idx, alt_idx]):
                        file.write(f"{x} {y}\n")
                    file.write("};\n")
                for m_idx in range(len(TDR_MCtot)):
                    if m_idx == 0:
                        file.write("\\addplot[semithick, mark=square, solid, mark options=solid, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif m_idx == 1:
                        file.write("\\addplot[semithick, mark=triangle, solid, mark options=solid, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif m_idx == 2:
                        file.write("\\addplot[semithick, mark=diamond, solid, mark options=solid, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    elif m_idx == 3:
                        file.write("\\addplot[semithick, mark=pentagon, solid, mark options=solid, color9, mark repeat=\\markrep, mark phase=\\markphase, line width=1pt]\ntable{%\n")
                    for x, y in zip(n_list, TDR_tot[0, :, m_idx, null_idx, alt_idx]):
                        file.write(f"{x} {y}\n")
                    file.write("};\n")