#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:56:56 2024

@author: martin
"""

import pandas as pd
import numpy as np
from rds2py import read_rds

import multiple_testing_module as mtm

def testing_wrapper(u_marg, alpha, m0, m, data_sims, control, null_model, test_model, lambda_, use_m0_est):
    """
    """
    if use_m0_est is True:
        pi0_hat, m0_hat = mtm.Storeys_correction(lambda_, u_marg, alpha, m, data_sims, null_model, test_model)
    else:
        m0_hat = np.ones(data_sims, dtype=np.int32)*m0
        pi0_hat = np.ones(data_sims, dtype=np.int32)*m0/m
    m_arr = np.ones(data_sims, dtype=np.int32)*m
    pi_arr = np.ones(data_sims, dtype=np.int32)*m0/m

    if control == "BH":
        if null_model == test_model:
            rejectBool = mtm.Benjamini_Hochberg_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = mtm.Benjamini_Hochberg_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "Hochberg":
        rejectBool = mtm.Hochberg_procedure(u_marg, alpha, m, data_sims)
    elif control == "Bonferroni":
        rejectBool = mtm.Bonferroni_procedure(u_marg, alpha, m, data_sims)
    elif control == "adaptive_Bonferroni":
        if null_model == test_model:
            rejectBool = mtm.adaptive_Bonferroni_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = mtm.adaptive_Bonferroni_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "adaptive_Hochberg":
        if null_model == test_model:
            rejectBool = mtm.adaptive_Hochberg_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = mtm.adaptive_Hochberg_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "adaptive_Holm":
        if null_model == test_model:
            rejectBool = mtm.adaptive_Holm_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = mtm.adaptive_Holm_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "Holm":
        rejectBool = mtm.Holm_procedure(u_marg, alpha, m, data_sims)
    elif control == "SharpBonferroni":
        rejectBool = mtm.SharpBonferroni_procedure(u_marg, alpha, m, data_sims)
    elif control == "adaptive_SharpBonferroni":
        if null_model == test_model:
            rejectBool = mtm.adaptive_SharpBonferroni_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = mtm.adaptive_SharpBonferroni_procedure(u_marg, alpha, m0_hat, m, data_sims)
    return rejectBool


if __name__ == "__main__":
    alpha = 0.05
    m = 10
    m0 = 5
    lambda_ = 0.25
    use_m0_est = True


    control_list = ["Hochberg", "BH"]
    # table_list = ["MMCTest Bonf.", "MMCTest BH", "CMMCTest Bonf.", "CMMCTest BH"]
    table_list = ["MMCTest Hoch.", "MMCTest BH", "CMMCTest Hoch.", "CMMCTest BH"]
    # table_list2 = ["adaptive_Bonferroni_MonteCarlo", "BH_MonteCarlo", "adaptive_Bonferroni_Proposed", "BH_Proposed"]
    table_list2 = ["Hochberg_MonteCarlo", "BH_MonteCarlo", "Hochberg_Proposed", "BH_Proposed"]
    method_list = ["MonteCarlo", "Proposed"]
    folder_append = ["_03", "_03big"]

    null_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    test_model_list = ["Strauss_Mrkvicka", "Poisson_Mrkvicka", "MatClust_Mrkvicka", "LGCP"]
    null_number_models = len(null_model_list)
    number_models = len(test_model_list)

    res_dict = dict()

    for control in control_list:
        for method, fapp in zip(method_list, folder_append):
            folder = f"{method}{fapp}"
            FDR_tot = np.zeros((null_number_models, number_models), dtype=np.float64)*np.nan
            FWER_tot = np.zeros((null_number_models, number_models), dtype=np.float64)*np.nan
            TDR_tot = np.zeros((null_number_models, number_models), dtype=np.float64)
            TDR_std_tot = np.zeros((null_number_models, number_models), dtype=np.float64)
            for idx1 in range(null_number_models):
                for idx2 in range(number_models):
                    name = f"{null_model_list[idx1]}_{test_model_list[idx2]}"

                    try:
                        u_marg = pd.read_csv(f"{folder}/{name}.csv", index_col=(0)).to_numpy()
                    except FileNotFoundError:
                        res = read_rds(f"{folder}/{name}.rds")
                        u_marg = res["data"].reshape(res["attributes"]["dim"]["data"], order="F")
                    if ("Estimate" in fapp) or ("Thinning" in fapp):
                        u_marg = u_marg[0]

                    data_sims = u_marg.shape[0]
                    assert m == u_marg.shape[1], "The number of test points, m, is mis-specified."

                    rejectBool = testing_wrapper(u_marg, alpha, m0, m, data_sims,
                                                 control, null_model_list[idx1], test_model_list[idx2],
                                                 lambda_, use_m0_est)

                    FDR, FDP = mtm.compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FDR_tot[idx1, idx2] = FDR

                    FWER = mtm.compute_FWER(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
                    FWER_tot[idx1, idx2] = FWER

                    TDR, TDP = mtm.compute_TDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, m, data_sims)
                    TDR_tot[idx1, idx2] = TDR
                    TDR_std_tot[idx1, idx2] = np.std(TDP)

            print(control, method, FWER_tot)
            print(control, method, FDR_tot)
            FWER_tot_mean = np.nanmean(FWER_tot, axis=1)
            FDR_tot_mean = np.nanmean(FDR_tot, axis=1)

            res_dict[f"{control}_{method}"] = dict()
            res_dict[f"{control}_{method}"]["FDR_tot_mean"] = FDR_tot_mean
            res_dict[f"{control}_{method}"]["FWER_tot_mean"] = FWER_tot_mean
            res_dict[f"{control}_{method}"]["TDR_tot"] = TDR_tot
            res_dict[f"{control}_{method}"]["TDR_std_tot"] = TDR_std_tot

    dict_listi = {"0": ["${\\rm Poisson}(200)$", "${\\rm MatClust}(200, 0.06, 1)$", "${\\rm LGCP}(5, 0.6, 0.05)$"],
                  "1": ["${\\rm Strauss}(250, 0.6, 0.03)$", "${\\rm MatClust}(200, 0.06, 1)$", "${\\rm LGCP}(5, 0.6, 0.05)$"],
                  "2": ["${\\rm Strauss}(250, 0.6, 0.03)$", "${\\rm Poisson}(200)$", "${\\rm LGCP}(5, 0.6, 0.05)$"],
                  "3": ["${\\rm Strauss}(250, 0.6, 0.03)$", "${\\rm Poisson}(200)$", "${\\rm MatClust}(200, 0.06, 1)$"]}
    listj = ["${\\rm Strauss}(250, 0.6, 0.03)$", "${\\rm Poisson}(200)$",
             "${\\rm MatClust}(200, 0.06, 1)$", "${\\rm LGCP}(5, 0.6, 0.05)$"]

    print("Power table\n------------")
    table_lines = ""
    table_lines += "Non-true null distribution      & Null distribution                 "
    for name in table_list:
        table_lines += f" & {name:14}"
    table_lines += " \\\ \\midrule \n"
    for j in range(4):
        for i in range(3):
            if j == 0:
                idx = [1, 2, 3]
            elif j == 1:
                idx = [0, 2, 3]
            elif j == 2:
                idx = [0, 1, 3]
            elif j == 3:
                idx = [0, 1, 2]
            table_lines += f"{dict_listi[str(j)][i]:31} & {listj[(3*j+i)//3]:33}"
            for key in table_list2:
                table_lines += f"  & {res_dict[key]['TDR_tot'][j,idx[i]]:.3f} ({res_dict[key]['TDR_std_tot'][j,idx[i]]:.3f})"
            table_lines += " \\\ \n"
    print(table_lines)

    # print("\n")
    # print("Error table\n------------")
    # table_lines = ""
    # table_lines += "Null hypothesis                "
    # for name in table_list:
    #     table_lines += f" & {name:17}"
    # table_lines += " \\\ \\midrule \n"
    # for j in range(3):
    #     table_lines += f"{listj[(2*j+i)//2]:32}"
    #     for key in table_list2:
    #         table_lines += f"& {res_dict[key]['FDR_tot_mean'][j]:.3f} $|$ {res_dict[key]['FWER_tot_mean'][j]:.3f}   "
    #     table_lines += " \\\ \n"
    # print(table_lines)

    print("\n")
    print("Error table\n------------")
    table_lines = ""
    table_lines += "Null hypothesis "
    for name in table_list:
        table_lines += " & \\multicolumn{2}{c|}{"+f"{name}"+"}"
    table_lines += " \\\ \n"
    table_lines += "& FDR & FWER & FDR & FWER & FDR & FWER & FDR & FWER \\\ \\midrule \n"
    for j in range(4):
        table_lines += f"{listj[(3*j+i)//3]:32}"
        for key in table_list2:
            table_lines += f"& {res_dict[key]['FDR_tot_mean'][j]:.3f} & {res_dict[key]['FWER_tot_mean'][j]:.3f}   "
        table_lines += " \\\ \n"
    print(table_lines)

