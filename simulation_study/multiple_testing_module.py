#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:54:18 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejling@gmail.com

Module for basic implementation of multiple testing procedures (Benjamini-Hochberg,
Hochberg, etc...), as well as evaluating the performance metrics (FDR, FWER,
TDR, etc...).
"""

import numpy as np

def testing_wrapper(u_marg, alpha, m0, m, data_sims, control, null_model, test_model, lambda_, use_m0_est):
    """
    """
    if use_m0_est is True:
        pi0_hat, m0_hat = Storeys_correction(lambda_, u_marg, alpha, m, data_sims, null_model, test_model)
    else:
        m0_hat = np.ones(data_sims, dtype=np.int32)*m0
    m_arr = np.ones(data_sims, dtype=np.int32)*m

    if control == "BH":
        if null_model == test_model:
            rejectBool = Benjamini_Hochberg_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = Benjamini_Hochberg_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "Hochberg":
        rejectBool = Hochberg_procedure(u_marg, alpha, m, data_sims)
    elif control == "Bonferroni":
        rejectBool = Bonferroni_procedure(u_marg, alpha, m, data_sims)
    elif control == "adaptive_Bonferroni":
        if null_model == test_model:
            rejectBool = adaptive_Bonferroni_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = adaptive_Bonferroni_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "adaptive_Holm":
        if null_model == test_model:
            rejectBool = adaptive_Holm_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = adaptive_Holm_procedure(u_marg, alpha, m0_hat, m, data_sims)
    elif control == "Holm":
        rejectBool = Holm_procedure(u_marg, alpha, m, data_sims)
    elif control == "SharpBonferroni":
        rejectBool = SharpBonferroni_procedure(u_marg, alpha, m, data_sims)
    elif control == "adaptive_SharpBonferroni":
        if null_model == test_model:
            rejectBool = adaptive_SharpBonferroni_procedure(u_marg, alpha, m_arr, m, data_sims)
        else:
            rejectBool = adaptive_SharpBonferroni_procedure(u_marg, alpha, m0_hat, m, data_sims)
    return rejectBool

def compute_FDP(rejectBool_, null_model, test_model, m0, data_sims):
    """
    Computing the false discovery proportion.
    """
    if null_model == test_model:
        R = np.sum(rejectBool_)
        RcapH = np.sum(rejectBool_)
        if R == 0:
            FDP = 0
        else:
            FDP = RcapH/R
    else:
        R = np.sum(rejectBool_)
        RcapH = np.sum(rejectBool_[:m0])
        if R == 0:
            FDP = 0
        else:
            FDP = RcapH/R
    return FDP

def compute_FDR(rejectBool, null_model, test_model, m0, data_sims):
    """
    Monte Carlo estimate of the false discovery rate, as well as an array of
    false discovery proportions.
    """
    FDP = np.zeros(data_sims, dtype=np.float64)
    for j in range(data_sims):
        FDP[j] = compute_FDP(rejectBool[j], null_model, test_model, m0, data_sims)
    FDR = np.mean(FDP)
    return FDR, FDP

def compute_FWER(rejectBool, null_model, test_model, m0, data_sims):
    """
    Monte Carlo estimate of the familywise error rate.
    """
    FWER = 0
    for j in range(data_sims):
        if null_model == test_model:
            RcapH = np.sum(rejectBool[j])
            if RcapH >= 1:
                FWER += 1
        else:
            RcapH = np.sum(rejectBool[j, :m0])
            if RcapH >= 1:
                FWER += 1
    FWER /= data_sims
    return FWER

def compute_TDP(rejectBool_, null_model, test_model, m0, m, data_sims):
    """
    Computing the true discovery proportion.
    """
    if null_model == test_model:
        TDP = 0
    else:
        m1P = m-m0
        RcapHc = np.sum(rejectBool_[m0:])
        if m == m0:
            TDP = 0
        else:
            TDP = RcapHc/m1P
    return TDP

def compute_TDR(rejectBool, null_model, test_model, m0, m, data_sims):
    """
    Monte Carlo estimate of the true discovery rate, as well as an array of
    true discovery proportions.
    """
    TDP = np.zeros(data_sims)
    for j in range(data_sims):
        TDP[j] = compute_TDP(rejectBool[j], null_model, test_model, m0, m, data_sims)
    TDR = np.mean(TDP)
    return TDR, TDP

def Bonferroni_procedure(p_hat, alpha, m, data_sims):
    """
    The Bonferroni procedure.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    rejectBool[p_hat <= alpha/m] = True
    return rejectBool

def SharpBonferroni_procedure(p_hat, alpha, m, data_sims):
    """
    The Bonferroni procedure.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    rejectBool[p_hat <= 1 - (1-alpha)**(1/m)] = True
    return rejectBool

def adaptive_SharpBonferroni_procedure(p_hat, alpha, m0, m, data_sims):
    """
    The Bonferroni procedure correcting for m0.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    rejectBool[p_hat <= 1 - (1-alpha)**(1/m0[:, None])] = True
    return rejectBool

def adaptive_Bonferroni_procedure(p_hat, alpha, m0, m, data_sims):
    """
    The Bonferroni procedure correcting for m0.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    rejectBool[p_hat <= alpha/m0[:, None]] = True
    return rejectBool

def Benjamini_Hochberg_procedure(p_hat, alpha, m0, m, data_sims):
    """
    Basic functionality to make a boolean array for rejection/not rejection
    based on p-values, p_hat, significance level, alpha, the number of tests, m,
    and an estimate of the number of true nulls, m0.
    """
    test_sequence = np.zeros((data_sims, m), dtype=np.float64)
    for j in range(data_sims):
        test_sequence[j] = np.arange(1, m+1)/m0[j] * alpha
    rejectBool = np.ones((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        iter_ = m-1
        while iter_ >= 0:
            if p_hat_sorted_j[iter_] <= test_sequence[j, iter_]:
                rejectBool[j, sort_[j, iter_]] = True
                iter_ -= 1
                break
            else:
                rejectBool[j, sort_[j, iter_]] = False
                iter_ -= 1
    return rejectBool

def Hochberg_procedure(p_hat, alpha, m, data_sims):
    """
    The Hochberg procedure, also known as Simes' procedure, given p-values, p_hat,
    a significance level, alpha, and the number of tests, m.
    """
    test_sequence = alpha/(m - np.arange(1, m+1, 1) + 1)
    rejectBool = np.ones((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        iter_ = m-1
        while iter_ >= 0:
            if p_hat_sorted_j[iter_] > test_sequence[iter_]:
                rejectBool[j, sort_[j, iter_]] = False
            else:
                break
            iter_ -= 1
    return rejectBool

def Holm_procedure(p_hat, alpha, m, data_sims):
    """
    Holm procedure.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        p_hat_sorted_j_Bool = np.where(p_hat_sorted_j <= alpha/(m - np.arange(1, m+1, 1) + 1), True, False)
        for i in range(1, m+1):
            if np.all(p_hat_sorted_j_Bool[:i]) == True:
                rejectBool[j, sort_[i-1]] = True
    return rejectBool

def adaptive_Holm_procedure(p_hat, alpha, m0, m, data_sims):
    """
    Adaptive Holm's procedure given as estimate of m0.
    """
    rejectBool = np.zeros((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        p_hat_j = p_hat[j]
        if m0[j] < m:
            step1Bool = np.where(p_hat_j <= alpha/m0[j], True, False)
            m1 = m-np.sum(step1Bool)
            while m1 < m0[j]:
                if m1 == 0:
                    step1Bool = np.zeros(m, dtype=bool)
                    break
                else:
                    step1Bool = np.where(p_hat_j <= alpha/m1, True, False)
                m2 = m-np.sum(step1Bool)
                if m2 == m1:
                    break
                else:
                    m1 = m2
            rejectBool[j] = step1Bool
        else:
            p_hat_sorted_j_Bool = np.where(p_hat_sorted_j <= alpha/(m - np.arange(1, m+1, 1) + 1), True, False)
            for i in range(1, m+1):
                if np.all(p_hat_sorted_j_Bool[:i]) == True:
                    rejectBool[j, i] = True
    return rejectBool

def Storeys_correction(lambda_, p_hat, alpha, m, data_sims, null_model, test_model):
    """
    Estimating the proportion of true nulls, pi0, using Storey's estimate.
    Then make the test sequence to be used for the Benjamini-Hochberg procedure.
    """
    pi0_hat = np.zeros(data_sims, dtype=np.float64)
    m0_hat = np.zeros(data_sims, dtype=np.int16)
    for j in range(data_sims):
        pi0_hat[j] = min(1, (1 + np.sum(p_hat[j] > lambda_)) / (m * (1 - lambda_)))
        m0_hat[j] = min(m, np.ceil(pi0_hat[j] * m))
    return pi0_hat, m0_hat
