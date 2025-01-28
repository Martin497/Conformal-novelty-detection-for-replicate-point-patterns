#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:54:18 2024

@author: martin
"""

import numpy as np


def ccv_correction(u_marg, n, delta):
    """
    Implementing the calibration-conditional validity correction of the
    p-values.
    """
    def b(i, n, delta):
        k = int(n/2)
        assert k == n/2, "k must be an integer."
        assert n == int(n), "n must be an integer."
        b_i = 1 - delta**(1/k)*(np.prod([(i-k+1+j)/(n-k+1+j) for j in range(0, k)]))**(1/k)
        return b_i

    arg_ = np.ceil(u_marg * (n+1))
    i_ = (n + 1 - arg_).astype(np.int32)
    u_ccv = np.zeros_like(i_, dtype=np.float64)
    for idx1, i_arr in enumerate(i_):
        for idx2, index_i in enumerate(i_arr):
            u_ccv[idx1, idx2] = b(index_i, n, delta)
    return u_ccv

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
    based on p-values, p_hat, and a test sequence, test_sequence.
    """
    test_sequence = np.zeros((data_sims, m), dtype=np.float64)
    for j in range(data_sims):
        test_sequence[j] = np.arange(1, m+1)/m0[j] * alpha
        # test_sequence[j] = np.arange(1, m+1)/m * alpha/pi0[j]
    # rejectBool = np.zeros((data_sims, m), dtype=bool)
    rejectBool = np.ones((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        # iter_ = 0
        # while iter_ < m:
        #     if p_hat_sorted_j[iter_] > test_sequence[j, iter_]:
        #         rejectBool[j, sort_[j, iter_]] = False
        #         iter_ += 1
        #         # break
        #     else:
        #         rejectBool[j, sort_[j, iter_]] = True
        #         iter_ += 1
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
    and a significance level, alpha.
    """
    test_sequence = alpha/(m - np.arange(1, m+1, 1) + 1)
    # test_sequence = np.arange(1, m+1, 1) * alpha / m
    rejectBool = np.ones((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        iter_ = m-1
        while iter_ >= 0:
            if p_hat_sorted_j[iter_] > test_sequence[iter_]:
                rejectBool[j, sort_[j, iter_]] = False
            else:
                # rejectBool[j, sort_[j, iter_]] = True
                break
            iter_ -= 1
    # test_sequence = alpha/(m - np.flip(np.arange(1, m+1, 1)) + 1)
    # rejectBool = np.ones((data_sims, m), dtype=bool)
    # sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    # for j in range(data_sims):
    #     p_hat_sorted_j = p_hat[j, sort_[j]]
    #     iter_ = m-1
    #     while iter_ >= 0:
    #         # if p_hat_sorted_j[iter_] <= test_sequence[iter_]:
    #         #     rejectBool[j, sort_[j, :iter_+1]] = True
    #         # else:
    #         #     rejectBool[j, sort_[j, iter_]] = False
    #         if p_hat_sorted_j[iter_] > test_sequence[iter_]:
    #             rejectBool[j, sort_[j, iter_]] = False
    #         else:
    #             break
    #         iter_ -= 1
    return rejectBool

def adaptive_Hochberg_procedure(p_hat, alpha, m0, m, data_sims):
    """
    """
    rejectBool = np.ones((data_sims, m), dtype=bool)
    sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    for j in range(data_sims):
        p_hat_sorted_j = p_hat[j, sort_[j]]
        if m0[j] < m:
            # step1Bool = np.where(p_hat_sorted_j <= alpha, True, False)
            # m1 = m - m0[j] # np.sum(step1Bool)
            m1 = np.sum(np.where(p_hat_sorted_j > alpha, True, False))
            # print(m1)
            step1reject = p_hat_sorted_j[m-m1-1] <= alpha/min(m0[j], m1+1)
            # print(p_hat_sorted_j[m-m1-1], m1)
            # print(p_hat_sorted_j[m-m1-1], m1, step1reject)
            while step1reject == False and m1 < m-1:
                m1 += 1
                step1reject = p_hat_sorted_j[m-m1-1] <= alpha/min(m0[j], m1+1)
                # print(p_hat_sorted_j[m-m1-1], m1, step1reject)
            rejectBool[j] = p_hat[j] <= p_hat_sorted_j[m-m1-1]
            # print(np.sum(rejectBool[j]))
            # print("")
        elif m0[j] == m:
            test_sequence = alpha/(m - np.arange(1, m+1, 1) + 1)
            iter_ = m-1
            while iter_ >= 0:
                if p_hat_sorted_j[iter_] > test_sequence[iter_]:
                    rejectBool[j, sort_[j, iter_]] = False
                else:
                    # rejectBool[j, sort_[j, iter_]] = True
                    break
                iter_ -= 1

    # test_sequence = alpha/(m - np.arange(1, m+1, 1) + 1)
    # rejectBool = np.zeros((data_sims, m), dtype=bool)
    # sort_ = np.argsort(p_hat, axis=1).astype(np.int16)
    # for j in range(data_sims):
    #     p_hat_sorted_j = p_hat[j, sort_[j]]
    #     iter_ = 0
    #     rejected = 0
    #     while iter_ < m:
    #         threshold = alpha/max(1, m0-rejected)
    #         if p_hat_sorted_j[iter_] > threshold:
    #             rejectBool[j, sort_[j, iter_]] = False
    #         else:
    #             rejectBool[j, sort_[j, iter_]] = True
    #             rejected += 1
    #         iter_ += 1
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


# def estimate_Storeys_lambda(p_hat, n, m, data_sims, B=10):
#     """
#     """
#     K_search = np.arange(1, n+1)
#     lambda_search = K_search/(n+1)
#     p_hat_bootstrap = np.random.choice(p_hat, size=(n, B, len(p_hat)))
#     FDR_hat = np.zeros((n, B), dtype=np.float64)
#     for i in range(n):
#         for j in range(B):
#             test_sequence = Storeys_correction(p_hat_bootstrap[i, j], alpha, n, m, data_sims)
#             rejectBool = Benjamini_Hochberg_procedure(p_hat_bootstrap[i, j], test_sequence, m, data_sims)
#             # FDR, FDP = compute_FDR(rejectBool, null_model_list[idx1], test_model_list[idx2], m0, data_sims)
#             # FDR_tot[idx1, idx2] = FDR
#             FDR_hat[i, j] = None

def Storeys_correction(lambda_, p_hat, alpha, m, data_sims, null_model, test_model):
    """
    Estimating the proportion of true nulls, pi0, using Storey's estimate.
    Then make the test sequence to be used for the Benjamini-Hochberg procedure.
    """
    pi0_hat = np.zeros(data_sims, dtype=np.float64)
    m0_hat = np.zeros(data_sims, dtype=np.int16)
    for j in range(data_sims):
        # pi0_hat[j] = (1 + np.sum(p_hat[j] > lambda_)) / (m * (1 - lambda_))
        pi0_hat[j] = min(1, (1 + np.sum(p_hat[j] > lambda_)) / (m * (1 - lambda_)))
        # m0_hat[j] = np.ceil(pi0_hat[j] * m)
        m0_hat[j] = min(m, np.ceil(pi0_hat[j] * m))
        # pi0_hat[j] = m0_hat[j]/m
    return pi0_hat, m0_hat

def estimate_m0():
    pass
