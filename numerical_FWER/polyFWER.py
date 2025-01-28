#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:12:58 2024

@author: Martin Voigt Vejling
Emails: mvv@math.aau.dk
        mvv@es.aau.dk
        martin.vejlig@gmail.com

Script to numerically evaluate the family-wise error rate for a given sequence
of thresholds. The implementation is based on Lemma 3 in the supplementary material.

Reproduces Figure 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def binom(n, k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def Bernoulli(Bnums, max_):
    m = len(Bnums)
    if m == max_:
        return Bnums
    else:
        Bnumsnew = np.zeros(m+1)
        Bnumsnew[:m] = Bnums
        Bnumsnew[-1] = 1
        for k in range(m):
            Bnumsnew[-1] -= binom(m, k)*Bnums[k]/(m-k+1)
        return Bernoulli(Bnumsnew, max_)

def polynomial_coefficients(avec, thresholds, Bnumsm):
    m = len(avec)
    if m == len(thresholds)+1:
        return avec
    else:
        avecnew = np.zeros(m+1)
        for j in range(m):
            for r in range(j+1):
                avecnew[0] -= avec[j]/(j+1)*binom(j+1, r)*Bnumsm[r]*thresholds[m-1]**(j+1-r)
        for i in range(1, m+1):
            for j in range(i-1, m):
                avecnew[i] += avec[j]*factorial(j)/(factorial(j+1-i)*factorial(i)) * Bnumsm[j+1-i]
        return polynomial_coefficients(avecnew, thresholds, Bnumsm)

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    m0 = 5
    m1 = 5
    m = m0+m1

    Bnums0 = np.array([1])
    Bnumsm = Bernoulli(Bnums0, m+1)

    alpha = 0.05
    n_arr = np.arange(1, 2500, 4)

    # =============================================================================
    # Hochberg
    # =============================================================================
    FWER_Hoch = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idxn, n in enumerate(n_arr):
        thresholds = np.floor(alpha/(m - np.arange(1, m+1, 1) + 1) * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_Hoch[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    # =============================================================================
    # Bonferroni \hat{m}_0 = m_0
    # =============================================================================
    FWER_bonf_m0 = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idxn, n in enumerate(n_arr):
        thresholds = np.floor((1 - (1 - alpha)**(1/(m0+1)) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_bonf_m0[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    # =============================================================================
    # Bonferroni \hat{m}_0 = m_0 + 1
    # =============================================================================
    FWER_bonf_m01 = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idxn, n in enumerate(n_arr):
        thresholds = np.floor((1 - (1 - alpha)**(1/m0) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_bonf_m01[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    plt.figure(figsize=(6.4*1.5, 4.8*1.2))
    plt.plot(n_arr, FWER_bonf_m01, color="tab:red", label="$\\text{Bonferroni}_{\hat{m}_0=m_0+1}$")
    plt.plot(n_arr, FWER_bonf_m0, color="tab:blue", label="$\\text{Bonferroni}_{\hat{m}_0=m_0}$")
    plt.plot(n_arr, FWER_Hoch, color="tab:green", label="Hochberg")
    plt.legend()
    plt.axhline(alpha, color="k")
    plt.ylim(0, alpha+0.01)
    plt.xlim(n_arr[0]-1, n_arr[-1]+1)
    plt.ylabel("FWER")
    plt.xlabel("n")
    plt.savefig(f"numerical_nplot_alpha{alpha}.png", dpi=500, bbox_inches="tight")
    plt.show()
