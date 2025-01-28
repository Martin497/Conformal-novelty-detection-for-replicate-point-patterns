#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:12:58 2024

@author: martin
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

computed = {}
def sterling1(n, k):
	key = str(n) + "," + str(k)

	if key in computed.keys():
		return computed[key]
	if n == k == 0:
		return 1
	if n > 0 and k == 0:
		return 0
	if k > n:
		return 0
	result = sterling1(n - 1, k - 1) + (n - 1) * sterling1(n - 1, k)
	computed[key] = result
	return result

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
    m0 = 5
    m1 = 5
    m = m0+m1

    Bnums0 = np.array([1])
    Bnumsm = Bernoulli(Bnums0, m+1)

    alpha = 0.05
    n_arr = np.arange(1, 2500, 4)
    FWER = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    # logpolyvec = np.zeros(m+1)
    for idxn, n in enumerate(n_arr):
        thresholds = np.floor(alpha/(m - np.arange(1, m+1, 1) + 1) * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(np.arange(1, m+1, 1)/m))  * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m+1) + 1/((n+1)/2)*np.arange(0+1, m+1, 1)/(m+1)))  * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m+1) + 1/np.sqrt(n+1)*np.arange(1, m+1, 1)/(m+1)))  * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m-np.arange(1,m+1,1)/np.sqrt(n+1))))  * (n+1))
        # thresholds = np.floor(np.arange(1, m+1, 1) * alpha/m * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        if n == 97:
            print(avecm, [(1-alpha*(n+1))*sterling1(m0, k) for k in range(0, m0+1)])
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
            # polyvec[j] = np.prod([(n+1) for k in range(j)])
        FWER[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)
        # FWER[idxn] = 1 - factorial(m)/np.prod([n+k for k in range(1, m+1, 1)]) * np.dot(avecm, polyvec)
        # logpolyvec = np.arange(0, m+1, 1)*np.log(n+1)
        # FWER[idxn] = 1 - factorial(n)*factorial(m)/factorial(n+m) * np.sum(np.exp(np.log(avecm) + logpolyvec))
        # print(n, thresholds, thresholds/(n+1), FWER[idxn])
        # print(n, FWER[idxn], polyvec)
        # print(FWER, alpha, FWER<=alpha)

    FWER_bonf = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idxn, n in enumerate(n_arr):
        # thresholds = np.floor((alpha/m * np.ones(m))  * (n+1))
        thresholds = np.floor((1 - (1 - alpha)**(1/(m0+1)) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_bonf[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    FWER_adaptive_bonf = np.zeros(len(n_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idxn, n in enumerate(n_arr):
        # thresholds = np.floor((alpha/m * np.ones(m))  * (n+1))
        thresholds = np.floor((1 - (1 - alpha)**(1/m0) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_adaptive_bonf[idxn] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(n_arr, FWER_adaptive_bonf, color="tab:red")
    plt.plot(n_arr, FWER_bonf, color="tab:blue")
    plt.plot(n_arr, FWER, color="tab:green")
    plt.axhline(alpha, color="k")
    plt.ylim(0, alpha+0.01)
    plt.xlim(n_arr[0]-1, n_arr[-1]+1)
    plt.ylabel("FWER")
    plt.xlabel("n")
    plt.savefig(f"polyFWER/numerical_nplot_alpha{alpha}.png", dpi=500, bbox_inches="tight")
    plt.show()

    with open(f"polyFWER/numerical_nplot_alpha{alpha}.txt", "w") as file:
        file.write("\\addplot[a]\ntable{%\n")
        for x, y in zip(n_arr, FWER_adaptive_bonf):
            file.write(f"{x} {y}\n")
        file.write("};\n")
        file.write("\\addplot[b]\ntable{%\n")
        for x, y in zip(n_arr, FWER_bonf):
            file.write(f"{x} {y}\n")
        file.write("};\n")
        file.write("\\addplot[c]\ntable{%\n")
        for x, y in zip(n_arr, FWER):
            file.write(f"{x} {y}\n")
        file.write("};\n")


    # print(np.max(FWER)-alpha)
    # plt.plot(FWER-FWER_bonf)
    # plt.show()

    n = 39
    alpha_arr = np.linspace(0.01, 0.5, 2000)
    FWER = np.zeros(len(alpha_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idx_alpha, alpha in enumerate(alpha_arr):
        thresholds = np.floor(alpha/(m - np.arange(1, m+1, 1) + 1) * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m+1) + 1/((n+1)/2)*np.arange(0+1, m+1, 1)/(m+1)))  * (n+1))
        # thresholds = np.floor(np.arange(1, m+1, 1) * alpha/m * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m+1) + 1/np.sqrt(n+1)*np.arange(1, m+1, 1)/(m+1)))  * (n+1))
        # thresholds = np.floor((1 - (1 - alpha)**(1/(m-np.arange(1,m+1,1)/np.sqrt(n+1))))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER[idx_alpha] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    FWER_bonf = np.zeros(len(alpha_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idx_alpha, alpha in enumerate(alpha_arr):
        # thresholds = np.floor((alpha/m * np.ones(m))  * (n+1))
        thresholds = np.floor((1 - (1 - alpha)**(1/m) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_bonf[idx_alpha] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)

    FWER_adaptive_bonf = np.zeros(len(alpha_arr))
    polyvec = np.zeros(m0+1, dtype=np.uint64)
    for idx_alpha, alpha in enumerate(alpha_arr):
        # thresholds = np.floor((alpha/m * np.ones(m))  * (n+1))
        thresholds = np.floor((1 - (1 - alpha)**(1/m0) * np.ones(m))  * (n+1))
        avec0 = np.array([1])
        avecm = polynomial_coefficients(avec0, thresholds[m1:], Bnumsm)
        for j in range(m0+1):
            polyvec[j] = (n+1)**j
        FWER_adaptive_bonf[idx_alpha] = 1 - factorial(n)*factorial(m0)/factorial(n+m0) * np.dot(avecm, polyvec)


    plt.figure(figsize=(6.4, 4.8))
    plt.plot(alpha_arr, FWER_adaptive_bonf, color="tab:red")
    plt.plot(alpha_arr, FWER_bonf, color="tab:blue")
    plt.plot(alpha_arr, FWER, color="tab:green")
    plt.plot(alpha_arr, alpha_arr, color="k")
    plt.ylim(0, alpha_arr[-1]+0.01)
    plt.xlim(alpha_arr[0], alpha_arr[-1])
    plt.ylabel("FWER")
    plt.xlabel("$\\alpha$")
    plt.show()

    # print(np.max(FWER-alpha_arr))
    # plt.plot(FWER-FWER_bonf)
    # plt.show()