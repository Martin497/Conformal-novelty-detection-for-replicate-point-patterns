#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:17:52 2024

@author: martin
"""


import numpy as np
import matplotlib.pyplot as plt
from rds2py import read_rds
# import tikzplotlib

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")

    path = "p_values/smwn_diagnosed_n5000"

    cr_data_rds = read_rds(f"{path}_data.rds")
    cr_data = cr_data_rds["data"].reshape(cr_data_rds["attributes"]["dim"]["data"], order="F")
    test_sequence = read_rds(f"{path}_test_sequence.rds")["data"]
    sorting_order = read_rds(f"{path}_sorting_order.rds")["data"]
    cutoff = 1000
    r = cr_data[0, :cutoff, 0]
    cr_data = cr_data[:, :cutoff, :]

    control_ids = [96, 149, 203, 205]
    MNA_diagnosed_ids = [23, 36, 42, 50, 73]
    MNA_suspected_ids = [10, 20, 40, 61, 71]
    # test_ids = MNA_suspected_ids + MNA_diagnosed_ids
    test_ids = MNA_suspected_ids + control_ids

    # np.save("cr_data.npy", cr_data)
    # np.save("sorting_order.npy", sorting_order)
    # np.save("test_sequence.npy", test_sequence)
    # test_sequence = np.load("test_sequence.npy")
    # sorting_order = np.load("sorting_order.npy")
    # cr_data = np.load("cr_data.npy")
    # r = cr_data[0, :, 0]

    fig, axs = plt.subplots(2, 5, figsize=(6.4*1.5, 4.8*1.2))
    for i in range(10):

        a, b = divmod(i, 5)
        if i == 9:
            axs[a,b].set_xlabel("Interpoint distance, r")
            axs[a,b].plot(r, cr_data[0, :, 1], linewidth=0)
            axs[a,b].set_ylim(-76, 28)
        else:
            outsideBool = np.where(np.logical_and(cr_data[i, :, 4] <= cr_data[i, :, 3], cr_data[i, :, 4] >= cr_data[i, :, 2]), 0, 1).astype(bool)
            axs[a,b].plot(r, cr_data[i, :, 1], color="k", linestyle="dashed")
            axs[a,b].plot(r, cr_data[i, :, 2], color="gray")
            axs[a,b].plot(r, cr_data[i, :, 3], color="gray")
            axs[a,b].fill_between(r, cr_data[i, :, 2], cr_data[i, :, 3], facecolor="gray", alpha=0.5)
            axs[a,b].plot(r, cr_data[i, :, 4], marker="o", markersize=2, linewidth=0, color="tab:green")
            axs[a,b].plot(r[outsideBool], cr_data[i, outsideBool, 4], marker="o", markersize=2, linewidth=0, color="tab:red")
            axs[a,b].set_ylim(-76, 28)
            axs[a,b].set_xlabel("Interpoint distance, r")
            axs[a,b].set_ylabel("Summary statistic, L(r) - r")
            axs[a,b].set_title(f"({test_ids[sorting_order[i]-1]}) {(1-test_sequence[i])*100:.2f} % cove.")

        # axs[i].plot(r, cr_data[i, :, 1], color="k")
        # axs[i].plot(r, cr_data[i, :, 2], color="gray")
        # axs[i].plot(r, cr_data[i, :, 3], color="gray")
        # axs[i].fill_between(r, cr_data[i, :, 2], cr_data[i, :, 3], facecolor="gray", alpha=0.5)
        # axs[i].plot(r, cr_data[i, :, 4], color="tab:green")
        # axs[i].plot(r[outsideBool], cr_data[i, outsideBool, 4], color="tab:red")
        # axs[i].set_ylim(-0.015, 0.015)
        # axs[i].set_xlabel("Interpoint distance, r")
        # axs[i].set_ylabel("Summary statistic, L(r) - r")
        # axs[i].set_title(f"{(1-test_sequence[i])*100:.2f} % cove.")

    for ax in fig.get_axes():
        ax.label_outer()    
    plt.tight_layout(pad=0, w_pad=0, h_pad=0.5)
    plt.savefig(f"{path}_gi.pdf", dpi=700, bbox_inches="tight")
    # tikzplotlib.save("p_values/gi_gm_control_n20000.tex")
    plt.show()
