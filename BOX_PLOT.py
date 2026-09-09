#!/usr/bin/env python3

import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt


folder = "./"

# x-axis labels
labels = ["HLV", "HL", "HV", "LV"]


suffix = ["HLV", "HL", "HV", "LV"]

bay_prefix = "Area_Search_BAY"
gw_prefix  = "Area_Search_GW"

def read_quantity(filename, quantity):

    with open(filename) as f:
        text = f.read()

    m = re.search(rf"{quantity}\s*:\s*(\[[^\]]+\])", text)

    if m is None:
        raise RuntimeError(f"{quantity} not found in {filename}")

    return ast.literal_eval(m.group(1))




bay90 = []
gw90 = []

bay50 = []
gw50 = []

baySearch = []
gwSearch = []

for s in suffix:

    bay_file = os.path.join(folder, f"{bay_prefix}_{s}.txt")
    gw_file  = os.path.join(folder, f"{gw_prefix}_{s}.txt")

    bay90.append(read_quantity(bay_file, "area_90"))
    gw90.append(read_quantity(gw_file, "area_90"))

    bay50.append(read_quantity(bay_file, "area_50"))
    gw50.append(read_quantity(gw_file, "area_50"))

    baySearch.append(read_quantity(bay_file, "search_area"))
    gwSearch.append(read_quantity(gw_file, "search_area"))


def make_panel(ax, bay_data, gw_data, title):

    pos = np.arange(len(labels))

    width = 0.35

    bp1 = ax.boxplot(
        bay_data,
        positions=pos-width/2,
        widths=0.30,
        patch_artist=True,
        showfliers=False
    )

    bp2 = ax.boxplot(
        gw_data,
        positions=pos+width/2,
        widths=0.30,
        patch_artist=True,
        showfliers=False
    )

    # colors
    for box in bp1["boxes"]:
        box.set(facecolor="#77AADD", edgecolor="black")

    for box in bp2["boxes"]:
        box.set(facecolor="#F6BE7A", edgecolor="black")

    for key in ["whiskers","caps","medians"]:
        plt.setp(bp1[key], color="black")
        plt.setp(bp2[key], color="black")

    ax.set_title(title, fontsize=15)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=12)

    ax.grid(axis="y", alpha=0.3)

    return bp1, bp2




fig, axs = plt.subplots(
    1,
    3,
    figsize=(15,5),
    sharey=True
)

bp1, bp2 = make_panel(
    axs[0],
    bay90,
    gw90,
    "90% Credible Area"
)

make_panel(
    axs[1],
    bay50,
    gw50,
    "50% Credible Area"
)

make_panel(
    axs[2],
    baySearch,
    gwSearch,
    "Searched Area"
)


axs[0].set_ylabel(r"Area (deg$^2$)", fontsize=16)

for ax in axs:
    ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend
fig.legend(
    [bp1["boxes"][0], bp2["boxes"][0]],
    ["BAYESTAR", "GW-SkyLocator-ARQS"],
    loc="upper center",
    ncol=2,
    fontsize=13,
    frameon=False,
    bbox_to_anchor=(0.5,1.03)
)

plt.tight_layout(rect=[0,0,1,0.95])


plt.savefig('BOX_PLOT.eps',format="eps")
