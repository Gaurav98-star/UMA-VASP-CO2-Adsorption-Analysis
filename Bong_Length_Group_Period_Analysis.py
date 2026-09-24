#!/usr/bin/env python3
"""
Analyze bond-length RMSE deviations by periodic-table group and period.

Input:
    Overall_Metal_Bond_Length_RMSE_summary.csv

The script creates:
    1. Group-wise mean RMSE bar plot
    2. Period-wise mean RMSE bar plot
    3. Metal RMSE grouped by periodic-table group
    4. Metal RMSE grouped by period
    5. Summary CSV files for group- and period-wise statistics

By default, Mean_RMSE_A is used as the RMSE metric.
Change RMSE_COLUMN below to RMS_RMSE_A if desired.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "Overall_Metal_Bond_Length_RMSE_summary.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Group_Period_Analysis")

RMSE_COLUMN = "Mean_RMSE_A"
RMSE_LABEL = "Mean bond-length RMSE (Å)"

# Periodic-table information for the metals included in the dataset.
# Group numbers follow IUPAC numbering: 1–18.
ELEMENT_INFO = {
    "Sc": (3, 4), "Ti": (4, 4), "V": (5, 4), "Cr": (6, 4),
    "Mn": (7, 4), "Fe": (8, 4), "Co": (9, 4), "Ni": (10, 4),
    "Cu": (11, 4), "Zn": (12, 4),
    "Y": (3, 5), "Zr": (4, 5), "Nb": (5, 5), "Mo": (6, 5),
    "Tc": (7, 5), "Ru": (8, 5), "Rh": (9, 5), "Pd": (10, 5),
    "Ag": (11, 5), "Cd": (12, 5),
    "Hf": (4, 6), "Ta": (5, 6), "W": (6, 6), "Re": (7, 6),
    "Os": (8, 6), "Ir": (9, 6), "Pt": (10, 6), "Au": (11, 6),
}

# Consistent publication-style settings based on Adsorbed.py.
X_TICK_SIZE = 16
Y_TICK_SIZE = 18
AXIS_LABEL_SIZE = 22
TITLE_SIZE = 22
VALUE_SIZE = 16


def apply_adsorbed_style(ax):
    """Apply the requested Adsorbed.py visual style."""
    ax.grid(False)
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE, top=True, right=True)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")


def annotate_bars(ax, bars, values, ymax):
    offset = 0.018 * ymax if ymax > 0 else 0.01
    for bar, value in zip(bars, values):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold",
                rotation=90 if len(values) > 8 else 0
            )


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, filename + ".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, filename + ".pdf"),
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = ["Metal", RMSE_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    data = df[required].copy()
    data["Metal"] = data["Metal"].astype(str).str.strip()
    data[RMSE_COLUMN] = pd.to_numeric(data[RMSE_COLUMN], errors="coerce")
    data = data.dropna(subset=required).copy()

    data["Group"] = data["Metal"].map(
        lambda metal: ELEMENT_INFO.get(metal, (np.nan, np.nan))[0]
    )
    data["Period"] = data["Metal"].map(
        lambda metal: ELEMENT_INFO.get(metal, (np.nan, np.nan))[1]
    )

    unknown = data.loc[data["Group"].isna(), "Metal"].unique().tolist()
    if unknown:
        raise ValueError(
            f"No periodic-table group/period mapping found for: {unknown}"
        )

    data["Group"] = data["Group"].astype(int)
    data["Period"] = data["Period"].astype(int)

    data = data.sort_values(["Period", "Group", "Metal"]).reset_index(drop=True)
    data.to_csv(
        os.path.join(OUTPUT_DIR, "Metal_Bond_Length_RMSE_with_Group_Period.csv"),
        index=False
    )

    # ========================================================
    # GROUP-WISE SUMMARY
    # ========================================================

    group_summary = (
        data.groupby("Group")[RMSE_COLUMN]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "Mean_RMSE_A",
            "median": "Median_RMSE_A",
            "std": "Std_RMSE_A",
            "count": "N_metals",
            "min": "Minimum_RMSE_A",
            "max": "Maximum_RMSE_A"
        })
    )
    group_summary["std"] = group_summary["Std_RMSE_A"].fillna(0)
    group_summary.to_csv(
        os.path.join(OUTPUT_DIR, "Group_Wise_Bond_Length_RMSE_Summary.csv"),
        index=False
    )

    # ========================================================
    # PERIOD-WISE SUMMARY
    # ========================================================

    period_summary = (
        data.groupby("Period")[RMSE_COLUMN]
        .agg(["mean", "median", "std", "count", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": "Mean_RMSE_A",
            "median": "Median_RMSE_A",
            "std": "Std_RMSE_A",
            "count": "N_metals",
            "min": "Minimum_RMSE_A",
            "max": "Maximum_RMSE_A"
        })
    )
    period_summary["Std_RMSE_A"] = period_summary["Std_RMSE_A"].fillna(0)
    period_summary.to_csv(
        os.path.join(OUTPUT_DIR, "Period_Wise_Bond_Length_RMSE_Summary.csv"),
        index=False
    )

    # ========================================================
    # PLOT 1: GROUP-WISE MEAN RMSE
    # ========================================================

    x = np.arange(len(group_summary))
    values = group_summary["Mean_RMSE_A"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(x)), 7.5))
    bars = ax.bar(
        x, values, width=0.7,
        color="#E69F00", edgecolor="black", linewidth=0.9
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Group {g}" for g in group_summary["Group"]],
        fontsize=X_TICK_SIZE,
        fontweight="bold",
        rotation=45,
        ha="right"
    )
    ax.set_xlabel("Periodic-table group", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(RMSE_LABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Group-wise Bond-Length RMSE: UMA vs VASP",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=14
    )

    ymax = max(values) if len(values) else 1
    ax.set_ylim(0, 1.3)
    annotate_bars(ax, bars, values, ymax)
    apply_adsorbed_style(ax)
    save_plot(fig, "Group_Wise_Bond_Length_RMSE")

    # ========================================================
    # PLOT 2: PERIOD-WISE MEAN RMSE
    # ========================================================

    x = np.arange(len(period_summary))
    values = period_summary["Mean_RMSE_A"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.bar(
        x, values, width=0.65,
        color="#E69F00", edgecolor="black", linewidth=0.9
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Period {p}" for p in period_summary["Period"]],
        fontsize=X_TICK_SIZE,
        fontweight="bold"
    )
    ax.set_xlabel("Periodic-table period", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(RMSE_LABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Period-wise Bond-Length RMSE: UMA vs VASP",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=14
    )

    ymax = max(values) if len(values) else 1
    ax.set_ylim(0, 1.1)
    annotate_bars(ax, bars, values, ymax)
    apply_adsorbed_style(ax)
    save_plot(fig, "Period_Wise_Bond_Length_RMSE")

    # ========================================================
    # PLOT 3: METAL-WISE RMSE COLORED BY GROUP
    # ========================================================

    group_colors = plt.cm.viridis(
        np.linspace(0.1, 0.9, data["Group"].nunique())
    )
    unique_groups = sorted(data["Group"].unique())
    color_map = dict(zip(unique_groups, group_colors))

    plot_data = data.sort_values(["Group", "Period", "Metal"]).reset_index(drop=True)
    x = np.arange(len(plot_data))
    values = plot_data[RMSE_COLUMN].to_numpy()

    fig, ax = plt.subplots(figsize=(max(13, 0.55 * len(x)), 7.5))
    bars = ax.bar(
        x, values, width=0.72,
        color=[color_map[g] for g in plot_data["Group"]],
        edgecolor="black", linewidth=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_data["Metal"],
        fontsize=X_TICK_SIZE,
        fontweight="bold",
        rotation=45,
        ha="right"
    )
    ax.set_xlabel("Transition metal", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(RMSE_LABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Metal-wise Bond-Length RMSE Ordered by Group",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=14
    )

    ymax = max(values) if len(values) else 1
    ax.set_ylim(0, 1.1)
    apply_adsorbed_style(ax)
    save_plot(fig, "Metal_Wise_Bond_Length_RMSE_Ordered_by_Group")

    # ========================================================
    # PLOT 4: METAL-WISE RMSE ORDERED BY PERIOD
    # ========================================================

    plot_data = data.sort_values(["Period", "Group", "Metal"]).reset_index(drop=True)
    x = np.arange(len(plot_data))
    values = plot_data[RMSE_COLUMN].to_numpy()

    fig, ax = plt.subplots(figsize=(max(13, 0.55 * len(x)), 7.5))
    bars = ax.bar(
        x, values, width=0.72,
        color="#E69F00", edgecolor="black", linewidth=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_data["Metal"],
        fontsize=X_TICK_SIZE,
        fontweight="bold",
        rotation=45,
        ha="right"
    )
    ax.set_xlabel("Transition metal", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(RMSE_LABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "Metal-wise Bond-Length RMSE Ordered by Period",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=14
    )

    ymax = max(values) if len(values) else 1
    ax.set_ylim(0, 1.1)
    apply_adsorbed_style(ax)
    save_plot(fig, "Metal_Wise_Bond_Length_RMSE_Ordered_by_Period")

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("=" * 80)
    print("BOND-LENGTH RMSE GROUP AND PERIOD ANALYSIS")
    print(f"RMSE metric used: {RMSE_COLUMN}")
    print("=" * 80)

    print("\nGroup-wise mean RMSE:")
    print(group_summary.to_string(index=False))

    print("\nPeriod-wise mean RMSE:")
    print(period_summary.to_string(index=False))

    print("\nOutputs saved in:")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
