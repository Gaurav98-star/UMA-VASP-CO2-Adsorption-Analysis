import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ColourFixed_Resized2")

UMA_CSV = os.path.join(SCRIPT_DIR, "energies_UMA_sorted.csv")
VASP_CSV = os.path.join(SCRIPT_DIR, "energies_VASP_sorted.csv")

# Number of columns in the parity-plot legend.
# Change this value later if you want a multi-column legend.
LEGEND_NCOL = 1


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"None of {candidates} found. Available columns: {list(df.columns)}"
    )


def calculate_rmse(errors):
    errors = pd.to_numeric(errors, errors="coerce").dropna()
    if len(errors) == 0:
        return np.nan
    return float(np.sqrt(np.mean(errors ** 2)))


# =============================================================================
# FIXED METAL COLOURS
# =============================================================================
# Colours are assigned to the chemical element itself, NOT to its plotting
# position. Therefore, each metal always retains the same colour in all plots.

ELEMENT_COLORS = {
    # Metals
    "Sc": "#1B9E77", "Ti": "#0072B2", "V": "#E69F00",
    "Cr": "#009E73", "Mn": "#D55E00", "Fe": "#CC79A7",
    "Co": "#56B4E9", "Ni": "#F0E442", "Cu": "#7A5195",
    "Zn": "#2F4B7C", "Y": "#D95F02", "Zr": "#7570B3",
    "Nb": "#E7298A", "Mo": "#A05195", "Ru": "#665191",
    "Rh": "#B2DF8A", "Pd": "#003F5C", "Ag": "#17BECF",
    "Hf": "#FB9A99", "Ta": "#66A61E", "W": "#E6AB02",
    "Re": "#A6761D", "Os": "#666666", "Ir": "#1F78B4",
    "Pt": "#8C564B", "Au": "#BCBD22",

    # Dopants / neighbour elements (neighbor3 and neighbor4)
    "H": "#A6CEE3", "B": "#8DD3C7", "C": "#FFFFB3",
    "N": "#BEBADA", "O": "#FB8072", "F": "#80B1D3",
    "Al": "#6A3D9A", "Si": "#FDB462", "P": "#B3DE69",
    "S": "#FCCDE5", "Cl": "#D9D9D9", "Br": "#BC80BD",
    "I": "#CCEBC5", "Ga": "#33A02C", "Ge": "#E31A1C",
    "As": "#B15928", "Se": "#FFED6F", "In": "#A6CEE3",
    "Sn": "#B2DF8A", "Sb": "#FB9A99", "Te": "#CAB2D6",
    "Bi": "#FDBF6F", "Pb": "#FF7F00",
}

DEFAULT_ELEMENT_COLOR = "#808080"

def get_metal_color(metal):
    """Return the fixed colour assigned to a metal element."""
    return ELEMENT_COLORS.get(str(metal).strip(), DEFAULT_ELEMENT_COLOR)


def make_parity_plot(data, output_dir):
    # -------------------------------------------------------------------------
    # AVERAGE ALL STRUCTURES FOR EACH METAL
    # -------------------------------------------------------------------------
    # This is the key modification:
    # one metal = one averaged VASP energy + one averaged UMA energy.
    # Make the string-converted metal name an explicit column before grouping.
    # This avoids the pandas FutureWarning and ensures that "Metal" remains
    # available in the resulting DataFrame.
    data_for_plot = data.copy()
    data_for_plot["Metal"] = data_for_plot["Metal"].astype(str)

    averaged = (
        data_for_plot.groupby("Metal", as_index=False)
        .agg(
            VASP_Energy=("VASP_Energy", "mean"),
            UMA_Energy=("UMA_Energy", "mean"),
            RMSE_eV=("Error", lambda x: calculate_rmse(x)),
            Count=("Error", "count")
        )
        .sort_values("Metal")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    all_values = pd.concat([
        averaged["VASP_Energy"],
        averaged["UMA_Energy"]
    ])

    vmin = all_values.min()
    vmax = all_values.max()
    pad = max((vmax - vmin) * 0.08, 0.1)

    line_min = vmin - pad
    line_max = vmax + pad

    # Ideal parity line spanning the complete plotting range.
    ax.plot(
        [-285, -255],
        [-285, -255],
        linestyle="--",
        linewidth=2.5,
        color="black",
        zorder=1
    )

    # EXACTLY ONE POINT PER METAL.
    for _, row in averaged.iterrows():
        metal = str(row["Metal"])
        color = get_metal_color(metal)

        ax.scatter(
            row["VASP_Energy"],
            row["UMA_Energy"],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.90,
            label=f"{metal} (RMSE = {row['RMSE_eV']:.3f} eV)",
            zorder=3
        )

    ax.set_xlabel(
        "Average VASP energy (eV)",
        fontsize=22,
        fontweight="bold"
    )
    ax.set_ylabel(
        "Average UMA energy (eV)",
        fontsize=22,
        fontweight="bold"
    )
    # Fixed parity-plot range for both axes.
    ax.set_xlim(-285, -255)
    ax.set_ylim(-285, -255)
    ax.set_aspect("equal", adjustable="box")

    # Major tick spacing: 5 eV on both axes.
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    ax.grid(
        axis="both",
        alpha=0.22,
        linewidth=0.7,
        linestyle="--",
        zorder=0
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        top=True,
        right=True
    )

    # Match x- and y-axis tick-label typography.
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout(pad=1.5)

    png = os.path.join(
        output_dir,
        "Energy_Parity_Plot_by_Metal_Averaged.png"
    )
    pdf = os.path.join(
        output_dir,
        "Energy_Parity_Plot_by_Metal_Averaged.pdf"
    )

    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    averaged.to_csv(
        os.path.join(output_dir, "Energy_Parity_Averaged_by_Metal.csv"),
        index=False
    )

    print("Averaged parity plot:")
    print(" ", png)
    print(" ", pdf)
    print("Averaged parity data:")
    print(" ", os.path.join(
        output_dir, "Energy_Parity_Averaged_by_Metal.csv"
    ))


def make_metal_rmse_plot(metal_stats, output_dir):
    stats = (
        metal_stats
        .sort_values("RMSE_eV", ascending=False)
        .reset_index(drop=True)
    )

    # Fixed colours based on the metal element itself, independent of RMSE rank.
    colors = [
        get_metal_color(metal)
        for metal in stats["Metal"].astype(str)
    ]

    fig_width = max(7.0, 0.6 * len(stats))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))

    x = np.arange(len(stats))

    bars = ax.bar(
        x,
        stats["RMSE_eV"],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.90,
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        stats["Metal"].astype(str),
        rotation=45 if len(stats) > 8 else 0,
        ha="right" if len(stats) > 8 else "center",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Metal", fontsize=18, fontweight="bold")
    ax.set_ylabel("Energy RMSE (eV)", fontsize=18, fontweight="bold")
    ymax = max(stats["RMSE_eV"]) if len(stats) else 1.0

    for bar, value in zip(bars, stats["RMSE_eV"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018 * ymax,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)

    ax.grid(
        axis="y",
        alpha=0.22,
        linewidth=0.7,
        linestyle="--",
        zorder=0
    )

    ax.set_axisbelow(True)
    # Increase y-axis tick-value font size.
    ax.tick_params(
        axis="y",
        labelsize=16,
        top=True,
        right=True
    )
    # Keep x-axis ticks large and bold for the metal labels.
    ax.tick_params(
        axis="x",
        labelsize=16,
        top=True,
        right=True
    )

    fig.tight_layout()

    png = os.path.join(output_dir, "Energy_RMSE_by_Metal.png")
    pdf = os.path.join(output_dir, "Energy_RMSE_by_Metal.pdf")

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Metal RMSE plot:")
    print(" ", png)
    print(" ", pdf)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("ENERGY RMSE ANALYSIS: UMA vs VASP")
    print("VASP = STANDARD / REFERENCE")
    print("=" * 80)

    if not os.path.isfile(UMA_CSV):
        raise FileNotFoundError(f"UMA CSV not found: {UMA_CSV}")

    if not os.path.isfile(VASP_CSV):
        raise FileNotFoundError(f"VASP CSV not found: {VASP_CSV}")

    uma = pd.read_csv(UMA_CSV)
    vasp = pd.read_csv(VASP_CSV)

    uma_path = find_column(uma, ["Folder Path"])
    uma_energy = find_column(
        uma, ["UMA Energy (eV)", "UMA_Energy", "Energy (eV)"]
    )
    metal = find_column(uma, ["Metal", "metal_element"])
    dopant = find_column(uma, ["Dopant", "dopant"])

    vasp_path = find_column(vasp, ["Folder Path"])
    vasp_energy = find_column(
        vasp, ["Energy (eV)", "VASP Energy (eV)", "VASP_Energy"]
    )

    print(f"UMA rows : {len(uma)}")
    print(f"VASP rows: {len(vasp)}")

    uma_use = uma[
        [uma_path, metal, dopant, uma_energy]
    ].copy()
    uma_use.columns = [
        "Folder Path", "Metal", "Dopant", "UMA_Energy"
    ]

    vasp_use = vasp[
        [vasp_path, vasp_energy]
    ].copy()
    vasp_use.columns = ["Folder Path", "VASP_Energy"]

    uma_use = uma_use.drop_duplicates(
        "Folder Path", keep="first"
    )
    vasp_use = vasp_use.drop_duplicates(
        "Folder Path", keep="first"
    )

    merged = pd.merge(
        uma_use,
        vasp_use,
        on="Folder Path",
        how="inner",
        validate="one_to_one"
    )

    merged["UMA_Energy"] = pd.to_numeric(
        merged["UMA_Energy"], errors="coerce"
    )
    merged["VASP_Energy"] = pd.to_numeric(
        merged["VASP_Energy"], errors="coerce"
    )

    merged = merged.dropna(
        subset=[
            "Folder Path",
            "Metal",
            "UMA_Energy",
            "VASP_Energy"
        ]
    ).copy()

    # VASP is standard: Error = UMA - VASP.
    merged["Error"] = (
        merged["UMA_Energy"] - merged["VASP_Energy"]
    )
    merged["Absolute_Error_eV"] = merged["Error"].abs()
    merged["Squared_Error_eV2"] = merged["Error"] ** 2

    overall_rmse = calculate_rmse(merged["Error"])
    overall_mae = merged["Absolute_Error_eV"].mean()
    overall_bias = merged["Error"].mean()

    # -------------------------------------------------------------------------
    # METAL RMSE
    # -------------------------------------------------------------------------
    metal_stats = (
        merged.groupby("Metal")
        .agg(
            RMSE_eV=("Error", calculate_rmse),
            MAE_eV=("Absolute_Error_eV", "mean"),
            Mean_Error_eV=("Error", "mean"),
            Std_Error_eV=("Error", "std"),
            Count=("Error", "count"),
            Max_Absolute_Error_eV=(
                "Absolute_Error_eV", "max"
            )
        )
        .reset_index()
        .sort_values("RMSE_eV", ascending=False)
        .reset_index(drop=True)
    )

    metal_stats["Std_Error_eV"] = (
        metal_stats["Std_Error_eV"].fillna(0.0)
    )

    # -------------------------------------------------------------------------
    # DOPANT RMSE
    # -------------------------------------------------------------------------
    dopant_stats = (
        merged.groupby("Dopant")
        .agg(
            RMSE_eV=("Error", calculate_rmse),
            MAE_eV=("Absolute_Error_eV", "mean"),
            Mean_Error_eV=("Error", "mean"),
            Std_Error_eV=("Error", "std"),
            Count=("Error", "count"),
            Max_Absolute_Error_eV=(
                "Absolute_Error_eV", "max"
            )
        )
        .reset_index()
        .sort_values("RMSE_eV", ascending=False)
        .reset_index(drop=True)
    )

    dopant_stats["Std_Error_eV"] = (
        dopant_stats["Std_Error_eV"].fillna(0.0)
    )

    # Save tables.
    merged.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "UMA_VASP_Energy_Error_Detailed.csv"
        ),
        index=False
    )

    metal_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Energy_RMSE_by_Metal.csv"
        ),
        index=False
    )

    dopant_stats.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "Energy_RMSE_by_Dopant.csv"
        ),
        index=False
    )

    make_parity_plot(merged, OUTPUT_DIR)
    make_metal_rmse_plot(metal_stats, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("OVERALL ENERGY AGREEMENT")
    print("=" * 80)
    print(f"Matched structures : {len(merged)}")
    print(f"Overall RMSE       : {overall_rmse:.6f} eV")
    print(f"Overall MAE        : {overall_mae:.6f} eV")
    print(f"Mean signed error  : {overall_bias:.6f} eV")

    print()
    print("=" * 80)
    print("ENERGY RMSE BY METAL")
    print("=" * 80)
    print(metal_stats.to_string(index=False))

    print()
    print("=" * 80)
    print("ENERGY RMSE BY DOPANT")
    print("=" * 80)
    print(dopant_stats.to_string(index=False))

    print()
    print("IMPORTANT PARITY-PLOT NOTE:")
    print("  Multiple structures of the same metal are averaged.")
    print("  Therefore the parity plot has ONE point per metal.")
    print()
    print("Output directory:")
    print(OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
