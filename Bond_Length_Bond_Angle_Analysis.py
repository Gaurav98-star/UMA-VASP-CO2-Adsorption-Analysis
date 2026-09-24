import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# =============================================================================
# SETTINGS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The script is located directly inside the 1.1M3N1X directory.
# Therefore Bond_Distance and Bond_Angle are direct subdirectories.
BASE_DIR = SCRIPT_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "Resized_SE")

# Error bars are NOT compulsory for RMSE ranking.
# True  -> show standard-deviation error bars.
# False -> show only the mean RMSE bars.
SHOW_ERROR_BARS = True


# =============================================================================
# FONT SIZE SETTINGS (publication-quality / ACS-style)
# =============================================================================
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
TITLE_SIZE = 22
BAR_VALUE_SIZE = 10


# =============================================================================
# FIXED ELEMENT COLOURS
# =============================================================================
# Colours are assigned to chemical elements, NOT bar positions.
# Therefore, the same metal/dopant always has the same colour in every plot,
# even when the bars are sorted differently.

ELEMENT_COLORS = {
    # Fixed colours for metals
    "Ti": "#0072B2", "V": "#E69F00", "Cr": "#009E73",
    "Mn": "#D55E00", "Fe": "#CC79A7", "Co": "#56B4E9",
    "Ni": "#F0E442", "Cu": "#7A5195", "Zn": "#2F4B7C",
    "Mo": "#A05195", "Ru": "#665191", "Pd": "#003F5C",
    "Pt": "#8C564B", "Ag": "#17BECF", "Au": "#BCBD22",
    "Sc": "#1B9E77", "Y": "#D95F02", "Zr": "#7570B3",
    "Nb": "#E7298A", "Ta": "#66A61E", "W": "#E6AB02",
    "Re": "#A6761D", "Os": "#666666", "Ir": "#1F78B4",
    "Rh": "#B2DF8A", "Hf": "#FB9A99",

    # Fixed colours for Neighbor4 / dopant elements
    "H": "#A6CEE3", "B": "#8DD3C7", "C": "#FFFFB3",
    "N": "#BEBADA", "O": "#FB8072", "F": "#80B1D3",
    "Si": "#FDB462", "P": "#B3DE69", "S": "#FCCDE5",
    "Cl": "#D9D9D9", "Br": "#BC80BD", "I": "#CCEBC5",
    "Se": "#FFED6F", "Te": "#CAB2D6", "As": "#B15928",

    # Other commonly encountered neighbour elements
    "Al": "#6A3D9A", "Ga": "#33A02C", "Ge": "#E31A1C",
    "In": "#A6CEE3", "Sn": "#B2DF8A", "Sb": "#FB9A99",
    "Bi": "#FDBF6F", "Pb": "#FF7F00",
}

# Fallback colour for an element not explicitly listed above.\n# Add any new Neighbor4/dopant element to ELEMENT_COLORS for a unique fixed colour.
DEFAULT_ELEMENT_COLOR = "#808080"


def get_element_color(element):
    """Return a fixed colour for an element, independent of bar position."""
    return ELEMENT_COLORS.get(str(element).strip(), DEFAULT_ELEMENT_COLOR)



# =============================================================================
# FIND CSV FILES
# =============================================================================

def find_csv_files(folder_name):
    """
    Robustly find RMSE CSV files.

    Primary search:
        BASE_DIR/Bond_Distance/*.csv
        BASE_DIR/Bond_Angle/*.csv

    If the expected folder is not found or contains no CSV files, perform a
    recursive fallback search for CSV files containing the appropriate RMSE
    column:
        Bond_Distance -> RMSE_A
        Bond_Angle    -> RMSE_deg

    This makes the script robust to slightly different folder structures.
    """

    direct_folder = os.path.join(BASE_DIR, folder_name)

    files = []

    if os.path.isdir(direct_folder):
        files = sorted(
            glob.glob(
                os.path.join(direct_folder, "*.csv")
            )
        )

    # -------------------------------------------------------------------------
    # Recursive fallback
    # -------------------------------------------------------------------------
    if not files:

        if folder_name == "Bond_Distance":
            required_column = "RMSE_A"
        else:
            required_column = "RMSE_deg"

        all_csv = glob.glob(
            os.path.join(BASE_DIR, "**", "*.csv"),
            recursive=True
        )

        for csv_file in sorted(all_csv):

            # Never read output files produced by this script.
            if os.path.abspath(OUTPUT_DIR) in os.path.abspath(csv_file):
                continue

            try:
                header = pd.read_csv(
                    csv_file,
                    nrows=0
                ).columns.tolist()
            except Exception:
                continue

            if required_column in header:
                files.append(csv_file)

    return sorted(set(files))


# =============================================================================
# READ AND COMBINE RMSE CSV FILES
# =============================================================================

def read_rmse_files(files, analysis_type):
    """
    Read all RMSE CSV files and combine their row-level RMSE values.

    Bond distance files are expected to contain RMSE_A.
    Bond angle files are expected to contain RMSE_deg.
    """

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {analysis_type}."
        )

    if analysis_type == "Bond_Length":
        rmse_column = "RMSE_A"
        unit = "Å"
    else:
        rmse_column = "RMSE_deg"
        unit = "degrees"

    all_data = []

    print()
    print("=" * 80)
    print(f"READING {analysis_type}")
    print("=" * 80)

    for csv_file in files:
        print("Reading:", os.path.basename(csv_file))

        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            print("  ERROR:", exc)
            continue

        required = ["metal_element", "neighbor4_element", rmse_column]

        missing = [c for c in required if c not in df.columns]

        if missing:
            print("  Skipped - missing columns:", missing)
            continue

        temp = df[
            ["metal_element", "neighbor4_element", rmse_column]
        ].copy()

        temp = temp.rename(
            columns={
                "neighbor4_element": "dopant",
                rmse_column: "RMSE"
            }
        )

        temp["RMSE"] = pd.to_numeric(
            temp["RMSE"],
            errors="coerce"
        )

        temp = temp.dropna(
            subset=["metal_element", "dopant", "RMSE"]
        )

        temp["source_file"] = os.path.basename(csv_file)

        all_data.append(temp)

        print(f"  Valid RMSE rows: {len(temp)}")

    if not all_data:
        raise ValueError(
            f"No valid RMSE data found for {analysis_type}. "
            f"Check that the CSV files contain the expected RMSE column."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined, unit


# =============================================================================
# CALCULATE AVERAGE RMSE
# =============================================================================

def calculate_group_statistics(data, group_column):
    """
    Calculate mean RMSE, standard deviation, median, count and maximum
    absolute deviation for a grouping variable.
    """

    stats = (
        data.groupby(group_column)["RMSE"]
        .agg(
            Mean_RMSE="mean",
            Std_RMSE="std",
            Median_RMSE="median",
            Count="count",
            Max_RMSE="max",
            Min_RMSE="min"
        )
        .reset_index()
    )

    # A single observation has undefined standard deviation.
    stats["Std_RMSE"] = stats["Std_RMSE"].fillna(0.0)

    # Sort from largest to smallest mean RMSE.
    stats = stats.sort_values(
        "Mean_RMSE",
        ascending=False
    ).reset_index(drop=True)

    return stats


# =============================================================================
# BAR PLOT
# =============================================================================

def plot_rmse_bar(
    stats,
    category_column,
    analysis_type,
    unit,
    filename
):
    """
    Publication-quality colourful bar plot of average RMSE.

    The mean RMSE is shown by bar height. Standard-deviation error bars are
    intentionally optional; here they are retained because they quantify the
    spread of the individual RMSE values contributing to each mean.
    """

    category_values = stats[category_column].astype(str).values
    mean_values = stats["Mean_RMSE"].values
    std_values = stats["Std_RMSE"].values

    fig_width = max(7.0, 0.55 * len(category_values))

    fig, ax = plt.subplots(figsize=(fig_width, 5.8))

    x = np.arange(len(category_values))

    # Fixed colours based on the chemical element itself.
    # This prevents colours from changing when bar positions change after sorting.
    colors = [
        get_element_color(element)
        for element in category_values
    ]

    # Mean RMSE bars.
    bars = ax.bar(
        x,
        mean_values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.90,
        zorder=3
    )

    # Standard deviation error bars.
    if SHOW_ERROR_BARS:
        ax.errorbar(
            x,
            mean_values,
            yerr=std_values,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        category_values,
        rotation=45 if len(category_values) > 8 else 0,
        ha="right" if len(category_values) > 8 else "center",
        fontsize=TICK_LABEL_SIZE,
        fontweight="bold"
    )

    if category_column == "metal_element":
        xlabel = "Metal"
        title_group = "Metal-dependent"
    else:
        xlabel = "Dopant"
        title_group = "Dopant-dependent"

    descriptor_title = (
        "Bond Length" if analysis_type == "Bond_Length"
        else "Bond Angle"
    )

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(
        f"Average RMSE ({unit})",
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold"
    )


    # Light horizontal guide lines improve quantitative comparison.
    ax.grid(
        axis="y",
        alpha=0.22,
        linewidth=0.7,
        linestyle="--",
        zorder=0
    )

    ax.set_axisbelow(True)

    # Numerical mean RMSE labels.
    ymax = (
        max(mean_values + std_values)
        if SHOW_ERROR_BARS and len(mean_values)
        else max(mean_values)
        if len(mean_values)
        else 1.0
    )

    for bar, value, error in zip(
        bars,
        mean_values,
        std_values
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (error if SHOW_ERROR_BARS else 0) + 0.018 * ymax,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_SIZE,
            fontweight="bold"
        )

    # Independent Y-axis settings for Bond Length and Bond Angle
    if analysis_type == "Bond_Angle":
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(10))
    else:
        ax.set_ylim(0, 5.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        top=True,
        right=True
    )

    # Match Adsorbed.py tick-label styling.
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Clean scientific-figure appearance.
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        filename + ".png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        filename + ".pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Plot written:")
    print(" ", png_path)
    print(" ", pdf_path)


# =============================================================================
# ANALYSE ONE CATEGORY
# =============================================================================

def analyse(
    data,
    analysis_type,
    unit
):
    """
    Generate metal and dopant statistics and plots.
    """

    # -------------------------------------------------------------------------
    # METAL
    # -------------------------------------------------------------------------

    metal_stats = calculate_group_statistics(
        data,
        "metal_element"
    )

    metal_csv = os.path.join(
        OUTPUT_DIR,
        f"{analysis_type}_RMSE_by_Metal.csv"
    )

    metal_stats.to_csv(
        metal_csv,
        index=False
    )

    print()
    print(f"{analysis_type} - average RMSE by METAL")
    print(metal_stats.to_string(index=False))

    plot_rmse_bar(
        metal_stats,
        "metal_element",
        analysis_type,
        unit,
        f"{analysis_type}_Average_RMSE_by_Metal"
    )

    # -------------------------------------------------------------------------
    # DOPANT
    # -------------------------------------------------------------------------

    dopant_stats = calculate_group_statistics(
        data,
        "dopant"
    )

    dopant_csv = os.path.join(
        OUTPUT_DIR,
        f"{analysis_type}_RMSE_by_Dopant.csv"
    )

    dopant_stats.to_csv(
        dopant_csv,
        index=False
    )

    print()
    print(f"{analysis_type} - average RMSE by DOPANT")
    print(dopant_stats.to_string(index=False))

    plot_rmse_bar(
        dopant_stats,
        "dopant",
        analysis_type,
        unit,
        f"{analysis_type}_Average_RMSE_by_Dopant"
    )

    return metal_stats, dopant_stats


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("AVERAGE RMSE ANALYSIS")
    print("Bond Length + Bond Angle")
    print("Metal + Dopant (neighbor4)")
    print("=" * 80)
    print("Script directory:", SCRIPT_DIR)
    print("Project directory (1.1M3N1X):", BASE_DIR)
    print("Output directory:", OUTPUT_DIR)
    print()

    # -------------------------------------------------------------------------
    # FIND FILES
    # -------------------------------------------------------------------------

    distance_files = find_csv_files(
        "Bond_Distance"
    )

    angle_files = find_csv_files(
        "Bond_Angle"
    )

    print("Bond Distance CSV files found:", len(distance_files))
    for f in distance_files:
        print("   ", f)

    if not distance_files:
        print("WARNING: No Bond Distance CSV containing RMSE_A was found.")
        print("         Recursive search was performed from:", BASE_DIR)

    print("Bond Angle CSV files found:", len(angle_files))
    for f in angle_files:
        print("   ", f)

    # -------------------------------------------------------------------------
    # BOND LENGTH
    # -------------------------------------------------------------------------

    distance_data, distance_unit = read_rmse_files(
        distance_files,
        "Bond_Length"
    )

    distance_metal, distance_dopant = analyse(
        distance_data,
        "Bond_Length",
        distance_unit
    )

    # -------------------------------------------------------------------------
    # BOND ANGLE
    # -------------------------------------------------------------------------

    angle_data, angle_unit = read_rmse_files(
        angle_files,
        "Bond_Angle"
    )

    angle_metal, angle_dopant = analyse(
        angle_data,
        "Bond_Angle",
        angle_unit
    )

    # -------------------------------------------------------------------------
    # OVERALL SUMMARY
    # -------------------------------------------------------------------------

    overall = pd.concat(
        [
            distance_data.assign(
                Descriptor="Bond_Length"
            ),
            angle_data.assign(
                Descriptor="Bond_Angle"
            )
        ],
        ignore_index=True
    )

    overall.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "All_RMSE_values_combined.csv"
        ),
        index=False
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print("Publication plots and CSV summaries are in:")
    print(OUTPUT_DIR)

    print()
    print("Most deviating metals:")
    print(
        "  Bond Length:",
        distance_metal.iloc[0]["metal_element"],
        f"(RMSE = {distance_metal.iloc[0]['Mean_RMSE']:.4f} Å)"
    )
    print(
        "  Bond Angle :",
        angle_metal.iloc[0]["metal_element"],
        f"(RMSE = {angle_metal.iloc[0]['Mean_RMSE']:.4f}°)"
    )

    print()
    print("Least deviating metals:")
    print(
        "  Bond Length:",
        distance_metal.iloc[-1]["metal_element"],
        f"(RMSE = {distance_metal.iloc[-1]['Mean_RMSE']:.4f} Å)"
    )
    print(
        "  Bond Angle :",
        angle_metal.iloc[-1]["metal_element"],
        f"(RMSE = {angle_metal.iloc[-1]['Mean_RMSE']:.4f}°)"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
