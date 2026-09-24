#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Ads_Energy")
INPUT_XLSX = os.path.join(SCRIPT_DIR, "1.1M4N1X-1.xlsx")

# Easy-to-change plotting settings
PARITY_FIGSIZE = (11.0, 9.0)

# Same colour sequence as the supplied metal-energy parity plot.
METAL_COLORS = {
    "Ag":"#0072B2", "Au":"#E69F00", "Cd":"#009E73", "Co":"#D55E00",
    "Cr":"#CC79A7", "Cu":"#56B4E9", "Fe":"#F0E442", "Hf":"#7A5195",
    "Ir":"#2F4B7C", "Mn":"#A05195", "Mo":"#665191", "Nb":"#003F5C",
    "Ni":"#0072B2", "Os":"#E69F00", "Pd":"#009E73", "Pt":"#D55E00",
    "Re":"#CC79A7", "Rh":"#56B4E9", "Ru":"#F0E442", "Sc":"#7A5195",
    "Ta":"#2F4B7C", "Tc":"#A05195", "Ti":"#665191", "V":"#003F5C",
    "W":"#0072B2", "Y":"#E69F00", "Zn":"#009E73", "Zr":"#D55E00"
}
DEFAULT_COLORS = ["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9","#F0E442","#7A5195","#2F4B7C","#A05195","#665191","#003F5C"]

def rmse(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan

def color_for(metal, i):
    return METAL_COLORS.get(str(metal), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(INPUT_XLSX):
        raise FileNotFoundError(f"Excel file not found: {INPUT_XLSX}")

    df = pd.read_excel(INPUT_XLSX, sheet_name=0)
    required = ["Metal", "Dopant", "Ads_Energy_UMA", "Ads_Energy_VASP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    data = df[required].copy()
    data["Ads_Energy_UMA"] = pd.to_numeric(data["Ads_Energy_UMA"], errors="coerce")
    data["Ads_Energy_VASP"] = pd.to_numeric(data["Ads_Energy_VASP"], errors="coerce")
    data = data.dropna(subset=required).copy()

    # VASP is the standard/reference.
    data["Error_eV"] = data["Ads_Energy_UMA"] - data["Ads_Energy_VASP"]
    data["Absolute_Error_eV"] = data["Error_eV"].abs()
    data["Squared_Error_eV2"] = data["Error_eV"] ** 2

    overall_rmse = rmse(data["Error_eV"])

    metal_stats = (data.groupby("Metal")
        .agg(RMSE_eV=("Error_eV", rmse),
             MAE_eV=("Absolute_Error_eV", "mean"),
             Mean_Error_eV=("Error_eV", "mean"),
             Std_Error_eV=("Error_eV", "std"),
             Count=("Error_eV", "count"),
             Mean_UMA_Ads_Energy_eV=("Ads_Energy_UMA", "mean"),
             Mean_VASP_Ads_Energy_eV=("Ads_Energy_VASP", "mean"))
        .reset_index().sort_values("RMSE_eV", ascending=False).reset_index(drop=True))
    metal_stats["Std_Error_eV"] = metal_stats["Std_Error_eV"].fillna(0)

    data.to_csv(os.path.join(OUTPUT_DIR, "UMA_VASP_Adsorption_Energy_Error_Detailed.csv"), index=False)
    metal_stats.to_csv(os.path.join(OUTPUT_DIR, "Adsorption_Energy_RMSE_by_Metal.csv"), index=False)

    # ---------------- Metal RMSE bar plot ----------------
    fig, ax = plt.subplots(figsize=(max(9, .58*len(metal_stats)), 6.5))
    x = np.arange(len(metal_stats))
    bars = ax.bar(x, metal_stats["RMSE_eV"],
                  color=[color_for(m,i) for i,m in enumerate(metal_stats["Metal"])],
                  edgecolor="black", linewidth=.9, alpha=.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metal_stats["Metal"].astype(str), rotation=45 if len(metal_stats)>8 else 0,
                       ha="right" if len(metal_stats)>8 else "center", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metal", fontsize=22, fontweight="bold")
    ax.set_ylabel("Adsorption energy RMSE (eV)", fontsize=22, fontweight="bold")
    ymax = max(metal_stats["RMSE_eV"]) if len(metal_stats) else 1
    for b,v in zip(bars, metal_stats["RMSE_eV"]):
        ax.text(b.get_x()+b.get_width()/2, v+.018*ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, ymax*1.16 if ymax>0 else 1)
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(axis="y", labelsize=18, top=True, right=True)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"Adsorption_Energy_RMSE_by_Metal.png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUTPUT_DIR,"Adsorption_Energy_RMSE_by_Metal.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- Averaged parity plot: ONE point per metal ----------------
    # Average UMA and VASP adsorption energies within each metal.
    averaged = (data.groupby("Metal", as_index=False)
        .agg(Ads_Energy_VASP_Mean_eV=("Ads_Energy_VASP","mean"),
             Ads_Energy_UMA_Mean_eV=("Ads_Energy_UMA","mean"),
             RMSE_eV=("Error_eV",rmse), Count=("Error_eV","count"))
        .sort_values("Metal").reset_index(drop=True))
    averaged.to_csv(os.path.join(OUTPUT_DIR,"Adsorption_Energy_Parity_Averaged_by_Metal.csv"), index=False)

    fig, ax = plt.subplots(figsize=PARITY_FIGSIZE)
    vals = pd.concat([averaged["Ads_Energy_VASP_Mean_eV"], averaged["Ads_Energy_UMA_Mean_eV"]])
    vmin, vmax = vals.min(), vals.max()
    pad = max((vmax-vmin)*.08, .1)
    lo, hi = vmin-pad, vmax+pad
    ax.plot([lo,hi],[lo,hi], "--", color="black", linewidth=2.0, label="Ideal: UMA = VASP", zorder=1)

    for i,row in averaged.iterrows():
        metal = str(row["Metal"])
        ax.scatter(row["Ads_Energy_VASP_Mean_eV"], row["Ads_Energy_UMA_Mean_eV"],
                   s=125, color=color_for(metal,i), edgecolor="black", linewidth=.9, alpha=.92,
                   label=f"{metal} (RMSE = {row['RMSE_eV']:.3f} eV)", zorder=3)

    ax.set_xlabel("Average VASP adsorption energy (eV)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Average UMA adsorption energy (eV)", fontsize=22, fontweight="bold")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", which="major", labelsize=18, top=True, right=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUTPUT_DIR,"Adsorption_Energy_Parity_Plot_by_Metal_Averaged.png"), dpi=600, bbox_inches="tight", pad_inches=.15, facecolor="white")
    fig.savefig(os.path.join(OUTPUT_DIR,"Adsorption_Energy_Parity_Plot_by_Metal_Averaged.pdf"), bbox_inches="tight", pad_inches=.15)
    plt.close(fig)

    print("="*80)
    print("ADSORPTION ENERGY RMSE: UMA vs VASP")
    print("VASP = STANDARD / REFERENCE")
    print("="*80)
    print(f"Valid rows: {len(data)}")
    print(f"Overall RMSE: {overall_rmse:.6f} eV")
    print("\nMetal-wise RMSE:")
    print(metal_stats.to_string(index=False))
    print("\nOutputs:")
    print(OUTPUT_DIR)
    print("="*80)

if __name__ == "__main__":
    main()
