#!/usr/bin/env python3

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# X COORDINATES
# ============================================================

x_labels = [
    "Str1",
    "Str2",
    "Str3",
    "Str4",
    "Str5",
    "Str6",
    "Str7",
    "Str8"
]

# ============================================================
# Y COORDINATES
# Replace these 8 example values with your actual Y values.
# Keep them in the same order as the X coordinates above.
# ============================================================

y_values = [
    0.7659,  # 1.1M4N1X-1
    0.8529,  # 2.1M4N1X-2
    0.8429,  # 3.1M4N1X-3
    0.8521,  # 4.1M4N1X-4
    0.8143,  # 5.1M4N1X-1
    0.8813,  # 6.1M4N1X-2
    0.8721,  # 7.1M4N1X-3
    0.9494   # 8.1M4N1X-4
]

# ============================================================
# CHECK DATA
# ============================================================

if len(x_labels) != 8 or len(y_values) != 8:
    raise ValueError("There must be exactly 8 X labels and 8 Y values.")

# ============================================================
# CREATE plot FOLDER
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(base_dir, "plot2")
os.makedirs(plot_dir, exist_ok=True)

output_file = os.path.join(plot_dir, "M4N1X_Y_Coordinates.png")

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7.5))

# Blue bars
ax.bar(
    range(8),
    y_values,
    width=0.75,
    color="#007C91",
    edgecolor="Black"
)

# X coordinate labels
ax.set_xticks(range(8))
ax.set_xticklabels(
    x_labels,
    fontsize=16,
    fontweight="bold",
    rotation=45,
    ha="right"
)

# Y-axis tick labels
ax.tick_params(axis="y", labelsize=18)

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.set_ylabel(
    "Spearman's rho (VASP vs UMA)",
    fontsize=22,
    fontweight="bold"
)

# X coordinate axis
ax.set_xlabel(
    "Structures",
    fontsize=22,
    fontweight="bold"
)
# Y-axis range
ax.set_ylim(0, max(y_values) * 1.08)

# Keep the complete rectangular box
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.2)

# White background
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# No grid
ax.grid(False)

plt.tight_layout()

# ============================================================
# SAVE IMAGE
# ============================================================

fig.savefig(
    output_file,
    dpi=600,
    facecolor="white",
    bbox_inches="tight"
)

plt.close(fig)

print("")
print("Plot generated successfully.")
print("X coordinate labels and Y coordinate values included.")
print("Saved to:")
print(output_file)
print("")
