import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

plt.rcParams["font.family"]="Times New Roman"

# Publication-level font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
})

OUTDIR="No_Title"
os.makedirs(OUTDIR,exist_ok=True)

files=sorted(glob.glob("*.csv"))
frames=[]
for f in files:
    d=pd.read_csv(f)
    frames.append(d)
raw=pd.concat(frames,ignore_index=True)

metal=[c for c in raw.columns if "metal" in c.lower() and "element" in c.lower()][0]
num=raw.select_dtypes(include=np.number).columns.tolist()
bond=[c for c in num if ("bond" in c.lower()) or ("distance" in c.lower())]
angle=[c for c in num if "angle" in c.lower()]

# Keep the ORIGINAL colour assignment unchanged.
metals=sorted(raw[metal].astype(str).unique())
cmap=cm.get_cmap("tab20",len(metals))
METAL_COL={m:cmap(i) for i,m in enumerate(metals)}

# Publication x-axis order: 3d -> 4d -> 5d.
# Within each series, metals are grouped by periodic-table group.
TRANSITION_METAL_ORDER = {
    # 3d
    "Sc": (3, 3), "Ti": (3, 4), "V": (3, 5), "Cr": (3, 6),
    "Mn": (3, 7), "Fe": (3, 8), "Co": (3, 9), "Ni": (3, 10),
    "Cu": (3, 11), "Zn": (3, 12),

    # 4d
    "Y": (4, 3), "Zr": (4, 4), "Nb": (4, 5), "Mo": (4, 6),
    "Tc": (4, 7), "Ru": (4, 8), "Rh": (4, 9), "Pd": (4, 10),
    "Ag": (4, 11), "Cd": (4, 12),

    # 5d
    "Hf": (5, 4), "Ta": (5, 5), "W": (5, 6), "Re": (5, 7),
    "Os": (5, 8), "Ir": (5, 9), "Pt": (5, 10), "Au": (5, 11),
    "Hg": (5, 12),
}

def periodic_metal_order(m):
    return TRANSITION_METAL_ORDER.get(str(m), (99, 99, str(m)))

def summary(cols):
    rows=[]
    for m,g in raw.groupby(metal):
        v=np.abs(g[cols].to_numpy(float).ravel())
        v=v[np.isfinite(v)]
        rows.append([m,v.mean(),np.median(v),v.std(ddof=1),v.max()])
    return pd.DataFrame(rows,columns=["Metal","Mean","Median","Std","Max"]).sort_values("Mean")

bonddf=summary(bond)
angledf=summary(angle)

bonddf.to_csv(f"{OUTDIR}/Bond_Percent_Deviation_By_Metal.csv",index=False)
angledf.to_csv(f"{OUTDIR}/Angle_Percent_Deviation_By_Metal.csv",index=False)

def rankplot(df,title,name):
    fig,ax=plt.subplots(figsize=(8,10))
    d=df.copy()
    d["_order"] = d["Metal"].astype(str).map(periodic_metal_order)
    d=d.sort_values("_order",kind="stable")

    ax.barh(d["Metal"],d["Mean"],
            color=[METAL_COL[m] for m in d["Metal"]],
            edgecolor="black")
    ax.set_xlabel("Average Deviation (%)",fontsize=20,fontweight="bold")
    ax.set_ylabel("Metal",fontsize=20,fontweight="bold")
    ax.tick_params(axis="x",labelsize=16)
    ax.tick_params(axis="y",labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{name}",dpi=600)
    plt.close()

rankplot(bonddf,"Bond Length Deviation","Bond_Ranking.png")
rankplot(angledf,"Bond Angle Deviation","Angle_Ranking.png")

def coloured_box(cols,title,name):
    order=sorted(
        summary(cols)["Metal"].astype(str).tolist(),
        key=periodic_metal_order
    )
    data=[]
    for m in order:
        g=raw[raw[metal]==m]
        v=np.abs(g[cols].to_numpy(float).ravel())
        v=v[np.isfinite(v)]
        data.append(v)
    fig,ax=plt.subplots(figsize=(max(14,len(order)*0.45),7))
    bp=ax.boxplot(data,patch_artist=True,showfliers=False)
    for box,m in zip(bp["boxes"],order):
        box.set(facecolor=METAL_COL[m],edgecolor="black",linewidth=1.1)
    for med in bp["medians"]:
        med.set(color="black",linewidth=1.6)
    ax.set_xticklabels(order,rotation=90,fontsize=16,fontweight="bold")
    ax.set_xlabel("Metal",fontsize=20,fontweight="bold",labelpad=10)
    ax.set_ylabel("Deviation (%)",fontsize=20,fontweight="bold")
    ax.tick_params(axis="y",labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{name}",dpi=600)
    plt.close()

coloured_box(bond,"Bond Length Deviation Distribution","Bond_Boxplot.png")
coloured_box(angle,"Bond Angle Deviation Distribution","Angle_Boxplot.png")

scatter=bonddf[["Metal","Mean"]].merge(angledf[["Metal","Mean"]],on="Metal",suffixes=("_Bond","_Angle"))

pts=scatter[["Mean_Bond","Mean_Angle"]].values
k=min(5,len(scatter))
labels=KMeans(n_clusters=k,random_state=42,n_init=20).fit_predict(pts)
scatter["Cluster"]=labels

cluster_colors=cm.Set2(np.linspace(0,1,k))

fig,ax=plt.subplots(figsize=(8,7))

for c,col in enumerate(cluster_colors):
    s=scatter[scatter.Cluster==c]
    if len(s)>=3:
        try:
            hull=ConvexHull(s[["Mean_Bond","Mean_Angle"]].values)
            poly=s.iloc[hull.vertices]
            ax.fill(poly["Mean_Bond"],poly["Mean_Angle"],alpha=0.18,color=col)
        except:
            pass
    ax.scatter(s["Mean_Bond"],s["Mean_Angle"],
               s=170,color=col,edgecolor="black",
               label=f"Cluster {c+1}")
    for _,r in s.iterrows():
        ax.text(r["Mean_Bond"],r["Mean_Angle"],r["Metal"],
                fontsize=16,fontweight="bold")

ax.set_xlabel("Mean Bond Deviation (%)",fontsize=22,fontweight="bold")
ax.set_ylabel("Mean Angle Deviation (%)",fontsize=22,fontweight="bold")
ax.tick_params(axis="both", labelsize=18)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(alpha=.3)
ax.legend(fontsize=15, title_fontsize=17)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/Clustered_Bond_Angle_Scatter.png",dpi=600)
plt.close()

print("Finished.")
