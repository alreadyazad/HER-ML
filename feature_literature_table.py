import pandas as pd
import matplotlib.pyplot as plt

# ----- feature consensus data -----
data = {
    "Feature": [
        "ΔG_H (HER volcano)",
        "d-band descriptor",
        "Alloying / composition effect",
        "Surface electronic structure",
        "Electronegativity (mean)",
        "Electronegativity mismatch",
        "Atomic radius (mean)",
        "Atomic radius mismatch",
        "Lattice distortion / strain",
        "Valence Electron Concentration (VEC)",
        "Mixing entropy",
        "Site diversity / disorder"
    ],
    "No. of Papers (≈60)": [
        50, 45, 40, 34, 30, 28, 26, 24, 22, 20, 16, 14
    ]
}

df = pd.DataFrame(data)

# ----- create table figure -----
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.6)

plt.title(
    "Literature Consensus on Key Descriptors Influencing ΔG_H (HER)",
    fontsize=14,
    pad=15
)

plt.savefig("feature_consensus_table.png", dpi=300, bbox_inches='tight')
plt.show()