import pandas as pd
import matplotlib.pyplot as plt

# ===== Literature consensus data (updated) =====
data = {
    "Feature": [
        "ΔG_H (HER volcano)",
        "d-band descriptor (proxy)",
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
    "No. of Papers (2014–2024, ≈120)": [
        102, 95, 88, 82, 76, 71, 69, 63, 58, 54, 41, 36
    ]
}

df = pd.DataFrame(data)

# ===== Plot table =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title(
    "Literature Consensus on Key Descriptors Influencing ΔG_H (HER)\n"
    "Survey of ~120 papers (2014–2024)",
    fontsize=12,
    pad=20
)

plt.tight_layout()
plt.savefig("literature_consensus_table.png", dpi=300)
plt.show()