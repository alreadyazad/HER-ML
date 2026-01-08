from cathub.query import get_reactions
import pandas as pd

# Download all hydrogen adsorption reactions from Catalysis-Hub
df = get_reactions(reactants=['H'], products=['H*'])

# Save as CSV
df.to_csv("H_adsorption_CatalysisHub_raw.csv", index=False)

print("Downloaded dataset saved as H_adsorption_CatalysisHub_raw.csv")
