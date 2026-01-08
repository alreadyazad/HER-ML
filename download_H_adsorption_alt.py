# download_H_adsorption_alt.py
import requests
import pandas as pd

# REST endpoint: fetch reactions with reactant H and product H*
url = "https://api.catalysis-hub.org/reactions?reactants=H&products=H*"

print("Requesting data from Catalysis-Hub REST API...")
resp = requests.get(url, timeout=60)
resp.raise_for_status()

data = resp.json()

# 'reactions' key contains the list of reaction entries (depends on API response)
if 'reactions' in data:
    df = pd.DataFrame(data['reactions'])
else:
    # fallback: try to convert whole JSON to a flat table
    df = pd.json_normalize(data)

df.to_csv("H_adsorption_CatalysisHub_raw.csv", index=False)
print("Downloaded dataset saved as H_adsorption_CatalysisHub_raw.csv")
print("Rows:", len(df))
