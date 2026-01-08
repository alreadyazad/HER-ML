import requests
import pandas as pd

url = "https://api.catalysis-hub.org/graphql"

query = """
{
  reactions(filter: {reactants: "H", products: "H*"}) {
    id
    reactants
    products
    surfaceComposition
    reactionEnergy
    activationEnergy
    dftCode
    dftFunctional
    sites
    publications {
      title
      authors
      year
    }
  }
}
"""

print("Requesting H adsorption data from Catalysis-Hub GraphQL API...")

response = requests.post(url, json={"query": query})
response.raise_for_status()

data = response.json()

# Extract the reaction list
reactions = data["data"]["reactions"]

df = pd.DataFrame(reactions)
df.to_csv("H_adsorption_CatalysisHub_raw.csv", index=False)

print("DONE! Saved as H_adsorption_CatalysisHub_raw.csv")
print("Rows downloaded:", len(df))
