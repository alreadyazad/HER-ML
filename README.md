# HER-ML  
#  Machine Learning–Driven Screening of HER Catalysts  
**(TEEP Research Internship, National Dong Hwa University (NDHU) Hualien, Taiwan)**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Research](https://img.shields.io/badge/Research-Computational%20Catalysis-green)
![TEEP](https://img.shields.io/badge/Program-TEEP-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

##  Overview

This project focuses on **machine learning–assisted discovery of efficient Hydrogen Evolution Reaction (HER) catalysts**.  
The objective is to **predict hydrogen adsorption free energy (ΔGₕ)** for transition-metal and alloy catalysts using **composition-derived descriptors**, enabling **large-scale virtual screening** without expensive **DFT calculations**.

The work was carried out as part of a **research internship at National Dong Hwa University (NDHU), Taiwan**, under the **Taiwan Education Experience Program (TEEP)**.

---

##  Objectives

- Build ML models to predict **ΔGₕ** for HER catalysts  
- Compare electronic structure descriptors:
  - **d-band center** (DFT-derived)
  - **d-band proxy** (composition-derived)
- Enable **high-throughput screening** of thousands of alloy compositions  
- Identify **physically interpretable and scalable descriptors** from literature  

---

##  Scientific Background

- Optimal HER activity occurs when **ΔGₕ ≈ 0 eV**
- Traditional descriptor: **d-band center** → requires DFT
- Practical alternative: **d-band proxy**, computed from alloy composition
- **100+ peer-reviewed papers (last 5–10 years)** surveyed to justify feature selection

---

##  Project Structure
HER-ML-Project/
├── data/
│ ├── her_dataset_200.csv
│ ├── screening_*.csv
│ ├── dband_center_dataset.csv
│ └── dband_proxy_dataset.csv
│
├── models/
│ ├── rf_fullfeature_model_fixed.joblib
│ └── xgb_model_fixed.joblib
│
├── scripts/
│ ├── feature_engineering.py
│ ├── model_rf.py
│ ├── model_xgb.py
│ ├── screening_predict.py
│ └── screening_generate.py
│
├── figures/
│ ├── true_vs_pred_rf.png
│ └── true_vs_pred_xgb.png
│
└── README.md

---

##  Features Used

### Composition-Based
- Elemental fractions: **Ni, Fe, Co, Mo, Cu, Mn, Cr, V**

### Physicochemical Descriptors
- Electronegativity (mean, variance, mismatch)
- Atomic radius (mean, variance, mismatch)
- Valence Electron Concentration (VEC)
- Mixing entropy
- **d-band proxy**

---

##  Machine Learning Models

### Models Implemented
- **Random Forest Regressor**
- **XGBoost Regressor**

### Evaluation Metrics
- R² score
- Mean Absolute Error (MAE)
- MAE (%) for interpretability

---

##  Key Findings

- **d-band center** requires direct DFT values → limited scalability
- **d-band proxy** can be computed from composition → scalable
- For the same alloy dataset, **d-band proxy model achieved lower MAE**
- Composition-based descriptors enable **screening of thousands of alloys**
- **Ni-bias** in literature datasets identified and analyzed

---

##  Screening Pipeline

1. Generate alloy compositions (binary, ternary, quaternary)  
2. Compute features from composition  
3. Predict **ΔGₕ** using trained ML models  
4. Filter candidates near **ΔGₕ ≈ 0 eV**  
5. Identify promising HER catalyst candidates  

---

##  Data Sources

- Peer-reviewed literature (DFT-based HER studies)
- Catalysis Hub (reaction energies)
- Curated datasets from published volcano plots

---

##  Internship & Funding

- **Host Institution:** National Dong Hwa University (NDHU), Taiwan  
- **Program:** Taiwan Experience Education Program (TEEP)  
- **Funding:** Ministry of Education, Taiwan  

---

##  Future Work

- Expand dataset with additional transition-metal alloys  
- Reduce elemental bias using balanced sampling  
- Incorporate uncertainty estimation  
- Validate top candidates with **DFT calculations**

---

##  License

This project is intended **for academic and research purposes only**.
