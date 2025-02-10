# PROMIS: PROcessing MItigation of Spatial Bias

This repository contains the official code for the paper titled **PROMIS: PROcessing MItigation of Spatial Bias**. 

## Table of Contents

- [Data Source and Download Information](#data-source-and-download-information)
- [Experiments](#experiments)
- [Main PROMIS Methods](#main-promis-methods)
- [Analysis](#analysis)
- [Example Use Cases of PROMIS](#example-use-cases-of-promis)
- [Requirements and Setup](#requirements-and-setup)

---

## Data Source and Download Information

The following datasets have been downloaded from their respective sources and stored in the data/datasets/ directory for reproducibility. Below are the details for each file:

* CRIME Dataset
    * Source: [Crime Data from 2010 to 2019](<https://data.lacity.org/Public-Safety/Crime-Data-from-2010-to-2019/63jg-8b9z>)
    * Download Date: 10/1/2024
    * The dataset is provided as a ZIP file and is updated monthly. For reproducibility, it was downloaded and saved as Crime_Data_from_2010_to_2019.csv.zip
* LAR Dataset
    * Source: [CFPB Modified LAR Data 2021](<https://ffiec.cfpb.gov/data-publication/modified-lar/2021>)
    * The dataset for 2021 was obtained by, selecting the year 2021, entering the Legal Entity Identifier (LEI) B4TYDEB6GKMZO031MB27 for Bank of America, choosing the option "Include File Header." and clicking on "Download Modified LAR with Header."
    * It was then saved as: B4TYDEB6GKMZO031MB27.csv.zip
* Census Gazetteer Data
    * Source: [Census Gazetteer Files 2021](<https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2021.html>)
    * The file was obtained by navigating to the "Census Tracts" section and clicking on "Download the National Census Tracts Gazetteer Files."
    * It was saved as: 2021_Gaz_tracts_national.txt.zip
---

## Experiments

1. **Creating Worlds (Datasets)**
   - Notebook: `src/create_worlds.ipynb`  
   - Description: Preprocesses datasets, fits an XGBoost model, and creates the audit regions and the semi-synthetic dataset.  
   **Important**: Please run this notebook first to ensure all necessary data and models are set up correctly. 
2. **Audit Experiment**
   - Notebook: `src/experiments/audit_exp.ipynb`  
   - Description: Runs the audit experiment.

3. **DNN Base Model Experiment**
   - Notebook: `src/experiments/dnn_exp.ipynb`  
   - Description: Experiments using a DNN as the base model.

4. **LAR Dataset Experiment**
   - Script: `src/experiments/lar_exp.py`  
   - Description: Experiments with the LAR dataset.

5. **PROMIS Opt (High Work Limit) on LAR Dataset**
   - Script: `src/experiments/promis_opt_high_wlimit_lar_exp.py`  
   - Description: Experiment with the LAR dataset using the PROMIS Opt method with a higher work limit.

6. **Unfair-by-Design (Semi-synthetic) Experiment**
   - Script: `src/experiments/semi_synthetic_exp.py`  
   - Description: Experiments on a semi-synthetic dataset designed to be unfair by design.

7. **XGB Base Model Experiment**
   - Script: `src/experiments/xgb_eq_opp_exp.py`  
   - Description: Experiments using XGBoost.

**Note**: Results from the experiments (except for the audit experiment) are automatically saved for later analysis.

---

## Main PROMIS Methods

The core PROMIS method implementations are located in:

- `src/methods/models/optimization_model.py`
- `src/methods/optimization/promis_app.py`
- `src/methods/optimization/promis_opt.py`

---

## Analysis

Use the following notebook for analyzing and comparing results:

- `src/analysis/results_analysis.ipynb`  
  This notebook includes code for inference, computation of scores, and generating plots.

---

## Example Use Cases of PROMIS

Two additional notebooks demonstrate PROMIS under different approaches:

- `src/promis_decision_bounraries.ipynb` (decision boundaries adjustment approach)
- `src/promis_direct_flips.ipynb` (direct application of flips approach)  

---

## Requirements and Setup

1. **Python Version**  
   This codebase uses **Python 3.10.12**.

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt

