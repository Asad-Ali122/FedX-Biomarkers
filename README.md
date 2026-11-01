<div align="center">

# 🦶 FedX-Biomarker

### Trustworthy Federated Discovery of Digital Gait Biomarkers for Parkinson's Disease

*Interpretable · Consistent across sites · Fair · Calibrated, trained in one federated objective*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.13-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/data-PhysioNet%20Gait%20in%20PD-8A2BE2.svg)](https://physionet.org/content/gaitpdb/1.0.0/)
[![Workshop](https://img.shields.io/badge/IEEE%20BIBM-Trustworthy%20AI%20for%20Biomedical%20Discovery-orange.svg)](#citation)

</div>

---

## ✨ Why FedX-Biomarker?

Gait recordings that could reveal Parkinson's disease (PD) biomarkers are spread across hospitals that cannot share raw data. Federated learning (FL) solves the sharing problem, but a plain FL model is still a black box. It may treat patient groups unequally and give over- or under-confident probabilities.

**FedX-Biomarker** trains *trust* into the federation itself. The three contributions below act on **one unified objective**:

| | Contribution | What it does |
|---|---|---|
| **C1** | **BioGait-Net** | Interpretable-by-design network: a sensor-attention gate over the insole layout, a weight-shared bilateral foot encoder with an asymmetry stream, temporal attention, and a knowledge-guided stride-dynamics branch. It also has an auxiliary Hoehn & Yahr head. |
| **C2** | **FedCAA**, consensus-attribution aggregation | Clients upload a 36-value attribution profile together with their weights. The server weights clients by peer agreement and broadcasts a **global biomarker profile** that acts as a consensus regulariser. The overhead is under 0.1 %. |
| **C3** | **Privacy-preserving trust layer** | A group-loss dispersion penalty across sex × age groups, plus **federated temperature scaling** in which clients share only likelihood curves. |

$$
\mathcal{L}_k=\mathcal{L}_{\text{BCE}}+\lambda_s\mathcal{L}_{\text{sev}}+\lambda_c\big[1-\rho(\bar{\alpha}_k,A_g)\big]+\lambda_f\,\mathrm{SD}_{g}\big(\mathcal{L}^{(g)}\big)
$$


## 📊 Headline results

The results come from the PhysioNet *Gait in Parkinson's Disease* database, with its three source studies (Ga, Ju, Si) used as three natural federated clients: 165 subjects and 306 recordings. All metrics are at subject level, over 3 seeds × 5 folds, and the comparison includes **federated methods only**.

| Method | AUROC | BACC | MCC | ECE ↓ | Worst-run AUROC | Unseen-site AUROC (LOSO) |
|---|---|---|---|---|---|---|
| FedAvg | 0.829 ± 0.103 | 0.710 | 0.441 | 0.175 | 0.508 | 0.746 |
| FedProx | 0.829 ± 0.104 | 0.710 | 0.441 | 0.175 | 0.504 | 0.751 |
| FedAdam | 0.844 ± 0.058 | 0.715 | 0.442 | 0.179 | 0.676 | 0.763 |
| q-FFL | 0.842 ± 0.056 | 0.717 | 0.465 | 0.175 | 0.714 | 0.763 |
| **FedX-Biomarker** | **0.875 ± 0.051** | **0.786** | **0.578** | **0.156** | **0.769** | **0.869** |

🔬 **Biomarker discovery.** The two stride features the federated model relied on most were **stance-phase proportion** and **swing-time asymmetry**. These were also the two largest PD-vs-control effects measured in the cohort (Cliff's δ = 0.48 and 0.43).

## 🗂️ Repository layout

```
FedX-Biomarker/
├── fedx_biomarker.py                 # full pipeline as a script (CLI)
├── notebooks/
│   ├── FedX_Biomarker_CV.ipynb       # main protocol: subject-level 5-fold CV + LOSO check (paper results)
│   └── FedX_Biomarker_LOSO.ipynb     # leave-one-study-out as primary protocol (4 sub-clients)
├── scripts/
│   └── make_paper_figures.py         # copies/renames result figures for the paper
├── docs/
│   └── FIGURE_PROMPTS.md             # captions + detailed AI prompts for every paper figure
├── data/                             # put the PhysioNet folder here (not tracked)
├── results/                          # tables/*.csv|.tex and figures/*.png are written here
├── requirements.txt · environment.yml · CITATION.cff · LICENSE
```

## 🚀 Quick start

```bash
# 1) environment (for users in China, the Tsinghua mirror is fastest)
conda env create -f environment.yml && conda activate fedx
#   or: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) data: download "Gait in Parkinson's Disease" v1.0.0 from PhysioNet and unzip it to
#    data/gait-in-parkinsons-disease-1.0.0/   (must contain *.txt recordings + demographics.txt)

# 3) smoke test (≈ 5–10 min on CPU)
python fedx_biomarker.py --data_dir data/gait-in-parkinsons-disease-1.0.0 --out_dir results_quick --quick

# 4) full paper run (≈ 2 h on a single GPU)
python fedx_biomarker.py --data_dir data/gait-in-parkinsons-disease-1.0.0 --out_dir results

# 5) paper figures
python scripts/make_paper_figures.py --results results --dest paper_figures
```

You can also open `notebooks/FedX_Biomarker_CV.ipynb`, set `DATA_DIR` in the first code cell, and run all cells. No pretrained weights or internet downloads are needed; every model is trained from scratch.

## ⚙️ Key settings (`CFG` in the first code cell)

| Key | Default | Meaning |
|---|---|---|
| `seeds` | `[42, 123, 2026]` | repeated runs used for mean, SD and worst-case statistics |
| `rounds`, `local_epochs` | `30`, `1` | federated rounds and local epochs per round |
| `lam_cons`, `lam_fair`, `tau` | `0.5`, `0.5`, `2.0` (validation tuning selected `0.2`, `0.2` in the paper) | C2 consensus weight, C3 fairness weight, FedCAA temperature |
| `baseline_backbone` | `'cnn'` | `'biogait'` gives every baseline the same network as ours |
| `baseline_ts` | `False` | `True` also gives the baselines federated temperature scaling |
| `tune` | `True` | validation-only hyperparameter grid for **every** method |

## 📑 Outputs

The pipeline writes 14 tables (`T0`–`T14`, as CSV and LaTeX) and 14 figures, including:
- the main comparison and worst-case robustness
- per-site results and leave-one-study-out
- fairness and calibration
- severity
- the cumulative ablation (FedAvg → +C1 → +C2a → +C2b → +C3)
- biomarker concordance
- paired bootstrap significance
- communication cost

## 🔒 Privacy note

Raw gait signals never leave a client. Each round a client shares its model weights and a 36-float aggregate attribution profile (144 bytes). At the end it shares a 93-point likelihood curve. Experiments simulate the federation on one machine. For deployment, combine the pipeline with secure aggregation or differential privacy.

When you use the data, please also cite the PhysioNet dataset (Hausdorff, *Gait in Parkinson's Disease*, v1.0.0) and the PhysioNet platform (Pollard et al., *Nature Health*, 2026).

## 📄 License

The code is released under the MIT License. The dataset is subject to its PhysioNet license terms.
