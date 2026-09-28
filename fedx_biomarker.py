#!/usr/bin/env python
"""FedX-Biomarker: trustworthy federated discovery of digital gait biomarkers (script version).

Exported from notebooks/FedX_Biomarker_CV.ipynb so the full pipeline runs without Jupyter.

Usage:
    python fedx_biomarker.py --data_dir /path/to/gait-in-parkinsons-disease-1.0.0 --out_dir results
    python fedx_biomarker.py --data_dir ... --quick            # 2-fold, 3-round smoke test
"""
import argparse, os, sys
_p = argparse.ArgumentParser(description="FedX-Biomarker experiment runner")
_p.add_argument("--data_dir", required=True, help="folder with GaCo01_01.txt ... and demographics.txt")
_p.add_argument("--out_dir", default="results", help="where tables/ and figures/ are written")
_p.add_argument("--quick", action="store_true", help="fast smoke test (2 folds, 3 rounds, 1 seed)")
_a = _p.parse_args()
os.environ["GAIT_DIR"], os.environ["GAIT_OUT"] = _a.data_dir, _a.out_dir
os.environ["GAIT_QUICK"] = "1" if _a.quick else "0"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: _plt.close("all")
def display(x):
    print(x)


# # FedX-Biomarker: Consensus-Attributed, Fairness-Aware Federated Learning for Trustworthy Digital Gait Biomarker Discovery in Parkinson's Disease
# 
# **Target venue:** 2nd International Workshop on Trustworthy AI for Biomedical Discovery (privacy-preserving, federated, ethical and responsible AI for biomedicine).
# 
# **Dataset:** PhysioNet *Gait in Parkinson's Disease* (Hausdorff, v1.0.0): 93 PD patients and 73 healthy controls, 16 VGRF insole sensors + 2 foot totals, 100 Hz, recorded in three independent studies (**Ga**, **Ju**, **Si**).
# These three studies act as three **natural federated clients** (real cross-site heterogeneity, no synthetic partitioning).
# 
# ### One unified framework, three contributions
# The single training objective optimised on every client is
# 
# $$\mathcal{L}_k = \underbrace{\mathcal{L}_{\text{BCE}} + \lambda_s \mathcal{L}_{\text{sev}}}_{\text{C1: BioGait-Net}} + \underbrace{\lambda_c\,[1-\rho(\bar\alpha_k, A_g)]}_{\text{C2: consensus}} + \underbrace{\lambda_f\,\mathrm{SD}_{g\in\mathcal{G}}(\mathcal{L}_{\text{BCE}}^{(g)})}_{\text{C3: fairness}}$$
# 
# | # | Contribution | What it does | Trust dimension |
# |---|---|---|---|
# | **C1** | **BioGait-Net**, an interpretable-by-design, biomechanics-aware network | Sensor-Attention Gate over the insole topology, a shared (Siamese) left/right foot encoder with an explicit bilateral-asymmetry stream, temporal attention pooling, and a gated fusion with a knowledge-guided stride-dynamics branch (stride/swing time and CV, per Hausdorff) and an auxiliary severity (Hoehn & Yahr) head | Interpretability, biological meaning |
# | **C2** | **FedCAA**, Federated Consensus-Attribution Aggregation | Each client uploads weights **plus** a 36-dim attribution profile (sensor gate + clinical grad×input). The server weights clients by *peer attribution agreement*, builds a **global digital-biomarker profile** $A_g$ and broadcasts it as a consensus regulariser | Reliability, reproducibility, cross-site generalisability |
# | **C3** | **Fair & calibrated trust layer** | Group-loss-dispersion regulariser over sex × age groups, plus **federated temperature scaling** (clients share only NLL curves, never logits or data) | Fairness, calibration, responsible AI |
# 
# **Evaluation protocol (federated scope only):**
# * **Comparison set = federated methods only**, since only FL keeps raw data on site and shares nothing but model updates: **FedAvg, FedProx, FedAdam (FedOpt) and q-FFL (fairness-aware FL)**, all used as in their original papers (plain averaging or their own server rules, no calibration) on a standard 1D-CNN backbone (`CFG['baseline_backbone']`).
# * **Cumulative ablation starting from traditional FedAvg:** A0 FedAvg → A1 + BioGait-Net (C1) → A2 + FedCAA weighting (C2a) → A3 + consensus regulariser (C2b) → A4 + fairness and federated temperature scaling (C3) = FedX. Every component that is not yet added falls back to the traditional FedAvg behaviour.
# * **Average and worst case:** several random seeds × subject-level stratified 5-fold CV inside each client (no subject leakage), reporting mean ± SD, **worst fold**, **worst client**, mean rank, win rate, paired bootstrap tests, and **leave-one-study-out** generalisation.
# * **Fair tuning:** each method's own hyperparameters (FedProx μ, FedAdam η, q-FFL q, FedX λ_c / λ_f) are chosen on **validation subjects only**, never on test subjects.
# 
# > Everything is trained **from scratch**. No pretrained weights and no internet downloads are needed; only `numpy, pandas, scipy, scikit-learn, matplotlib, torch`.

# ## 0. Setup and configuration

# If a package is missing (China mirror):
# !pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch scikit-learn scipy pandas matplotlib
import os, re, glob, time, math, copy, random, json, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, brier_score_loss, roc_curve, mean_absolute_error)
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 60)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

# ------------------------------------------------------------------ paths
DATA_DIR = os.environ.get('GAIT_DIR', '/home/settings/gait_parkinson/gait-in-parkinsons-disease-1.0.0')
OUT_DIR  = os.environ.get('GAIT_OUT', './results_fedx_biomarker')
FIG_DIR, TAB_DIR = os.path.join(OUT_DIR, 'figures'), os.path.join(OUT_DIR, 'tables')
for d in (OUT_DIR, FIG_DIR, TAB_DIR): os.makedirs(d, exist_ok=True)

QUICK = os.environ.get('GAIT_QUICK', '0') == '1'   # set True for a fast smoke test

CFG = dict(
    seed=42,
    fs=100, ds=2,                 # 100 Hz raw -> 50 Hz for the deep branch (stride features use 100 Hz)
    win_s=10.0, hop_s=5.0,        # 10-s windows, 50 % overlap
    trim_s=3.0,                   # drop gait-initiation transient
    n_folds=5, val_frac=0.2,
    rounds=30, local_epochs=1,    # federated rounds x local epochs
    batch=64, lr=2e-3, wd=1e-4, width=64, emb=32,
    lam_sev=0.3,                  # C1 auxiliary severity head
    lam_cons=0.5, tau=2.0,        # C2 consensus regulariser and aggregation temperature
    lam_fair=0.5,                 # C3 group-loss dispersion
    prox_mu=0.01,                 # FedProx baseline (tuned on validation if tune=True)
    fedadam_lr=1e-2,              # FedAdam server learning rate
    qffl_q=1.0,                   # q-FFL fairness exponent
    baseline_backbone='cnn',      # 'cnn' = standard 1D-CNN for baselines (traditional); 'biogait' = same backbone as ours
    baseline_ts=False,            # False = baselines as in their papers (no calibration); True = give them our fed-TS too
    seeds=[42, 123, 2026],        # repeated runs -> average / worst-case statistics
    tune=True,                    # validation-only hyperparameter selection for every FL method
    run_component_ablation=False, # optional supplementary leave-one-out of C1 sub-modules
    age_cut=65,                   # age groups for fairness (<65 vs >=65)
    profile_n=1024,               # windows per client used to compute attribution profiles
    run_loso=True, n_boot=1000,
)
if QUICK:
    CFG.update(rounds=3, n_folds=2, n_boot=100, seeds=[42], tune=False)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    torch.set_num_threads(max(1, os.cpu_count() or 1))
else:
    torch.backends.cudnn.benchmark = True

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG['seed'])
RNG = np.random.default_rng(CFG['seed'])
CLIENTS = ['Ga', 'Ju', 'Si']
print('torch', torch.__version__, '| device:', DEVICE, '| QUICK:', QUICK)
print('DATA_DIR:', DATA_DIR, '| exists:', os.path.isdir(DATA_DIR))

# ## 1. Demographics and clinical labels
# Subjects are matched to recordings via `(Study, Group, Subjnum)`, which is robust to ID typos in the sheet (e.g. `Juc010`). Height is recorded in m for Ga/Si and cm for Ju, so it is harmonised.

def load_demographics(data_dir):
    p_txt, p_xls = os.path.join(data_dir, 'demographics.txt'), os.path.join(data_dir, 'demographics.xls')
    if os.path.exists(p_txt):
        with open(p_txt, 'r', encoding='latin-1') as f:
            lines = [l.rstrip('\r\n') for l in f]
        header = [h.strip() for h in lines[0].split('\t') if h.strip()]
        rows = []
        for l in lines[1:]:
            parts = [x.strip() for x in l.split('\t')]
            if not parts or not parts[0]: continue
            rows.append((parts + [''] * len(header))[:len(header)])
        df = pd.DataFrame(rows, columns=header)
    elif os.path.exists(p_xls):
        df = pd.read_excel(p_xls, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        df = df[[c for c in df.columns if c and not c.startswith('Unnamed')]].dropna(subset=['ID'])
    else:
        raise FileNotFoundError('demographics.txt / .xls not found in ' + data_dir)
    num_cols = [c for c in df.columns if c not in ('ID', 'Study')]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].replace({'': np.nan, 'NaN': np.nan}), errors='coerce')
    df['Study'] = df['Study'].str.strip()
    df = df.dropna(subset=['Group', 'Subjnum'])
    df['key'] = df.apply(lambda r: f"{r.Study}_{int(r.Group)}_{int(r.Subjnum)}", axis=1)
    df['label'] = (df['Group'] == 1).astype(int)                       # 1 = PD, 0 = control
    df['severity'] = np.where(df.label == 1, df['HoehnYahr'], 0.0)       # H&Y (controls = 0)
    df['Height'] = np.where(df['Height'] > 3, df['Height'] / 100, df['Height'])
    df['sex'] = df['Gender'].astype('Int64')                             # 1 = male, 2 = female
    df['age_grp'] = (df['Age'] >= CFG['age_cut']).astype(int)
    df['fair_grp'] = ((df['sex'] == 2).astype(int) * 2 + df['age_grp']).astype(int)   # 4 sex x age groups
    return df.reset_index(drop=True)

demo = load_demographics(DATA_DIR)
print(demo.shape); display(demo.head())

def cohort_table(d):
    rows = []
    for st in CLIENTS + ['All']:
        s = d if st == 'All' else d[d.Study == st]
        for lab, nm in [(1, 'PD'), (0, 'Control')]:
            x = s[s.label == lab]
            rows.append({'Study': st, 'Group': nm, 'N': len(x),
                         'Age (mean±sd)': f"{x.Age.mean():.1f}±{x.Age.std():.1f}",
                         'Male %': f"{100*(x.sex==1).mean():.0f}",
                         'H&Y': f"{x.HoehnYahr.mean():.2f}±{x.HoehnYahr.std():.2f}" if lab == 1 else '-',
                         'UPDRS': f"{x.UPDRS.mean():.1f}±{x.UPDRS.std():.1f}" if lab == 1 else '-',
                         'TUG (s)': f"{x.TUAG.mean():.1f}±{x.TUAG.std():.1f}"})
    return pd.DataFrame(rows)
T_cohort = cohort_table(demo); T_cohort.to_csv(f'{TAB_DIR}/T1_cohort.csv', index=False); display(T_cohort)

# ## 2. Signal processing: VGRF, centre of pressure, gait events, and stride-dynamics features
# * Forces are normalised by a per-recording body-weight proxy (mean of L+R totals), so no demographics leak into the signal.
# * COP per foot uses the insole sensor coordinates from `format.txt`. The left foot is mirrored so that both feet share one anatomical frame, which lets the two feet share encoder weights.
# * Heel-strike and toe-off events come from the total force of each foot and give stride, stance and swing times. These are the classical Hausdorff gait-rhythm biomarkers.

SENSOR_XY_L = np.array([[-500,-800],[-700,-400],[-300,-400],[-700,0],[-300,0],[-700,400],[-300,400],[-500,800]], float)
FOOT_XY = SENSOR_XY_L.copy(); FOOT_XY[:, 0] *= -1          # common foot frame (x: 300 medial ... 700 lateral)
FOOT_CH = [f'S{i}' for i in range(1, 9)] + ['Total', 'COPx', 'COPy']
CH_NAMES = ['L-' + c for c in FOOT_CH] + ['R-' + c for c in FOOT_CH]     # 22 deep channels
SENSOR_REGION = {'S1': 'heel', 'S2': 'rear-mid', 'S3': 'rear-mid', 'S4': 'midfoot', 'S5': 'midfoot',
                 'S6': 'forefoot', 'S7': 'forefoot', 'S8': 'toe'}
CLIN_NAMES = ['L_stride_mean', 'L_stride_CV', 'L_swing_mean', 'L_swing_CV', 'L_stance_pct',
              'R_stride_mean', 'R_stride_CV', 'R_swing_mean', 'R_swing_CV', 'R_stance_pct',
              'swing_asym', 'cadence', 'peakF_mean', 'peakF_CV']
N_CH, N_CLIN = len(CH_NAMES), len(CLIN_NAMES)
FNAME_RE = re.compile(r'^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$', re.I)

def parse_name(fp):
    m = FNAME_RE.match(os.path.basename(fp))
    if not m: return None
    return m.group(1).capitalize(), (1 if m.group(2).lower() == 'pt' else 2), int(m.group(3)), int(m.group(4))

def read_record(fp):
    try:
        df = pd.read_csv(fp, sep=r'\s+', header=None, engine='c')
    except Exception:
        df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python', on_bad_lines='skip')
    if df.shape[1] < 19: return None
    a = df.iloc[:, :19].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
    return a[np.isfinite(a).all(1)]

def movavg(x, k):
    return np.convolve(x, np.ones(k) / k, mode='same')

def clean_runs(c, minlen):
    c = c.astype(bool).copy()
    for _ in range(2):
        idx = np.flatnonzero(np.diff(c.astype(np.int8))) + 1
        b = np.r_[0, idx, len(c)]
        for s, e in zip(b[:-1], b[1:]):
            if e - s < minlen and s > 0 and e < len(c):
                c[s:e] = ~c[s]
    return c

def stride_table(ft, fs):
    # rows: [heel-strike index, stride time, stance time, swing time, peak force]
    f = movavg(ft, 5)
    lo, hi = np.percentile(f, 5), np.percentile(f, 95)
    c = clean_runs(f > lo + 0.2 * (hi - lo), int(0.08 * fs))
    d = np.diff(c.astype(np.int8))
    on, off = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    rows = []
    for i in range(len(on) - 1):
        s, e = on[i], on[i + 1]
        o = off[(off > s) & (off < e)]
        if len(o) != 1: continue
        stride, stance = (e - s) / fs, (o[0] - s) / fs
        swing = stride - stance
        if not (0.6 <= stride <= 2.5) or stance <= 0.1 or swing <= 0.05: continue
        rows.append((s, stride, stance, swing, ft[s:o[0]].max()))
    return np.array(rows) if rows else np.zeros((0, 5))

def clin_features(stL, stR, s0, s1, bw):
    out, sub = [], {}
    for side, st in (('L', stL), ('R', stR)):
        m = st[(st[:, 0] >= s0) & (st[:, 0] < s1)] if len(st) else st
        if len(m) >= 3:
            sr, sn, sw, pk = m[:, 1], m[:, 2], m[:, 3], m[:, 4] / bw
            f = [sr.mean(), 100 * sr.std() / sr.mean(), sw.mean(), 100 * sw.std() / sw.mean(), 100 * (sn / sr).mean()]
        else:
            f, pk = [np.nan] * 5, np.array([])
        out += f; sub[side] = (f, pk)
    swL, swR = sub['L'][0][2], sub['R'][0][2]
    asym = 100 * abs(np.log(swL / swR)) if np.isfinite(swL) and np.isfinite(swR) and swL > 0 and swR > 0 else np.nan
    sm = np.nanmean([sub['L'][0][0], sub['R'][0][0]])
    cad = 120.0 / sm if np.isfinite(sm) else np.nan
    pk = np.r_[sub['L'][1], sub['R'][1]]
    pm, pc = (pk.mean(), 100 * pk.std() / pk.mean()) if len(pk) >= 3 else (np.nan, np.nan)
    return np.array(out + [asym, cad, pm, pc], np.float32)

def preprocess_record(a, fs=100):
    a = a[int(CFG['trim_s'] * fs):]
    L, R = np.clip(a[:, 1:9], 0, None), np.clip(a[:, 9:17], 0, None)
    Lt, Rt = np.clip(a[:, 17], 0, None), np.clip(a[:, 18], 0, None)
    bw = float(np.mean(Lt + Rt))
    if bw <= 1e-6: return None
    def cop(S):
        den = S.sum(1); valid = den > 0.05 * bw
        x = np.where(valid, (S @ FOOT_XY[:, 0]) / np.maximum(den, 1e-6), 500.0)
        y = np.where(valid, (S @ FOOT_XY[:, 1]) / np.maximum(den, 1e-6), 0.0)
        return (x - 500) / 400, y / 800
    cxL, cyL = cop(L); cxR, cyR = cop(R)
    deep = np.concatenate([L / bw, Lt[:, None] / bw, cxL[:, None], cyL[:, None],
                           R / bw, Rt[:, None] / bw, cxR[:, None], cyR[:, None]], 1).T.astype(np.float32)
    return deep, stride_table(Lt, fs), stride_table(Rt, fs), bw

# ## 3. Build the windowed dataset (cached)

def build_dataset():
    fs, ds = CFG['fs'], CFG['ds']
    win, hop = int(CFG['win_s'] * fs), int(CFG['hop_s'] * fs)
    files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, '*.txt')) if FNAME_RE.match(os.path.basename(f)))
    keys = set(demo.key)
    X, C, meta, rec, skipped = [], [], [], [], []
    t0 = time.time()
    for n, fp in enumerate(files):
        st, grp, num, walk = parse_name(fp); key = f'{st}_{grp}_{num}'
        if key not in keys: skipped.append((os.path.basename(fp), 'not in demographics')); continue
        a = read_record(fp)
        if a is None or len(a) < fs * (CFG['trim_s'] + CFG['win_s']): skipped.append((os.path.basename(fp), 'too short/bad')); continue
        out = preprocess_record(a, fs)
        if out is None: skipped.append((os.path.basename(fp), 'zero force')); continue
        deep, stL, stR, bw = out
        N = deep.shape[1]
        for s0 in range(0, N - win + 1, hop):
            seg = deep[:, s0:s0 + win].reshape(N_CH, win // ds, ds).mean(-1)
            X.append(seg); C.append(clin_features(stL, stR, s0, s0 + win, bw))
            meta.append((key, st, walk, os.path.basename(fp)))
        rec.append(dict(key=key, walk=walk, file=os.path.basename(fp), n_strides_L=len(stL), n_strides_R=len(stR),
                        **dict(zip(CLIN_NAMES, clin_features(stL, stR, 0, 10**9, bw)))))
        if (n + 1) % 50 == 0: print(f'  {n+1}/{len(files)} files  ({time.time()-t0:.0f}s)')
    X = np.stack(X).astype(np.float32); C = np.stack(C).astype(np.float32)
    W = pd.DataFrame(meta, columns=['key', 'study', 'walk', 'file'])
    return X, C, W, pd.DataFrame(rec), skipped

cache = os.path.join(OUT_DIR, f"cache_w{CFG['win_s']}_h{CFG['hop_s']}_ds{CFG['ds']}.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    X, C = z['X'], z['C']
    W = pd.read_csv(cache.replace('.npz', '_meta.csv')); REC = pd.read_csv(cache.replace('.npz', '_rec.csv'))
    print('Loaded cache', X.shape)
else:
    X, C, W, REC, skipped = build_dataset()
    np.savez(cache, X=X, C=C)
    W.to_csv(cache.replace('.npz', '_meta.csv'), index=False); REC.to_csv(cache.replace('.npz', '_rec.csv'), index=False)
    print('Skipped files:', len(skipped), skipped[:10])

W = W.merge(demo[['key', 'label', 'severity', 'sex', 'age_grp', 'fair_grp']], on='key', how='left')
Y   = W['label'].to_numpy(np.float32)
SEV = W['severity'].fillna(0).to_numpy(np.float32)
GRP = W['fair_grp'].to_numpy(np.int64)
Xt  = torch.from_numpy(X)
demo_s = demo[demo.key.isin(W.key.unique())].reset_index(drop=True)
print(f'Windows: {X.shape} | clinical feats: {C.shape} | subjects: {W.key.nunique()} | recordings: {W.file.nunique()}')
print(W.groupby(['study', 'label']).agg(windows=('key', 'size'), subjects=('key', 'nunique')))
print('NaN rate of stride features per window: %.1f%%' % (100 * np.isnan(C).any(1).mean()))

# ### 3.1 Data overview figure and a ground-truth biomarker table
# Subject-level PD vs control statistics on the usual-walk records (Mann-Whitney U, Cliff's δ, BH-FDR). This table is the **clinical reference** used later to test whether the federated model's attributions agree with known gait biomarkers.

def cliffs_delta(a, b):
    U = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    return 2 * U / (len(a) * len(b)) - 1

def bh_fdr(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)

rec_u = REC[REC.walk == 1] if (REC.walk == 1).any() else REC
subj_feat = rec_u.groupby('key')[CLIN_NAMES].mean().reset_index().merge(demo_s[['key', 'label', 'Study']], on='key')
bm = []
for f in CLIN_NAMES:
    a = subj_feat.loc[subj_feat.label == 1, f].dropna(); b = subj_feat.loc[subj_feat.label == 0, f].dropna()
    if len(a) > 2 and len(b) > 2:
        p = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
        bm.append(dict(feature=f, PD=f'{a.mean():.3f}±{a.std():.3f}', Control=f'{b.mean():.3f}±{b.std():.3f}',
                       cliffs_delta=cliffs_delta(a, b), p=p))
T_bm = pd.DataFrame(bm); T_bm['q_FDR'] = bh_fdr(T_bm.p.values); T_bm['abs_delta'] = T_bm.cliffs_delta.abs()
T_bm = T_bm.sort_values('abs_delta', ascending=False)
T_bm.to_csv(f'{TAB_DIR}/T2_reference_gait_biomarkers.csv', index=False); display(T_bm)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
cnt = W.groupby(['study', 'label']).key.nunique().unstack()
cnt.columns = ['Control', 'PD']; cnt.plot.bar(ax=axs[0], color=['#4C72B0', '#DD8452']); axs[0].set_title('Subjects per client'); axs[0].set_xlabel('')
for lab, col in [(0, '#4C72B0'), (1, '#DD8452')]:
    axs[1].hist(demo_s[demo_s.label == lab].Age, bins=15, alpha=.6, color=col, label=['Control', 'PD'][lab])
axs[1].set_title('Age distribution'); axs[1].legend()
top = T_bm.feature.iloc[0]
sns_data = [subj_feat[(subj_feat.Study == s) & (subj_feat.label == l)][top].dropna() for s in CLIENTS for l in (0, 1)]
axs[2].boxplot(sns_data); axs[2].set_xticks(range(1, 7)); axs[2].set_xticklabels([f'{s}-{"Co" if l == 0 else "PD"}' for s in CLIENTS for l in (0, 1)])
axs[2].set_title(f'Cross-site heterogeneity: {top}')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F1_data_overview.png', bbox_inches='tight'); plt.show()

# example raw signals with detected gait events
fig, axs = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
for ax, lab in zip(axs, [0, 1]):
    fp = os.path.join(DATA_DIR, W[W.label == lab].file.iloc[0])
    a = read_record(fp); deep, stL, stR, bw = preprocess_record(a)
    t = np.arange(deep.shape[1]) / 100; sl = slice(0, 1000)
    ax.plot(t[sl], deep[8, sl], label='Left total / BW'); ax.plot(t[sl], deep[19, sl], label='Right total / BW', alpha=.8)
    hs = stL[stL[:, 0] < 1000, 0].astype(int); ax.plot(t[hs], deep[8, hs], 'kv', ms=5, label='L heel-strike')
    ax.set_title(f"{os.path.basename(fp)} ({'PD' if lab else 'Control'})"); ax.legend(loc='upper right', fontsize=8)
axs[-1].set_xlabel('time (s)'); plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F2_signals_events.png', bbox_inches='tight'); plt.show()

# ## 4. C1: BioGait-Net (interpretable-by-design)
# * **Sensor-Attention Gate (SAG):** a softmax distribution α over the 11 channels of each foot (8 sensors, total, COPx, COPy) computed from per-window statistics. α is directly an anatomical importance map.
# * **Shared Siamese foot encoder:** depthwise-separable dilated temporal convolutions with GroupNorm (safe for non-IID FL), shared across both feet.
# * **Bilateral asymmetry stream:** fuses `[h_L + h_R, |h_L − h_R|]`. Gait asymmetry is a known PD marker.
# * **Temporal attention pooling:** β shows *when* within the gait cycle the model looks.
# * **Knowledge-guided stride branch + gated fusion:** the gate g gives each window's reliance on deep vs clinical features.
# * **Heads:** PD logit and Hoehn & Yahr severity.

class SensorGate(nn.Module):
    def __init__(self, C, enabled=True):
        super().__init__(); self.C, self.enabled = C, enabled
        h = max(16, 2 * C)
        self.mlp = nn.Sequential(nn.Linear(2 * C, h), nn.GELU(), nn.Linear(h, C))
    def forward(self, x):
        if not self.enabled:
            return x, torch.full((x.size(0), self.C), 1.0 / self.C, device=x.device)
        s = torch.cat([x.mean(-1), x.std(-1)], 1)
        a = torch.softmax(self.mlp(s), 1)
        return x * a.unsqueeze(-1) * self.C, a

class DSBlock(nn.Module):
    def __init__(self, ch, k=7, d=1):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, k, padding=d * (k - 1) // 2, dilation=d, groups=ch)
        self.pw = nn.Conv1d(ch, ch, 1); self.gn = nn.GroupNorm(8, ch); self.drop = nn.Dropout(0.1)
    def forward(self, x):
        return x + self.drop(F.gelu(self.gn(self.pw(self.dw(x)))))

class Encoder(nn.Module):
    def __init__(self, in_ch, width, use_gate=True):
        super().__init__()
        self.gate = SensorGate(in_ch, use_gate)
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 7, padding=3), nn.GroupNorm(8, width), nn.GELU(), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(DSBlock(width, 7, 1), nn.MaxPool1d(2), DSBlock(width, 7, 2), nn.MaxPool1d(2),
                                    DSBlock(width, 7, 4), DSBlock(width, 7, 8))
    def forward(self, x):
        x, a = self.gate(x)
        return self.blocks(self.stem(x)), a

class TemporalAttnPool(nn.Module):
    def __init__(self, w):
        super().__init__(); self.proj = nn.Conv1d(w, w // 2, 1); self.v = nn.Conv1d(w // 2, 1, 1)
    def forward(self, h):
        b = torch.softmax(self.v(torch.tanh(self.proj(h))).squeeze(1), -1)
        return (h * b.unsqueeze(1)).sum(-1), b

class BioGaitNet(nn.Module):
    def __init__(self, n_foot=11, n_clin=N_CLIN, width=CFG['width'], d=CFG['emb'],
                 use_gate=True, use_clin=True, bilateral=True):
        super().__init__()
        self.n_foot, self.use_gate, self.use_clin, self.bilateral = n_foot, use_gate, use_clin, bilateral
        if bilateral:
            self.enc = Encoder(n_foot, width, use_gate)
            self.fuse = nn.Sequential(nn.Conv1d(2 * width, width, 1), nn.GroupNorm(8, width), nn.GELU())
        else:
            self.enc = Encoder(2 * n_foot, width, use_gate)
        self.pool = TemporalAttnPool(width)
        self.pdz = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(0.2))
        if use_clin:
            self.clin = nn.Sequential(nn.Linear(n_clin, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d), nn.GELU())
            self.gatef = nn.Linear(2 * d, d)
        self.cls, self.sev = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, x, c):
        B = x.size(0)
        if self.bilateral:
            h, a = self.enc(torch.cat([x[:, :self.n_foot], x[:, self.n_foot:]], 0))     # shared weights
            hl, hr = h[:B], h[B:]
            alpha = torch.cat([a[:B], a[B:]], 1) / 2
            h = self.fuse(torch.cat([hl + hr, (hl - hr).abs()], 1))
        else:
            h, alpha = self.enc(x)
        z, beta = self.pool(h)
        zd = self.pdz(z)
        if self.use_clin:
            zc = self.clin(c)
            g = torch.sigmoid(self.gatef(torch.cat([zd, zc], 1)))
            f, gm = g * zd + (1 - g) * zc, g.mean(1)
        else:
            f, gm = zd, torch.ones(B, device=x.device)
        return dict(logit=self.cls(f).squeeze(1), sev=self.sev(f).squeeze(1), alpha=alpha, beta=beta, gate=gm)

_m = BioGaitNet()
N_PARAMS = sum(p.numel() for p in _m.parameters())
_o = _m(torch.randn(4, N_CH, X.shape[-1]), torch.randn(4, N_CLIN))
print(f'BioGait-Net params: {N_PARAMS:,}  | alpha {tuple(_o["alpha"].shape)}  beta {tuple(_o["beta"].shape)}')

# ## 5. C2 and C3: federated engine (FedCAA consensus aggregation, fairness regulariser, federated temperature scaling)
# **FedCAA server step (round t):**
# 1. Client k returns weights θ_k, sample count n_k and the profile p_k = [mean α (22) ‖ |grad×input| on stride features (14)], each normalised.
# 2. Peer agreement s_k = ρ(p_k, mean_{j≠k} p_j) (Pearson; leave-one-out, so a client cannot vote for itself).
# 3. Weights w_k ∝ n_k · exp(τ s_k); θ_g = Σ w_k θ_k; global biomarker profile A_g = Σ w_k p_k.
# 4. A_g is broadcast. In the next round each client adds λ_c[1 − ρ(ᾱ_batch, A_g)].
# 
# Only θ plus a 36-float vector is shared, and the vector is aggregate statistics rather than per-subject data.

def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0

def t_pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-6)

def get_batch(ib, Cz):
    it = torch.as_tensor(ib)
    return (Xt[it].to(DEVICE, non_blocking=True), torch.from_numpy(Cz[ib]).to(DEVICE),
            torch.from_numpy(Y[ib]).to(DEVICE), torch.from_numpy(SEV[ib]).to(DEVICE), torch.from_numpy(GRP[ib]).to(DEVICE))

def train_local(model, idx, Cz, epochs, A_glob=None, lam_cons=0.0, lam_fair=0.0, prox_ref=None, mu=0.0):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    yl = Y[idx]; pos_w = torch.tensor((yl == 0).sum() / max((yl == 1).sum(), 1), dtype=torch.float32, device=DEVICE)
    A = torch.tensor(A_glob, dtype=torch.float32, device=DEVICE) if A_glob is not None else None
    use_cons = A is not None and lam_cons > 0 and model.use_gate
    for _ in range(epochs):
        perm = RNG.permutation(idx)
        for i in range(0, len(perm), CFG['batch']):
            ib = perm[i:i + CFG['batch']]
            if len(ib) < 4: continue
            x, c, y, sv, g = get_batch(ib, Cz)
            o = model(x, c)
            bce = F.binary_cross_entropy_with_logits(o['logit'], y, pos_weight=pos_w, reduction='none')
            loss = bce.mean() + CFG['lam_sev'] * F.smooth_l1_loss(o['sev'], sv)
            if lam_fair > 0:                                        # C3: group-loss dispersion
                gl = [bce[g == k].mean() for k in torch.unique(g) if (g == k).sum() >= 2]
                if len(gl) >= 2:
                    gl = torch.stack(gl); loss = loss + lam_fair * torch.sqrt(((gl - gl.mean()) ** 2).mean() + 1e-8)
            if use_cons:                                            # C2: consensus to global biomarker profile
                loss = loss + lam_cons * (1 - t_pearson(o['alpha'].mean(0), A))
            if prox_ref is not None and mu > 0:
                loss = loss + mu / 2 * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), prox_ref))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model

@torch.no_grad()
def predict(model, idx, Cz, mask_ch=None, bs=512, keep_beta=False):
    model.eval(); out = {k: [] for k in ('logit', 'sev', 'alpha', 'gate', 'beta')}
    for i in range(0, len(idx), bs):
        ib = idx[i:i + bs]
        x = Xt[torch.as_tensor(ib)].to(DEVICE)
        if mask_ch is not None and len(mask_ch):
            x[:, mask_ch, :] = x[:, mask_ch, :].mean(-1, keepdim=True)     # mean-ablation of channels
        o = model(x, torch.from_numpy(Cz[ib]).to(DEVICE))
        for k in ('logit', 'sev', 'alpha', 'gate'): out[k].append(o[k].float().cpu().numpy())
        if keep_beta: out['beta'].append(o['beta'].float().cpu().numpy())
    return {k: (np.concatenate(v) if len(v) else None) for k, v in out.items()}

def sub_idx(idx, n):
    return idx if len(idx) <= n else np.sort(RNG.choice(idx, n, replace=False))

def sensor_profile(model, idx, Cz, n=CFG['profile_n']):
    a = predict(model, sub_idx(idx, n), Cz)['alpha'].mean(0)
    return a / a.sum()

def clin_attribution(model, idx, Cz, n=CFG['profile_n'], bs=256):
    if not model.use_clin: return np.full(N_CLIN, 1.0 / N_CLIN)
    model.eval(); tot = np.zeros(N_CLIN); sel = sub_idx(idx, n)
    for i in range(0, len(sel), bs):
        ib = sel[i:i + bs]
        x = Xt[torch.as_tensor(ib)].to(DEVICE)
        c = torch.from_numpy(Cz[ib]).to(DEVICE).requires_grad_(True)
        model(x, c)['logit'].sum().backward()
        tot += (c.grad * c).abs().sum(0).detach().cpu().numpy()
    model.zero_grad(set_to_none=True)
    return tot / max(tot.sum(), 1e-12)

def full_profile(model, idx, Cz):
    return np.r_[sensor_profile(model, idx, Cz), clin_attribution(model, idx, Cz)]

def fed_aggregate(states, w):
    return {k: (sum(wi * s[k] for wi, s in zip(w, states)) if states[0][k].is_floating_point() else states[0][k].clone())
            for k in states[0]}

T_GRID = np.round(np.exp(np.linspace(np.log(0.1), np.log(10.0), 93)), 4)
def nll_curve(logits, y):
    z = logits[:, None] / T_GRID[None]
    return (np.logaddexp(0, z) - y[:, None] * z).sum(0), len(y)

def fed_temperature(curves):                       # clients share only their NLL(T) curves
    tot = sum(c[0] for c in curves)
    return float(T_GRID[int(np.argmin(tot))])

def val_loss(model, idx, Cz):
    lg = predict(model, idx, Cz)['logit']
    return float(np.mean(np.logaddexp(0, lg) - Y[idx] * lg))

# ## 6. Evaluation metrics (subject level): discrimination, calibration, fairness, severity, bootstrap

def ece_score(y, p, n_bins=10):
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1); e = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def clf_metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    return dict(AUROC=roc_auc_score(y, p) if two else np.nan, AUPRC=average_precision_score(y, p) if two else np.nan,
                ACC=accuracy_score(y, yh), BACC=balanced_accuracy_score(y, yh) if two else np.nan,
                Sens=tp / max(tp + fn, 1), Spec=tn / max(tn + fp, 1), PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1),
                F1=f1_score(y, yh, zero_division=0), MCC=matthews_corrcoef(y, yh), Kappa=cohen_kappa_score(y, yh),
                Brier=brier_score_loss(y, p), ECE=ece_score(y, p), NLL=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))

def fairness_metrics(df, attr, thr=0.5):
    r = {}
    rows = []
    for g, d in df.groupby(attr):
        y, p = d.label.values, d.prob.values; yh = p >= thr
        rows.append(dict(g=g, sel=yh.mean(), TPR=yh[y == 1].mean() if (y == 1).any() else np.nan,
                         FPR=yh[y == 0].mean() if (y == 0).any() else np.nan, acc=(yh == y).mean(),
                         auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan))
    t = pd.DataFrame(rows)
    r['DPD'] = t.sel.max() - t.sel.min(); r['EOD'] = t.TPR.max() - t.TPR.min()
    r['FPRD'] = t.FPR.max() - t.FPR.min(); r['EqOdds'] = max(r['EOD'], r['FPRD'])
    r['AUROC_gap'] = t.auc.max() - t.auc.min(); r['WorstGroupACC'] = t.acc.min()
    return r

def severity_metrics(df):
    pdd = df[df.label == 1]; r = {}
    for col in ('HoehnYahr', 'UPDRS', 'UPDRSM'):
        d = pdd[['sev', col]].dropna()
        r[f'rho_{col}'] = stats.spearmanr(d.sev, d[col]).correlation if len(d) > 5 else np.nan
    d = pdd[['sev', 'HoehnYahr']].dropna()
    r['MAE_HY'] = mean_absolute_error(d.HoehnYahr, d.sev) if len(d) else np.nan
    d = df[['prob', 'UPDRS']].dropna()
    r['rho_prob_UPDRS_all'] = stats.spearmanr(d.prob, d.UPDRS).correlation if len(d) > 5 else np.nan
    return r

def boot_ci(y, p, fn=roc_auc_score, n=CFG['n_boot'], seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])

def paired_boot_diff(y, p1, p2, fn=roc_auc_score, n=CFG['n_boot'], seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        d.append(fn(y[i], p1[i]) - fn(y[i], p2[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))

SUBJ_COLS = ['key', 'Study', 'label', 'sex', 'age_grp', 'fair_grp', 'Age', 'HoehnYahr', 'UPDRS', 'UPDRSM']
def subject_predictions(model, T, idx, Cz, mask_ch=None, use_ts=True):
    o = predict(model, idx, Cz, mask_ch=mask_ch)
    df = pd.DataFrame(dict(key=W.key.values[idx], prob_raw=1 / (1 + np.exp(-o['logit'])),
                           prob_ts=1 / (1 + np.exp(-o['logit'] / T)), sev=o['sev'], gate=o['gate']))
    df = df.groupby('key').mean().reset_index()
    df['prob'] = df.prob_ts if use_ts else df.prob_raw          # probability reported under the method's own protocol
    return df.merge(demo_s[SUBJ_COLS], on='key')

def subj_auc(model, idx, Cz, mask_ch=None):
    d = subject_predictions(model, 1.0, idx, Cz, mask_ch)
    return roc_auc_score(d.label, d.prob) if d.label.nunique() == 2 else np.nan

# ## 7. Subject-level splits (no leakage) and the federated engine
# Server rules implemented (traditional baselines exactly as in their papers):
# * **FedAvg**: θ_g = Σ (n_k/n) θ_k
# * **FedProx**: FedAvg plus a local proximal term (μ/2)‖θ − θ_g‖²
# * **FedAdam** (FedOpt, Reddi et al. 2021): Adam step on the averaged pseudo-gradient at the server
# * **q-FFL** (Li et al. 2020, loss-reweighted form): w_k ∝ n_k · L_k^q, which up-weights clients with higher loss (fairness across clients)
# * **FedX (ours)**: FedCAA consensus-attribution weights plus the consensus and fairness regularisers and federated temperature scaling

def fit_clin_scaler(idx):
    med = np.nanmedian(C[idx], 0); med = np.where(np.isfinite(med), med, 0)
    Ci = np.where(np.isnan(C), med, C); mu = Ci[idx].mean(0); sd = Ci[idx].std(0) + 1e-6
    return np.clip((Ci - mu) / sd, -5, 5).astype(np.float32)

def keys_to_idx(keys):
    return np.flatnonzero(W.key.isin(set(keys)).values)

def make_cv_split(fold, seed=CFG['seed']):
    sp = {'clients': {}}
    for st in CLIENTS:
        s = demo_s[demo_s.Study == st].reset_index(drop=True)
        skf = StratifiedKFold(CFG['n_folds'], shuffle=True, random_state=seed)
        tr, te = list(skf.split(s.key, s.label))[fold]
        trk, vak = train_test_split(s.key.values[tr], test_size=CFG['val_frac'], stratify=s.label.values[tr], random_state=seed + fold)
        sp['clients'][st] = dict(train=keys_to_idx(trk), val=keys_to_idx(vak), test=keys_to_idx(s.key.values[te]))
    sp['Cz'] = fit_clin_scaler(np.concatenate([v['train'] for v in sp['clients'].values()]))
    return sp

def make_loso_split(holdout, seed=CFG['seed']):
    sp = {'clients': {}}
    for st in [c for c in CLIENTS if c != holdout]:
        s = demo_s[demo_s.Study == st]
        trk, vak = train_test_split(s.key.values, test_size=CFG['val_frac'], stratify=s.label.values, random_state=seed)
        sp['clients'][st] = dict(train=keys_to_idx(trk), val=keys_to_idx(vak), test=np.array([], int))
    sp['holdout'] = keys_to_idx(demo_s[demo_s.Study == holdout].key)
    sp['Cz'] = fit_clin_scaler(np.concatenate([v['train'] for v in sp['clients'].values()]))
    return sp

def run_federated(sp, mc, seed=CFG['seed']):
    global RNG
    set_seed(seed); RNG = np.random.default_rng(seed)
    cl = list(sp['clients']); Cz = sp['Cz']; algo = mc['algo']; mk = mc.get('model', {})
    gm = BioGaitNet(**mk).to(DEVICE)
    gstate = {k: v.detach().clone() for k, v in gm.state_dict().items()}
    if algo == 'fedadam':
        m_s = {k: torch.zeros_like(v) for k, v in gstate.items() if v.is_floating_point()}
        v_s = {k: torch.zeros_like(v) for k, v in gstate.items() if v.is_floating_point()}
    A_glob, P_glob, best, hist = None, None, (np.inf, None, None), []
    for r in range(CFG['rounds']):
        states, ns, profs, losses = [], [], [], []
        for k in cl:
            m = BioGaitNet(**mk).to(DEVICE); m.load_state_dict(gstate)
            tr = sp['clients'][k]['train']
            if algo == 'qffl': losses.append(val_loss(m, sub_idx(tr, CFG['profile_n']), Cz))
            prox = [p.detach().clone() for p in m.parameters()] if algo == 'fedprox' else None
            train_local(m, tr, Cz, CFG['local_epochs'], A_glob=A_glob,
                        lam_cons=mc.get('lam_cons', 0), lam_fair=mc.get('lam_fair', 0),
                        prox_ref=prox, mu=mc.get('mu', CFG['prox_mu']))
            states.append({kk: v.detach().clone() for kk, v in m.state_dict().items()}); ns.append(len(tr))
            if algo == 'fedx': profs.append(full_profile(m, tr, Cz))
        ns = np.array(ns, float); s = np.full(len(cl), np.nan)
        if algo == 'fedx':
            P = np.stack(profs)
            s = np.array([pearson(P[i], np.delete(P, i, 0).mean(0)) for i in range(len(P))])
            w = ns * np.exp(mc.get('tau', CFG['tau']) * s) if mc.get('use_caa', True) else ns.copy()
        elif algo == 'qffl':
            w = ns * (np.array(losses) + 1e-8) ** mc.get('q', CFG['qffl_q'])
        else:
            w = ns.copy()
        w = w / w.sum()
        avg = fed_aggregate(states, w)
        if algo == 'fedadam':                                   # server-side Adam on the pseudo-gradient
            eta, b1, b2, eps = mc.get('server_lr', CFG['fedadam_lr']), 0.9, 0.99, 1e-3
            new = {}
            for kk in gstate:
                if not gstate[kk].is_floating_point(): new[kk] = avg[kk]; continue
                d = avg[kk] - gstate[kk]
                m_s[kk] = b1 * m_s[kk] + (1 - b1) * d; v_s[kk] = b2 * v_s[kk] + (1 - b2) * d * d
                new[kk] = gstate[kk] + eta * m_s[kk] / (v_s[kk].sqrt() + eps)
            gstate = new
        else:
            gstate = avg
        if algo == 'fedx':
            P_glob = (w[:, None] * P).sum(0); A_glob = P_glob[:N_CH] / P_glob[:N_CH].sum()
        gm.load_state_dict(gstate)
        vls = {k: val_loss(gm, sp['clients'][k]['val'], Cz) for k in cl}
        va = np.concatenate([sp['clients'][k]['val'] for k in cl])
        vl = float(np.mean(list(vls.values())))
        hist.append(dict(round=r + 1, val_loss=vl, val_auc=subj_auc(gm, va, Cz), **{f'w_{k}': wk for k, wk in zip(cl, w)},
                         **{f's_{k}': sk for k, sk in zip(cl, s)}, **{f'vl_{k}': v for k, v in vls.items()}))
        if vl < best[0]: best = (vl, copy.deepcopy(gstate), None if P_glob is None else P_glob.copy())
    gm.load_state_dict(best[1])
    # federated temperature is always computed (for the plug-in calibration analysis) but only USED if mc['use_ts']
    T = fed_temperature([nll_curve(predict(gm, sp['clients'][k]['val'], Cz)['logit'], Y[sp['clients'][k]['val']]) for k in cl])
    return dict(model=gm, T=T, use_ts=mc.get('use_ts', False), hist=pd.DataFrame(hist), P_glob=best[2], best_val=best[0])

# ## 8. Methods: federated baselines, proposed method, and cumulative ablation

FULL = dict(use_gate=True, use_clin=True, bilateral=True)
CNN  = dict(use_gate=False, use_clin=False, bilateral=False)          # standard 1D-CNN backbone
BASE_BB = CNN if CFG['baseline_backbone'] == 'cnn' else FULL
BTS = CFG['baseline_ts']
OURS = 'FedX-Biomarker (ours)'

FL_BASELINES = {
    'FedAvg':  dict(algo='fedavg',  model=BASE_BB, use_ts=BTS),
    'FedProx': dict(algo='fedprox', model=BASE_BB, use_ts=BTS, mu=CFG['prox_mu']),
    'FedAdam': dict(algo='fedadam', model=BASE_BB, use_ts=BTS, server_lr=CFG['fedadam_lr']),
    'q-FFL':   dict(algo='qffl',    model=BASE_BB, use_ts=BTS, q=CFG['qffl_q']),
}
OURS_CFG = dict(algo='fedx', model=FULL, use_caa=True, tau=CFG['tau'], lam_cons=CFG['lam_cons'], lam_fair=CFG['lam_fair'], use_ts=True)
MAIN = list(FL_BASELINES) + [OURS]

# ---- cumulative ablation: start from traditional FedAvg and add one contribution at a time
A0N, A1N = 'A0: FedAvg (standard 1D-CNN)', 'A1: + BioGait-Net (C1)'
A2N, A3N, A4N = 'A2: + FedCAA weighting (C2a)', 'A3: + consensus regulariser (C2b)', 'A4: + fairness reg. & fed-TS (C3) = FedX'
ABL_EXTRA = {}
if CFG['baseline_backbone'] == 'cnn':
    ABL_EXTRA[A1N] = dict(algo='fedavg', model=FULL, use_ts=False)
    ABL_MAP = {A0N: 'FedAvg', A1N: A1N}
else:
    ABL_EXTRA[A0N] = dict(algo='fedavg', model=CNN, use_ts=False)
    ABL_MAP = {A0N: A0N, A1N: 'FedAvg'}
ABL_EXTRA[A2N] = dict(algo='fedx', model=FULL, use_caa=True, tau=CFG['tau'], lam_cons=0.0, lam_fair=0.0, use_ts=False)
ABL_EXTRA[A3N] = dict(algo='fedx', model=FULL, use_caa=True, tau=CFG['tau'], lam_cons=CFG['lam_cons'], lam_fair=0.0, use_ts=False)
ABL_MAP.update({A2N: A2N, A3N: A3N, A4N: OURS})
IREF = ABL_MAP[A1N]                  # FedAvg + BioGait-Net: reference for the interpretability comparison

# ---- optional supplementary: leave-one-out of C1 sub-modules inside FedX
COMP = {}
if CFG['run_component_ablation']:
    COMP = {'FedX w/o stride branch': dict(OURS_CFG, model=dict(FULL, use_clin=False)),
            'FedX w/o sensor gate':   dict(OURS_CFG, model=dict(FULL, use_gate=False)),
            'FedX w/o bilateral enc': dict(OURS_CFG, model=dict(FULL, bilateral=False))}

def build_runs():
    return dict(FL_BASELINES, **{OURS: OURS_CFG}, **ABL_EXTRA, **COMP)
RUNS = build_runs()
print('Runs per fold:', list(RUNS))
print('Ablation rows:', ABL_MAP)

# ### 8.1 Validation-only hyperparameter selection (same budget for every FL method)
# Each method gets a small grid over **its own** key hyperparameter. The configuration with the lowest federated validation loss is kept. Test subjects are never touched.

TUNE_GRID = {
    'FedProx': [dict(mu=v) for v in (0.001, 0.01, 0.1)],
    'FedAdam': [dict(server_lr=v) for v in (3e-3, 1e-2, 3e-2)],
    'q-FFL':   [dict(q=v) for v in (0.5, 1.0, 2.0)],
    OURS:      [dict(lam_cons=a, lam_fair=b) for a in (0.2, 0.5) for b in (0.2, 0.5)],
}
T_tune = pd.DataFrame()
if CFG['tune']:
    sp0 = make_cv_split(0, CFG['seeds'][0]); rows = []
    for name, grid in TUNE_GRID.items():
        for g in grid:
            mc = dict(RUNS[name], **g); res = run_federated(sp0, mc, CFG['seeds'][0])
            rows.append(dict(method=name, params=json.dumps(g), val_loss=res['best_val']))
            print(f'  tune {name:22s} {g} -> val loss {res["best_val"]:.4f}')
    T_tune = pd.DataFrame(rows)
    for name in TUNE_GRID:
        bestp = json.loads(T_tune[T_tune.method == name].sort_values('val_loss').params.iloc[0])
        (FL_BASELINES[name] if name in FL_BASELINES else OURS_CFG).update(bestp)
    # the cumulative ablation inherits FedX's tuned consensus strength so that A3 -> A4 adds only C3
    ABL_EXTRA[A3N]['lam_cons'] = OURS_CFG['lam_cons']
    for k in COMP: COMP[k].update(lam_cons=OURS_CFG['lam_cons'], lam_fair=OURS_CFG['lam_fair'])
    RUNS = build_runs()
    T_tune.to_csv(f'{TAB_DIR}/T0_hyperparameter_selection.csv', index=False); display(T_tune)
print({k: {kk: v for kk, v in RUNS[k].items() if kk not in ('model',)} for k in MAIN})

# ### 8.2 Interpretability metrics
# * **Cross-client consistency (CCC):** mean pairwise Pearson / Kendall-τ between the sensor-importance profiles each client derives from the *same* global model.
# * **Faithfulness (deletion test):** mean-ablate the top-k channels of the global profile vs random k channels (Deletion gap > 0 means the profile is faithful).
# * **Clinical concordance:** Spearman ρ between stride-feature attribution and the reference |Cliff's δ| (Table 2).
# 
# These metrics need the sensor gate, so they are computed for the gated models: A1 (FedAvg + BioGait-Net), A2, A3 and FedX.

DEL_K = [1, 2, 4, 6, 8]
def interpretability(model, sp, name):
    Cz = sp['Cz']; cl = list(sp['clients'])
    sp_prof = {k: sensor_profile(model, sp['clients'][k]['train'], Cz, n=2048) for k in cl}
    cl_prof = {k: clin_attribution(model, sp['clients'][k]['train'], Cz, n=2048) for k in cl}
    ns = np.array([len(sp['clients'][k]['train']) for k in cl], float); ns /= ns.sum()
    g_s = sum(n * sp_prof[k] for n, k in zip(ns, cl)); g_c = sum(n * cl_prof[k] for n, k in zip(ns, cl))
    pairs = list(itertools.combinations(cl, 2))
    r = dict(method=name,
             CCC_sensor_r=np.mean([pearson(sp_prof[a], sp_prof[b]) for a, b in pairs]),
             CCC_sensor_tau=np.mean([stats.kendalltau(sp_prof[a], sp_prof[b]).correlation for a, b in pairs]),
             CCC_clin_r=np.mean([pearson(cl_prof[a], cl_prof[b]) for a, b in pairs]))
    ref = T_bm.set_index('feature').reindex(CLIN_NAMES).abs_delta.values; ok = np.isfinite(ref)
    r['Concordance_rho'] = stats.spearmanr(g_c[ok], ref[ok]).correlation if model.use_clin else np.nan
    te = np.concatenate([sp['clients'][k]['test'] for k in cl]); base = subj_auc(model, te, Cz)
    order = np.argsort(-g_s); drops_top, drops_rnd = [], []; rng = np.random.default_rng(0)
    for k in DEL_K:
        drops_top.append(base - subj_auc(model, te, Cz, mask_ch=order[:k]))
        drops_rnd.append(np.mean([base - subj_auc(model, te, Cz, mask_ch=rng.choice(N_CH, k, replace=False)) for _ in range(5)]))
    r.update(base_AUROC=base, DelDrop_top=np.mean(drops_top), DelDrop_rand=np.mean(drops_rnd),
             Deletion_gap=np.mean(drops_top) - np.mean(drops_rnd))
    mat = np.array([[pearson(sp_prof[a], sp_prof[b]) for b in cl] for a in cl])
    return r, dict(global_sensor=g_s, global_clin=g_c, client_sensor=sp_prof, client_clin=cl_prof, ccc_matrix=mat,
                   del_top=drops_top, del_rand=drops_rnd)

def plist(name, key=None):
    v = [p for (n, s, f), p in PROFILES.items() if n == name]
    return v if key is None else [p[key] for p in v]

# ## 9. Main experiment: seeds × subject-level stratified K-fold CV inside each client (federated methods only)

OOF, FOLD_MET, HIST, INTERP, PROFILES, TIMES, TEMPS, MODELS_LAST = [], [], {}, [], {}, {}, {}, {}
t_all = time.time()
for seed in CFG['seeds']:
    for fold in range(CFG['n_folds']):
        sp = make_cv_split(fold, seed); Cz = sp['Cz']
        te_all = np.concatenate([v['test'] for v in sp['clients'].values()])
        print(f'\n===== seed {seed} | fold {fold+1}/{CFG["n_folds"]} | train {sum(len(v["train"]) for v in sp["clients"].values())} '
              f'val {sum(len(v["val"]) for v in sp["clients"].values())} test {len(te_all)} windows =====')
        for name, mc in RUNS.items():
            t0 = time.time()
            res = run_federated(sp, mc, seed + fold)
            subj = subject_predictions(res['model'], res['T'], te_all, Cz, use_ts=res['use_ts'])
            HIST[(name, seed, fold)] = res['hist']
            if res['model'].use_gate:
                r, prof = interpretability(res['model'], sp, name)
                INTERP.append(dict(seed=seed, fold=fold, **r)); PROFILES[(name, seed, fold)] = prof
            if seed == CFG['seeds'][0] and fold == CFG['n_folds'] - 1 and name == OURS:
                MODELS_LAST[name] = (res['model'], res['T'], sp)
            dt = time.time() - t0; TIMES.setdefault(name, []).append(dt); TEMPS.setdefault(name, []).append(res['T'])
            OOF.append(subj.assign(method=name, seed=seed, fold=fold))
            m = clf_metrics(subj.label, subj.prob); FOLD_MET.append(dict(method=name, seed=seed, fold=fold, **m))
            print(f'  {name:42s} AUROC={m["AUROC"]:.3f} BACC={m["BACC"]:.3f} F1={m["F1"]:.3f} MCC={m["MCC"]:.3f} ECE={m["ECE"]:.3f} ({dt:.0f}s)')
OOF = pd.concat(OOF, ignore_index=True); FOLD_MET = pd.DataFrame(FOLD_MET); INTERP = pd.DataFrame(INTERP)
OOF.to_csv(f'{TAB_DIR}/oof_subject_predictions.csv', index=False); FOLD_MET.to_csv(f'{TAB_DIR}/fold_metrics_all_runs.csv', index=False)
print(f'\nTotal time: {(time.time()-t_all)/60:.1f} min')

# ## 10. Results
# ### 10.1 Main comparison with federated baselines (Table 3): mean ± SD over seeds × folds, worst fold, and pooled AUROC with 95 % CI

MET_COLS = ['AUROC', 'AUPRC', 'ACC', 'BACC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'MCC', 'Kappa', 'Brier', 'ECE']
def mean_sd_table(fm, rows_map):
    rows = []
    for label, mth in rows_map.items():
        d = fm[fm.method == mth]
        if d.empty: continue
        row = {'Method': label}
        for c in MET_COLS: row[c] = f'{d[c].mean():.3f}±{d[c].std():.3f}'
        row['Worst-fold AUROC'] = f'{d.AUROC.min():.3f}'; row['Worst-fold MCC'] = f'{d.MCC.min():.3f}'
        o = OOF[OOF.method == mth]; lo, hi = boot_ci(o.label.values, o.prob.values)
        row['Pooled AUROC [95% CI]'] = f'{roc_auc_score(o.label, o.prob):.3f} [{lo:.3f},{hi:.3f}]'
        rows.append(row)
    return pd.DataFrame(rows)

def save_table(df, name):
    df.to_csv(f'{TAB_DIR}/{name}.csv', index=False)
    try:
        with open(f'{TAB_DIR}/{name}.tex', 'w') as f: f.write(df.to_latex(index=False, escape=True))
    except Exception:
        pass

T_main = mean_sd_table(FOLD_MET, {m: m for m in MAIN}); save_table(T_main, 'T3_main_results'); display(T_main)

# ### 10.2 Robustness: average vs worst case, mean rank, and win rate of FedX (Table 3b)

RANK_UP, RANK_DOWN = ['AUROC', 'BACC', 'F1', 'MCC'], ['ECE', 'Brier']
fm = FOLD_MET[FOLD_MET.method.isin(MAIN)].copy()
rk = []
for (s, f), d in fm.groupby(['seed', 'fold']):
    r = pd.concat([d.set_index('method')[RANK_UP].rank(ascending=False), d.set_index('method')[RANK_DOWN].rank(ascending=True)], axis=1)
    rk.append(r.mean(1).rename('rank'))
mean_rank = pd.concat(rk, axis=1).mean(1)
ours_f = fm[fm.method == OURS].set_index(['seed', 'fold'])
rows = []
for mth in MAIN:
    d = fm[fm.method == mth]; o = OOF[OOF.method == mth]
    per_client = [roc_auc_score(o[o.Study == c].label, o[o.Study == c].prob) for c in CLIENTS if o[o.Study == c].label.nunique() == 2]
    other = d.set_index(['seed', 'fold'])
    row = dict(Method=mth, AUROC_mean=d.AUROC.mean(), AUROC_SD=d.AUROC.std(), AUROC_median=d.AUROC.median(),
               AUROC_worst_fold=d.AUROC.min(), MCC_worst_fold=d.MCC.min(), BACC_worst_fold=d.BACC.min(),
               Worst_client_AUROC=min(per_client) if per_client else np.nan, Mean_rank=mean_rank.get(mth, np.nan))
    if mth != OURS:
        row['FedX_winrate_AUROC'] = float((ours_f.AUROC >= other.AUROC.reindex(ours_f.index)).mean())
        row['FedX_winrate_MCC'] = float((ours_f.MCC >= other.MCC.reindex(ours_f.index)).mean())
    rows.append(row)
T_rob = pd.DataFrame(rows).round(3); save_table(T_rob, 'T3b_robustness_worst_case'); display(T_rob)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.boxplot([fm[fm.method == m].AUROC.values for m in MAIN], vert=True)
ax.set_xticks(range(1, len(MAIN) + 1)); ax.set_xticklabels(MAIN, rotation=15); ax.set_ylabel('subject AUROC per run')
ax.set_title('Distribution over seeds × folds (average and worst case)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F3b_auroc_distribution.png', bbox_inches='tight'); plt.show()

# ### 10.3 Per-client (per-site) performance (Table 4)

rows = []
for mth in MAIN:
    for st in CLIENTS:
        o = OOF[(OOF.method == mth) & (OOF.Study == st)]
        if len(o) and o.label.nunique() == 2:
            m = clf_metrics(o.label, o.prob); rows.append(dict(Method=mth, Client=st, **{k: round(m[k], 3) for k in ['AUROC', 'BACC', 'F1', 'MCC', 'ECE']}))
T_client = pd.DataFrame(rows)
T_client_w = T_client.pivot(index='Method', columns='Client', values=['AUROC', 'BACC', 'F1']).round(3)
T_client_w['Worst-client AUROC'] = T_client_w['AUROC'].min(1); T_client_w['Client AUROC SD'] = T_client_w['AUROC'].std(1).round(3)
T_client_w = T_client_w.reindex(MAIN)
T_client_w.to_csv(f'{TAB_DIR}/T4_per_client.csv'); display(T_client_w)

# ### 10.4 Fairness (Table 5), sex and age groups
# *Note:* DPD depends on PD prevalence per group, so **EOD / Equalised-Odds and AUROC gap** are the main clinical fairness criteria. q-FFL is the fairness-oriented FL baseline (fairness across clients). FedX targets demographic groups inside clients.

rows = []
for mth in MAIN:
    o = OOF[OOF.method == mth]
    for attr, nm in [('sex', 'Sex'), ('age_grp', f'Age(<{CFG["age_cut"]}/≥{CFG["age_cut"]})')]:
        rows.append(dict(Method=mth, Attribute=nm, **{k: round(v, 3) for k, v in fairness_metrics(o, attr).items()}))
T_fair = pd.DataFrame(rows); save_table(T_fair, 'T5_fairness'); display(T_fair)

fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, att in zip(axs, T_fair.Attribute.unique()):
    d = T_fair[T_fair.Attribute == att].set_index('Method')[['EOD', 'EqOdds', 'AUROC_gap']]
    d.plot.barh(ax=ax); ax.set_title(f'Fairness gaps ({att}); lower is better'); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F3_fairness.png', bbox_inches='tight'); plt.show()

# ### 10.5 Calibration (Table 6)
# `reported` = the method's own protocol (baselines uncalibrated, as published; FedX with federated temperature scaling). The `fed-TS plug-in` column shows what C3 would also give each baseline, which demonstrates that C3 is a generic, privacy-preserving add-on.

def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return ece_score(y, p), brier_score_loss(y, p), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
rows = []
for mth in MAIN:
    o = OOF[OOF.method == mth]; y = o.label.values
    for col, tag in [('prob', 'reported'), ('prob_raw', 'uncalibrated'), ('prob_ts', 'fed-TS plug-in')]:
        e, b, n = calib(y, o[col].values)
        rows.append(dict(Method=mth, Probabilities=tag, ECE=round(e, 4), Brier=round(b, 4), NLL=round(n, 4), mean_T=round(np.mean(TEMPS[mth]), 2)))
T_cal = pd.DataFrame(rows); save_table(T_cal, 'T6_calibration'); display(T_cal)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, mth in zip(axs, ['FedAvg', 'FedProx', OURS]):
    o = OOF[OOF.method == mth]
    for col, lab in [('prob_raw', 'uncalibrated'), ('prob_ts', 'fed-TS')]:
        bins = np.linspace(0, 1, 6); b = np.clip(np.digitize(o[col], bins) - 1, 0, 4)
        xs = [o[col][b == k].mean() for k in range(5) if (b == k).any()]; ys = [o.label[b == k].mean() for k in range(5) if (b == k).any()]
        ax.plot(xs, ys, 'o-', label=f'{lab} (ECE={ece_score(o.label.values, o[col].values):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=.8); ax.set_title(mth); ax.set_xlabel('predicted P(PD)'); ax.set_ylabel('observed PD rate'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F4_reliability.png', bbox_inches='tight'); plt.show()

# ### 10.6 Disease severity (Table 7). The auxiliary head is trained on H&Y only; UPDRS / UPDRS-III serve as **external** validation

rows = []
for mth in MAIN:
    rows.append(dict(Method=mth, **{k: round(v, 3) for k, v in severity_metrics(OOF[OOF.method == mth]).items()}))
T_sev = pd.DataFrame(rows); save_table(T_sev, 'T7_severity'); display(T_sev)

o = OOF[(OOF.method == OURS) & (OOF.label == 1)]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].scatter(o.HoehnYahr + np.random.uniform(-.05, .05, len(o)), o.sev, alpha=.5); axs[0].set_xlabel('Hoehn & Yahr'); axs[0].set_ylabel('predicted severity')
axs[1].scatter(o.UPDRS, o.sev, alpha=.5, color='#DD8452'); axs[1].set_xlabel('UPDRS total (never seen in training)'); axs[1].set_ylabel('predicted severity')
d = o[['sev', 'UPDRS']].dropna()
if len(d) > 5: axs[1].set_title(f"Spearman ρ = {stats.spearmanr(d.sev, d.UPDRS).correlation:.2f}")
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F5_severity.png', bbox_inches='tight'); plt.show()

# ### 10.7 Digital-biomarker discovery and interpretability (Table 8, Figures 6–9)

IC = ['CCC_sensor_r', 'CCC_sensor_tau', 'CCC_clin_r', 'Concordance_rho', 'base_AUROC', 'DelDrop_top', 'DelDrop_rand', 'Deletion_gap']
def run_stability(name):
    g = plist(name, 'global_sensor')
    return np.mean([stats.kendalltau(a, b).correlation for a, b in itertools.combinations(g, 2)]) if len(g) > 1 else np.nan
INV = {v: k for k, v in ABL_MAP.items()}
T_int = INTERP.groupby('method')[IC].agg(['mean', 'std'])
T_int.columns = [f'{a}_{b}' for a, b in T_int.columns]
T_int['RunStability_tau'] = [run_stability(m) for m in T_int.index]
T_int.index = [INV.get(m, m) for m in T_int.index]
T_int = T_int.round(3).sort_index(); T_int.to_csv(f'{TAB_DIR}/T8_interpretability.csv'); display(T_int)

def plot_insole(ax, prof, title, vmax=None):
    imp = np.r_[prof[0:8], prof[11:19]]; vmax = vmax or imp.max()
    xs = np.r_[SENSOR_XY_L[:, 0], -SENSOR_XY_L[:, 0]]; ys = np.r_[SENSOR_XY_L[:, 1], SENSOR_XY_L[:, 1]]
    for cx in (-500, 500): ax.add_patch(Ellipse((cx, 0), 650, 2000, fill=False, lw=1.2, color='grey'))
    sc = ax.scatter(xs, ys, c=imp, s=40 + 700 * imp / vmax, cmap='viridis', vmin=0, vmax=vmax, edgecolors='k')
    for i, (x, y) in enumerate(zip(xs, ys)): ax.text(x, y - 150, ('L' if i < 8 else 'R') + str(i % 8 + 1), ha='center', fontsize=7)
    ax.set_title(title + f"\nTotal L/R = {prof[8]:.3f}/{prof[19]:.3f}\nCOP L/R = {prof[9]+prof[10]:.3f}/{prof[20]+prof[21]:.3f}", fontsize=8)
    ax.set_xlim(-1000, 1000); ax.set_ylim(-1150, 1150); ax.set_aspect('equal'); ax.axis('off')
    return sc

rank, T_conc = None, None
if plist(OURS):
    g = np.mean(plist(OURS, 'global_sensor'), 0)
    cmaps = {k: np.mean([p['client_sensor'][k] for p in plist(OURS)], 0) for k in CLIENTS}
    vmax = max([np.r_[g[0:8], g[11:19]].max()] + [np.r_[v[0:8], v[11:19]].max() for v in cmaps.values()])
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    sc = plot_insole(axs[0], g, 'FedX global biomarker map', vmax)
    for ax, k in zip(axs[1:], CLIENTS): plot_insole(ax, cmaps[k], f'client {k}', vmax)
    fig.colorbar(sc, ax=axs, shrink=.6, label='sensor importance α'); plt.savefig(f'{FIG_DIR}/F6_insole_biomarker_maps.png', bbox_inches='tight'); plt.show()
    rank = pd.DataFrame(dict(channel=CH_NAMES, importance=g, region=[SENSOR_REGION.get(c.split('-')[1], c.split('-')[1]) for c in CH_NAMES]))
    rank = rank.sort_values('importance', ascending=False); rank.to_csv(f'{TAB_DIR}/T9_global_sensor_biomarkers.csv', index=False); display(rank.head(10))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, mth, lab in zip(axs, [IREF, OURS], ['FedAvg + BioGait-Net (A1)', 'FedX (ours)']):
    if not plist(mth): continue
    M = np.mean(plist(mth, 'ccc_matrix'), 0)
    im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r'); ax.set_xticks(range(3)); ax.set_xticklabels(CLIENTS); ax.set_yticks(range(3)); ax.set_yticklabels(CLIENTS)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center')
    ax.set_title(f'{lab}: cross-client attribution r')
fig.colorbar(im, ax=axs, shrink=.8); plt.savefig(f'{FIG_DIR}/F7_cross_client_consistency.png', bbox_inches='tight'); plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4))
for mth, lab, c in [(IREF, 'A1 FedAvg+BioGait', '#4C72B0'), (A2N, 'A2 +FedCAA', '#55A868'), (A3N, 'A3 +consensus', '#8172B2'), (OURS, 'FedX', '#C44E52')]:
    if not plist(mth): continue
    ax.plot(DEL_K, np.mean(plist(mth, 'del_top'), 0), 'o-', color=c, label=f'{lab}: top-k')
    ax.plot(DEL_K, np.mean(plist(mth, 'del_rand'), 0), 'x--', color=c, alpha=.5, label=f'{lab}: random-k')
ax.set_xlabel('# channels ablated'); ax.set_ylabel('subject AUROC drop'); ax.set_title('Deletion faithfulness'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F8_deletion_faithfulness.png', bbox_inches='tight'); plt.show()

if plist(OURS):
    gc = np.mean(plist(OURS, 'global_clin'), 0); ref = T_bm.set_index('feature').reindex(CLIN_NAMES)
    T_conc = pd.DataFrame(dict(feature=CLIN_NAMES, FedX_attr=gc, abs_cliffs_delta=ref.abs_delta.values, q_FDR=ref.q_FDR.values)).sort_values('FedX_attr', ascending=False)
    T_conc.to_csv(f'{TAB_DIR}/T10_stride_biomarker_concordance.csv', index=False); display(T_conc)
    fig, ax = plt.subplots(figsize=(11, 3.8)); xx = np.arange(len(T_conc))
    ax.bar(xx - .2, T_conc.FedX_attr / T_conc.FedX_attr.max(), .4, label='FedX attribution (norm.)')
    ax.bar(xx + .2, T_conc.abs_cliffs_delta / np.nanmax(T_conc.abs_cliffs_delta), .4, label="|Cliff's δ| PD vs Co (norm.)")
    ax.set_xticks(xx); ax.set_xticklabels(T_conc.feature, rotation=45, ha='right'); ax.legend()
    ax.set_title(f"Clinical concordance ρ = {stats.spearmanr(T_conc.FedX_attr, T_conc.abs_cliffs_delta, nan_policy='omit').correlation:.2f}")
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F9_clinical_concordance.png', bbox_inches='tight'); plt.show()

# ### 10.8 Where in the gait cycle does the model look? (temporal attention β, fusion gate g)

if OURS in MODELS_LAST:
    m, T, sp = MODELS_LAST[OURS]
    te = np.concatenate([v['test'] for v in sp['clients'].values()])
    o = predict(m, te, sp['Cz'], keep_beta=True)
    fig, axs = plt.subplots(3, 1, figsize=(13, 7.5))
    for ax, lab in zip(axs[:2], [0, 1]):
        cand = np.flatnonzero(Y[te] == lab)
        if not len(cand): continue
        j = cand[np.argmax(o['logit'][cand]) if lab == 1 else np.argmin(o['logit'][cand])]
        t = np.arange(X.shape[-1]) / (CFG['fs'] / CFG['ds'])
        ax.plot(t, X[te[j], 8], label='L total/BW'); ax.plot(t, X[te[j], 19], label='R total/BW', alpha=.8)
        b = np.repeat(o['beta'][j], X.shape[-1] // o['beta'].shape[1] + 1)[:X.shape[-1]]
        ax2 = ax.twinx(); ax2.fill_between(t, 0, b, color='red', alpha=.25); ax2.set_yticks([])
        ax.set_title(f"{'PD' if lab else 'Control'} window {W.key.values[te[j]]}: P(PD)={1/(1+np.exp(-o['logit'][j]/T)):.2f}, gate g={o['gate'][j]:.2f}  (red = attention β)")
        ax.legend(loc='upper left', fontsize=8)
    axs[2].hist(o['gate'][Y[te] == 0], bins=30, alpha=.6, label='Control'); axs[2].hist(o['gate'][Y[te] == 1], bins=30, alpha=.6, label='PD')
    axs[2].set_xlabel('fusion gate g (1 = deep VGRF branch, 0 = stride-dynamics branch)'); axs[2].legend()
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F10_temporal_attention_gate.png', bbox_inches='tight'); plt.show()

# ### 10.9 Cumulative ablation from traditional FedAvg (Table 11)
# Each row adds **one** contribution to the previous row. Anything not yet added uses the traditional FedAvg behaviour (sample-size averaging, no regulariser, no calibration). Δ columns are relative to A0 (plain FedAvg).

rows = []
base = FOLD_MET[FOLD_MET.method == ABL_MAP[A0N]]
for lab, mth in ABL_MAP.items():
    d = FOLD_MET[FOLD_MET.method == mth]; o = OOF[OOF.method == mth]
    if d.empty: continue
    it = INTERP[INTERP.method == mth]; fs_ = fairness_metrics(o, 'sex'); fa = fairness_metrics(o, 'age_grp')
    rows.append(dict(Configuration=lab,
                     AUROC=f'{d.AUROC.mean():.3f}±{d.AUROC.std():.3f}', BACC=f'{d.BACC.mean():.3f}±{d.BACC.std():.3f}',
                     F1=f'{d.F1.mean():.3f}±{d.F1.std():.3f}', MCC=f'{d.MCC.mean():.3f}±{d.MCC.std():.3f}',
                     ECE=f'{d.ECE.mean():.3f}±{d.ECE.std():.3f}', Worst_fold_AUROC=round(d.AUROC.min(), 3),
                     dAUROC_vs_A0=round(d.AUROC.mean() - base.AUROC.mean(), 3), dMCC_vs_A0=round(d.MCC.mean() - base.MCC.mean(), 3),
                     dECE_vs_A0=round(d.ECE.mean() - base.ECE.mean(), 3),
                     EOD_sex=round(fs_['EOD'], 3), EOD_age=round(fa['EOD'], 3),
                     CCC_sensor_r=round(it.CCC_sensor_r.mean(), 3) if len(it) else np.nan,
                     Deletion_gap=round(it.Deletion_gap.mean(), 3) if len(it) else np.nan))
T_abl = pd.DataFrame(rows); save_table(T_abl, 'T11_cumulative_ablation'); display(T_abl)

labs = [l for l in ABL_MAP if not FOLD_MET[FOLD_MET.method == ABL_MAP[l]].empty]
d = pd.DataFrame({l: FOLD_MET[FOLD_MET.method == ABL_MAP[l]][['AUROC', 'BACC', 'MCC']].mean() for l in labs}).T
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
d.plot(ax=axs[0], marker='o'); axs[0].set_xticks(range(len(labs))); axs[0].set_xticklabels([l.split(':')[0] for l in labs]); axs[0].set_title('Cumulative ablation (mean over runs)')
e = pd.DataFrame({l: [FOLD_MET[FOLD_MET.method == ABL_MAP[l]].ECE.mean(), fairness_metrics(OOF[OOF.method == ABL_MAP[l]], 'sex')['EOD']] for l in labs}, index=['ECE', 'EOD(sex)']).T
e.plot(ax=axs[1], marker='o'); axs[1].set_xticks(range(len(labs))); axs[1].set_xticklabels([l.split(':')[0] for l in labs]); axs[1].set_title('Trust metrics (lower is better)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F11_cumulative_ablation.png', bbox_inches='tight'); plt.show()

if COMP:
    T_comp = mean_sd_table(FOLD_MET, dict({OURS: OURS}, **{k: k for k in COMP}))[['Method', 'AUROC', 'BACC', 'F1', 'MCC', 'ECE', 'Worst-fold AUROC']]
    save_table(T_comp, 'T11b_component_ablation_supplementary'); display(T_comp)

# ### 10.10 ROC curves, convergence and FedCAA dynamics

fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
for mth in MAIN:
    o = OOF[OOF.method == mth]; fpr, tpr, _ = roc_curve(o.label, o.prob)
    axs[0].plot(fpr, tpr, lw=2 if mth == OURS else 1, label=f'{mth} ({roc_auc_score(o.label, o.prob):.3f})')
axs[0].plot([0, 1], [0, 1], 'k--', lw=.7); axs[0].set_xlabel('FPR'); axs[0].set_ylabel('TPR'); axs[0].set_title('Pooled ROC (subject level)'); axs[0].legend(fontsize=7)
for mth in MAIN:
    h = pd.concat([v for (n, s, f), v in HIST.items() if n == mth]).groupby('round').mean(numeric_only=True)
    axs[1].plot(h.index, h.val_loss, label=mth); axs[2].plot(h.index, h.val_auc, label=mth)
axs[1].set_title('Global val loss per round'); axs[2].set_title('Global val subject-AUROC per round')
for a in axs[1:]: a.set_xlabel('round'); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F12_roc_convergence.png', bbox_inches='tight'); plt.show()

h = pd.concat([v for (n, s, f), v in HIST.items() if n == OURS]).groupby('round').mean(numeric_only=True)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
for k in CLIENTS: axs[0].plot(h.index, h[f'w_{k}'], label=k); axs[1].plot(h.index, h[f's_{k}'], label=k)
axs[0].set_title('FedCAA aggregation weight w_k'); axs[1].set_title('Peer attribution agreement s_k')
for a in axs: a.set_xlabel('round'); a.legend()
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F13_fedcaa_dynamics.png', bbox_inches='tight'); plt.show()

cm = confusion_matrix(OOF[OOF.method == OURS].label, OOF[OOF.method == OURS].prob >= .5)
fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2): ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=13)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Co', 'PD']); ax.set_yticks([0, 1]); ax.set_yticklabels(['Co', 'PD'])
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('FedX (all runs)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/F14_confusion.png', bbox_inches='tight'); plt.show()

# ### 10.11 Statistical significance: paired subject-level bootstrap (FedX vs each FL baseline and each ablation step)

rows = []
ours = OOF[OOF.method == OURS][['key', 'seed', 'fold', 'label', 'prob']]
comps = [(m, m) for m in MAIN if m != OURS] + [(l, ABL_MAP[l]) for l in (A1N, A2N, A3N) if ABL_MAP[l] not in MAIN]
for lab, mth in comps:
    b = OOF[OOF.method == mth][['key', 'seed', 'fold', 'prob']]
    d = ours.merge(b, on=['key', 'seed', 'fold'], suffixes=('_o', '_b'))
    for fn, nm in [(roc_auc_score, 'ΔAUROC'), (lambda y, p: balanced_accuracy_score(y, p >= .5), 'ΔBACC'),
                   (lambda y, p: matthews_corrcoef(y, p >= .5), 'ΔMCC')]:
        mu, ci, p = paired_boot_diff(d.label.values, d.prob_o.values, d.prob_b.values, fn)
        rows.append(dict(Comparison=f'FedX vs {lab}', Metric=nm, Diff=round(mu, 4), CI95=f'[{ci[0]:.3f},{ci[1]:.3f}]', p_value=round(p, 4)))
T_sig = pd.DataFrame(rows); save_table(T_sig, 'T12_significance'); display(T_sig)

# ## 11. Generalisability: leave-one-study-out (Table 13), federated methods only
# Train on two studies and test on the unseen third site. Results are averaged over the seeds.

if CFG['run_loso']:
    rows = []
    for seed in CFG['seeds']:
        for ho in CLIENTS:
            sp = make_loso_split(ho, seed)
            for name in MAIN:
                res = run_federated(sp, RUNS[name], seed)
                subj = subject_predictions(res['model'], res['T'], sp['holdout'], sp['Cz'], use_ts=res['use_ts'])
                m = clf_metrics(subj.label, subj.prob); f = fairness_metrics(subj, 'sex')
                rows.append(dict(seed=seed, Held_out=ho, Method=name, **{k: m[k] for k in ['AUROC', 'BACC', 'F1', 'MCC', 'ECE']}, EOD_sex=f['EOD']))
                print(f'seed {seed} held-out {ho} | {name:24s} AUROC={m["AUROC"]:.3f} BACC={m["BACC"]:.3f} ECE={m["ECE"]:.3f}')
    T_loso = pd.DataFrame(rows)
    T_loso_site = T_loso.groupby(['Held_out', 'Method'])[['AUROC', 'BACC', 'F1', 'MCC', 'ECE', 'EOD_sex']].mean().round(3)
    g = T_loso.groupby('Method')
    T_loso_mean = pd.concat([g[['AUROC', 'BACC', 'F1', 'MCC', 'ECE', 'EOD_sex']].mean().round(3),
                             g.AUROC.min().rename('Worst_site_AUROC').round(3)], axis=1).reindex(MAIN)
    T_loso.to_csv(f'{TAB_DIR}/T13_loso_all.csv', index=False); T_loso_site.to_csv(f'{TAB_DIR}/T13_loso_per_site.csv')
    save_table(T_loso_mean.reset_index(), 'T13_loso_mean'); display(T_loso_site); display(T_loso_mean)

# ## 12. Complexity and communication cost (Table 14)

rows = []
for name, kw in [('BioGait-Net (FedX)', FULL), ('Standard 1D-CNN (baselines)', CNN)]:
    m = BioGaitNet(**kw).to(DEVICE).eval(); npar = sum(p.numel() for p in m.parameters())
    x = torch.randn(256, N_CH, X.shape[-1], device=DEVICE); c = torch.randn(256, N_CLIN, device=DEVICE)
    with torch.no_grad():
        for _ in range(3): m(x, c)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): m(x, c)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
    rows.append(dict(Model=name, Params=npar, Size_KB=round(npar * 4 / 1024, 1),
                     Latency_ms_per_window=round((time.time() - t0) / 10 / 256 * 1000, 4)))
T_cx = pd.DataFrame(rows)
p_b = sum(p.numel() for p in BioGaitNet(**BASE_BB).parameters()); p_o = N_PARAMS
comm = [('FedAvg / FedProx / FedAdam', p_b * 4), ('q-FFL', p_b * 4 + 4), (OURS, p_o * 4 + (N_CH + N_CLIN) * 4)]
T_comm = pd.DataFrame([dict(Method=n, Upload_per_client_round_KB=round(b / 1024, 2)) for n, b in comm])
T_comm['Total_upload_MB (all rounds, 3 clients)'] = (T_comm.Upload_per_client_round_KB * CFG['rounds'] * 3 / 1024).round(2)
T_comm['FedX_extra_vs_weights_pct'] = [np.nan, np.nan, round(100 * (N_CH + N_CLIN) / p_o, 3)]
T_time = pd.DataFrame([(k, np.mean(v)) for k, v in TIMES.items()], columns=['Method', 'Train_time_s_per_run']).round(1)
save_table(T_cx, 'T14a_complexity'); save_table(T_comm, 'T14b_communication'); save_table(T_time, 'T14c_time')
display(T_cx); display(T_comm); display(T_time)

# ## 13. Summary for the paper

def fmt(m, c): d = FOLD_MET[FOLD_MET.method == m][c]; return f'{d.mean():.3f}±{d.std():.3f} (worst {d.min() if c != "ECE" else d.max():.3f})'
print('=' * 110); print('FedX-Biomarker vs federated baselines (mean±SD over seeds×folds, worst run in brackets)'); print('=' * 110)
for m in MAIN:
    o = OOF[OOF.method == m]; fs_ = fairness_metrics(o, 'sex')
    print(f'{m:24s} AUROC {fmt(m,"AUROC")} | MCC {fmt(m,"MCC")} | ECE {fmt(m,"ECE")} | EOD(sex) {fs_["EOD"]:.3f} | mean rank {mean_rank.get(m, np.nan):.2f}')
print('\nCumulative ablation (AUROC mean):', ' -> '.join(f'{l.split(":")[0]} {FOLD_MET[FOLD_MET.method == ABL_MAP[l]].AUROC.mean():.3f}' for l in ABL_MAP if not FOLD_MET[FOLD_MET.method == ABL_MAP[l]].empty))
if rank is not None: print('Top-5 global digital biomarkers (sensor channels):', ', '.join(rank.channel.head(5)))
if T_conc is not None: print('Top-5 stride biomarkers (attribution):', ', '.join(T_conc.feature.head(5)))
json.dump({k: (v if not isinstance(v, dict) else str(v)) for k, v in CFG.items()}, open(f'{OUT_DIR}/config.json', 'w'), indent=2)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in RUNS.items()}, open(f'{OUT_DIR}/methods_final.json', 'w'), indent=2, default=str)
print(f'\nAll tables -> {TAB_DIR}\nAll figures -> {FIG_DIR}')

# ---
# ### Mapping results to the manuscript
# | Paper section | Notebook output |
# |---|---|
# | Cohort / dataset | T1, T2, F1, F2 |
# | Method (C1, C2, C3) | Sections 4, 5, 7 and the objective at the top |
# | Hyperparameter protocol | T0 (validation-only selection) |
# | Main results vs FL baselines | T3, T3b (worst case, rank, win rate), T4, F3b, F12, F14 |
# | Trustworthiness: fairness | T5, F3 |
# | Trustworthiness: calibration | T6, F4 |
# | Biomedical discovery | T9, T10, F6 (insole biomarker map), F9, F10 |
# | Reproducibility / consistency | T8, F7, F8, F13 |
# | Severity (disease modelling) | T7, F5 |
# | Cumulative ablation | T11, F11 (optional T11b) |
# | Generalisability | T13 (LOSO) |
# | Significance | T12 |
# | Efficiency | T14 |
# 
# **Citation (required by PhysioNet):** Hausdorff J. *Gait in Parkinson's Disease* (v1.0.0), PhysioNet, 2008, together with the PhysioNet platform citation (Pollard et al., 2026, *Nature Health*) and the three study references (Yogev 2005; Hausdorff 2007; Frenkel-Toledo 2005). FL baselines: McMahan 2017 (FedAvg), Li 2020 (FedProx), Reddi 2021 (FedAdam), Li 2020 ICLR (q-FFL).
