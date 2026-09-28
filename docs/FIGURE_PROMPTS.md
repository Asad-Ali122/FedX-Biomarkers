# Figure captions and AI generation prompts

This file gives the caption and a detailed generation prompt for each figure in the paper.

- **Figure 1** is a conceptual schematic, so it is drawn with an AI image or diagram tool.
- **Figures 2–4** are data figures. They are produced by the code (`scripts/make_paper_figures.py`). The prompts below are only for a professional redraw, and they quote the exact values that must appear.

Never let an AI tool change a number in a data figure.

General style for every figure:
- white background
- sans-serif font (Arial or Helvetica), at least 8 pt at final print size
- colour-blind-safe palette (blue #4C72B0, orange #DD8452, green #55A868, red #C44E52, grey #8C8C8C)
- vector PDF or 600 DPI PNG
- no 3-D effects, no drop shadows, no decorative gradients

---

## Figure 1: Framework overview (full width, `figures/fig1_framework.pdf`)

**Caption.** Overview of FedX-Biomarker. Each hospital trains BioGait-Net (C1) on its own insole recordings under one objective that joins the diagnostic loss, a consensus term and a fairness term. After every round a site uploads its weights θ_k, a 36-value attribution profile p_k and, at the end, a likelihood curve NLL_k(T). The server computes peer agreement s_k, aggregates with FedCAA weights w_k ∝ n_k·exp(τ·s_k) (C2), and returns the global model, the global biomarker profile A_g and a shared temperature T* (C3). Raw gait signals never leave a site.

**AI prompt.**
> Create a clean, publication-quality scientific schematic for an IEEE conference paper, landscape 2:1 aspect ratio, pure white background, flat vector style, thin dark-grey outlines (1 pt), sans-serif labels (Arial), colour-blind-safe palette: blue #4C72B0 for model components, orange #DD8452 for clinical/stride knowledge, green #55A868 for server/aggregation, red #C44E52 for trust/fairness/calibration, light grey #EDEDED for container boxes. No shadows, no 3-D, no gradients, no clip-art people.
>
> LEFT THIRD – three vertically stacked rounded rectangles labelled "Hospital Ga (dual-task study)", "Hospital Ju (auditory-cue study)", "Hospital Si (treadmill study)". Each contains: a small padlock icon with the text "raw data stay local"; a top-down pair of shoe insoles with 8 small circles per foot at positions heel, rear-midfoot (2), midfoot (2), forefoot (2), toe; a tiny waveform labelled "VGRF 100 Hz".
>
> CENTRE THIRD – one enlarged box titled "BioGait-Net (C1) – local model", drawn as a left-to-right pipeline:
> (1) two input streams "Left foot 11 ch" and "Right foot 11 ch";
> (2) a blue block "Sensor-Attention Gate α" on each stream, with a small bar chart over 11 channels;
> (3) both streams entering ONE shared blue block "Shared foot encoder (dilated DS-Conv + GroupNorm)", with a label "shared weights" and a double arrow;
> (4) a merge block "[h_L + h_R ; |h_L − h_R|] asymmetry fusion";
> (5) a block "Temporal attention β" with a small curve over a gait-cycle waveform;
> (6) an orange side branch from below "Stride-dynamics features (14): stride/swing CV, stance %, swing asymmetry, cadence, peak force" feeding a gate symbol "g" (a small valve icon) that fuses deep and stride representations;
> (7) two output heads "P(PD)" and "H&Y stage".
> Under the box a red banner shows the unified loss: "L_k = L_BCE + λ_s·L_sev + λ_c·(1 − ρ(ᾱ, A_g)) + λ_f·SD_g(L_g)", with the three extra terms coloured orange, green and red.
>
> RIGHT THIRD – a green rounded box "Server: FedCAA (C2) + Trust layer (C3)" containing three stacked mini-panels:
> (a) "Peer agreement s_k = ρ(p_k, mean_{j≠k} p_j)" with three small profile bar charts and correlation lines between them;
> (b) "Weights w_k ∝ n_k·e^{τ s_k}" with a small pie chart of three slices;
> (c) "Federated temperature T* = argmin Σ_k NLL_k(T)" with three thin U-shaped curves summed into one bold curve and its minimum marked.
>
> ARROWS: upward arrows from each hospital to the server labelled "θ_k + p_k (36 floats, <0.1 % overhead) + NLL_k(T)"; return arrows labelled "θ_g, A_g (global biomarker profile), T*".
>
> BOTTOM STRIP across the full width: a thin grey band with three icons and short labels: "Interpretable (C1)", "Consistent across sites (C2)", "Fair & calibrated (C3)". Keep all text legible at 17 cm print width. Do not add a title inside the image.

---

## Figure 2: AUROC distribution over runs (column width, `figures/fig2_auroc_distribution.png`)

**Caption.** Distribution of subject-level AUROC over the 15 runs (3 seeds × 5 folds) for each federated method. Boxes show the interquartile range and median, whiskers the range excluding outliers. FedX-Biomarker raises both the median and the lower tail, which removes the near-chance failures seen for FedAvg and FedProx.

**Source.** `results/figures/F3b_auroc_distribution.png`. The raw values are in `results/tables/fold_metrics_all_runs.csv`.

**AI / redraw prompt.**
> Redraw a box-and-whisker plot for a two-column IEEE paper, width 8.5 cm, height 5.5 cm, white background, Arial 8 pt. X-axis categories in this order: FedAvg, FedProx, FedAdam, q-FFL, FedX-Biomarker (ours). Y-axis "Subject-level AUROC", range 0.45–1.00, light horizontal grid lines at 0.1 steps. Use the 15 per-run AUROC values of each method exactly as given in fold_metrics_all_runs.csv (do not invent or smooth values). Summary statistics to verify: FedAvg mean 0.829 (min 0.508), FedProx mean 0.829 (min 0.504), FedAdam mean 0.844 (min 0.676), q-FFL mean 0.842 (min 0.714), FedX-Biomarker mean 0.875 (min 0.769). Baseline boxes in light grey with dark-grey outlines; FedX-Biomarker box in red #C44E52 with 40 % fill. Overlay individual runs as small jittered black dots (size 6). Add a dashed horizontal line at 0.5 labelled "chance". No title.

---

## Figure 3: Cumulative ablation (column width, `figures/fig3_cumulative_ablation.png`)

**Caption.** Cumulative ablation from traditional FedAvg (A0) to the full FedX-Biomarker (A4). Left: mean AUROC, BACC and MCC over 15 runs as each contribution is added. Right: trust metrics (ECE and sex EOD; lower is better). Architecture (C1) gives the largest single gain, and C2 and C3 add further gains on top of it.

**Source.** `results/figures/F11_cumulative_ablation.png`, from table T11.

**AI / redraw prompt.**
> Two side-by-side line panels for an IEEE column figure, total width 8.5 cm, height 4.5 cm, white background, Arial 8 pt. Shared x-axis with five ordered ticks: "A0 FedAvg", "A1 +C1", "A2 +C2a", "A3 +C2b", "A4 FedX". Left panel, y-axis "Score" 0.40–0.90, three lines with circle markers: AUROC = [0.829, 0.861, 0.867, 0.868, 0.875] (blue), BACC = [0.710, 0.750, 0.747, 0.764, 0.786] (green), MCC = [0.441, 0.514, 0.522, 0.548, 0.578] (red). Right panel, y-axis "Lower is better" 0.00–0.22, two lines with square markers: ECE = [0.175, 0.170, 0.173, 0.167, 0.156] (grey), sex EOD = [0.196, 0.146, 0.167, 0.180, 0.161] (orange). Annotate the A0→A1 step on the left panel with a small bracket "+C1: largest gain". Legends inside each panel, top-left, no frame. Use exactly these numbers.

---

## Figure 4: Knowledge-grounded biomarker concordance (column width, `figures/fig4_biomarker_concordance.png`)

**Caption.** Knowledge-grounded validation of discovered biomarkers. For each of the 14 stride features, the normalised federated attribution of FedX-Biomarker (blue) is shown next to the absolute Cliff's δ between PD patients and controls measured in the cohort (orange). The two leading features, stance proportion and swing-time asymmetry, coincide.

**Source.** `results/figures/F9_clinical_concordance.png`. The values are in `results/tables/T10_stride_biomarker_concordance.csv`.

**AI / redraw prompt.**
> Grouped bar chart for an IEEE column figure, width 8.5 cm, height 5 cm, white background, Arial 7–8 pt. 14 categories on the x-axis, ordered by model attribution, with 45° rotated labels: R stance %, swing asymmetry, peak force mean, L stance %, R swing mean, L swing mean, L stride CV, R stride mean, R stride CV, L stride mean, cadence, peak force CV, R swing CV, L swing CV. For each category draw two adjacent bars normalised to their maximum. Blue #4C72B0 = FedX attribution [0.172, 0.153, 0.096, 0.093, 0.092, 0.065, 0.046, 0.046, 0.044, 0.043, 0.042, 0.040, 0.035, 0.033] ÷ 0.172. Orange #DD8452 = |Cliff's δ| [0.483, 0.431, 0.293, 0.309, 0.110, 0.033, 0.279, 0.127, 0.259, 0.129, 0.127, 0.113, 0.316, 0.251] ÷ 0.483. Mark bars whose FDR-adjusted q < 0.05 with a small asterisk above the orange bar (R stance %, swing asymmetry, peak force mean, L stance %, L stride CV, R stride CV, R swing CV, L swing CV). Y-axis "Normalised value" 0–1.05. Legend top-right: "FedX attribution", "|Cliff's δ| (PD vs control)". Highlight the first two categories with a light-yellow background band labelled "top-2 agree". Use exactly these numbers.
