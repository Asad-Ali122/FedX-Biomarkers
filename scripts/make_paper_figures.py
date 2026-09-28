#!/usr/bin/env python
"""Copy the result figures used in the paper to their paper file names.

    python scripts/make_paper_figures.py --results results --dest paper_figures

Figure 1 (framework schematic) is drawn separately; see docs/FIGURE_PROMPTS.md.
"""
import argparse, os, shutil

MAP = {
    "F3b_auroc_distribution.png": "fig2_auroc_distribution.png",
    "F11_cumulative_ablation.png": "fig3_cumulative_ablation.png",
    "F9_clinical_concordance.png": "fig4_biomarker_concordance.png",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="output folder of fedx_biomarker.py / the notebook")
    ap.add_argument("--dest", default="paper_figures")
    a = ap.parse_args()
    src_dir = os.path.join(a.results, "figures")
    os.makedirs(a.dest, exist_ok=True)
    for src, dst in MAP.items():
        s = os.path.join(src_dir, src)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.dest, dst)); print(f"{src:32s} -> {dst}")
        else:
            print(f"missing: {s} (run the full experiment first)")
    print("Figure 1: create fig1_framework.pdf with the prompt in docs/FIGURE_PROMPTS.md")

if __name__ == "__main__":
    main()
