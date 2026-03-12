# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:00:59 2026

@author: JoanaCatarino
"""

from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from colors import STRAIN_COLORS



# ---------- Paths - Cohort dependent + where to ouput plots ----------

COHORT_DIRS = {
    1: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_1"),
    2: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_2"),
}


animals_info = Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/animals.xlsx")
OUTDIR = Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/plots/free_licking")
OUTDIR.mkdir(parents=True, exist_ok=True)


# OPTIONAL: exclude animals easily
exclude_animals = []  # e.g. ["986535"]



# ---------- Choose plot settings ----------

plot_config = {
    "figsize": (10, 6),
    "dpi": 500,

    "bar_color": "#9CC2BF",
    "bar_alpha": 0.75,
    "bar_width": 0.6,
    "capsize": 6,

    "y_lim": (0, 800),

    "tick_fontsize": 12,
    "label_fontsize": 14,
    "title_fontsize": 15,

    "dot_size": 80,
    "dot_edge_lw": 0.6,
    "dot_jitter": 0.18,

    "legend_fontsize": 12,
    "legend_title_fontsize": 13,

    # Legend outside, to the right, aligned near the x-axis (bottom)
    "legend_loc": "lower left",
    "legend_bbox": (1.02, 0.0),
}


# ---------- Helpers for data extraction ----------

def extract_datetime_from_filename(filename: str):
    m = re.search(r"_(\d{8})_(\d{6})_", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

def find_free_licking_files(meta_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, r in meta_df.iterrows():
        animal = str(r["animal"])
        cohort = int(r["cohort"])
        base_dir = COHORT_DIRS[cohort]

        behavior_dir = base_dir / animal / "Behavior"
        if not behavior_dir.exists():
            continue

        day_folders = [d for d in behavior_dir.iterdir() if d.is_dir()]
        for day in sorted(day_folders):
            csvs = list(day.glob("*.csv"))
            if not csvs:
                continue

            for csv_path in csvs:
                fname = csv_path.name
                if "FreeLick" not in fname:
                    continue

                dt = extract_datetime_from_filename(fname)
                records.append({
                    "animal": animal,
                    "cohort": cohort,
                    "dt": dt,
                    "path": str(csv_path),
                })

    if not records:
        raise FileNotFoundError("No FreeLick CSV files found.")

    df = pd.DataFrame(records).sort_values(["cohort", "animal", "dt"]).reset_index(drop=True)

    # day index within free licking, per animal (cohort doesn't matter here)
    df["day_in_task"] = df.groupby("animal").cumcount() + 1

    return df

_csv_cache = {}
def read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _csv_cache:
        _csv_cache[path] = pd.read_csv(path)
    return _csv_cache[path]

def plot_free_licking(metrics, title, out_png, out_svg, cfg, out_pdf=None):
    """
    metrics columns needed:
      - day_in_task (int)
      - licks (int/float)
      - strain (str)
    """
    if metrics.empty:
        print(f"⚠️ No data for: {title}")
        return

    # mean ± SEM per day
    stats = (
        metrics.groupby("day_in_task")["licks"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values("day_in_task")
    )

    fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=cfg["dpi"])

    x = np.arange(len(stats))

    ax.bar(
        x,
        stats["mean"],
        yerr=stats["sem"],
        capsize=cfg["capsize"],
        color=cfg["bar_color"],
        alpha=cfg["bar_alpha"],
        width=cfg["bar_width"],
    )

    # dots (colored by strain)
    for i, day in enumerate(stats["day_in_task"]):
        dd = metrics[metrics["day_in_task"] == day]
        for _, rr in dd.iterrows():
            jitter = (np.random.rand() - 0.5) * cfg["dot_jitter"]
            col = STRAIN_COLORS.get(rr["strain"], "#333333")
            ax.scatter(
                i + jitter,
                rr["licks"],
                s=cfg["dot_size"],
                color=col,
                edgecolor="white",
                linewidth=cfg["dot_edge_lw"],
                zorder=3,
            )

    # axes text sizes
    ax.set_xticks(x)
    ax.set_xticklabels([f"Day {d}" for d in stats["day_in_task"]],
                       fontsize=cfg["tick_fontsize"])
    ax.tick_params(axis="y", labelsize=cfg["tick_fontsize"])

    ax.set_ylabel("Number of licks", fontsize=cfg["label_fontsize"], labelpad=15)
    ax.set_title(title, fontsize=cfg["title_fontsize"], loc="center", pad=25)

    if cfg.get("y_lim"):
        ax.set_ylim(*cfg["y_lim"])

    # clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Legend: ONLY strains that appear in this plot ----
    strains_in_plot = (
        metrics["strain"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    legend_handles = []
    for s in sorted(strains_in_plot):
        if s in STRAIN_COLORS:  # only include known strains
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=STRAIN_COLORS[s],
                    markeredgecolor="white",
                    markersize=9,
                    label=s
                )
            )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Strain",
            frameon=False,
            fontsize=cfg["legend_fontsize"],
            title_fontsize=cfg["legend_title_fontsize"],
            loc=cfg["legend_loc"],
            bbox_to_anchor=cfg["legend_bbox"],
            borderaxespad=0.0,
        )

    # make room for the outside legend
    fig.tight_layout()
    fig.subplots_adjust(right=0.82)  # tweak if legend overlaps/clips

    plt.savefig(out_png, dpi=cfg["dpi"])
    plt.savefig(out_svg, dpi=cfg["dpi"])
    if out_pdf is not None:
        plt.savefig(out_pdf)  # pdf ignores dpi mostly; vector output

# -----------------------
# Main
# -----------------------
def main():
    meta = pd.read_excel(animals_info, dtype={"animal": str})
    meta.columns = [c.strip() for c in meta.columns]

    for col in ["animal", "strain", "cohort"]:
        if col not in meta.columns:
            raise ValueError(f"animals.csv missing required column: {col}")

    if exclude_animals:
        meta = meta[~meta["animal"].isin(exclude_animals)].copy()

    files = find_free_licking_files(meta)

    # merge strain
    files = files.merge(meta[["animal", "strain"]], on="animal", how="left")

    # per-session licks
    recs = []
    for _, r in files.iterrows():
        df = read_csv_cached(r["path"])
        if "lick" not in df.columns:
            raise ValueError(f"Missing 'lick' column in: {r['path']}")

        licks = int((df["lick"] == 1).sum())

        recs.append({
            "animal": r["animal"],
            "strain": r["strain"],
            "cohort": int(r["cohort"]),
            "day_in_task": int(r["day_in_task"]),
            "dt": r["dt"],
            "licks": licks,
        })

    metrics = pd.DataFrame(recs).sort_values(["cohort", "animal", "day_in_task"])
    metrics.to_csv(OUTDIR / "free_licking_session_metrics.csv", index=False)

    # --------- Plot cohort 1 ---------
    plot_free_licking(
    metrics[metrics["cohort"] == 1],
    "Free Licking (Cohort 1) — Total licks per day (mean ± SEM)",
    OUTDIR / "free_licking_cohort1.png",
    OUTDIR / "free_licking_cohort1.svg",
    plot_config,
    out_pdf=OUTDIR / "free_licking_cohort1.pdf"
    )

    plot_free_licking(
    metrics[metrics["cohort"] == 2],
    "Free Licking (Cohort 2) — Total licks per day (mean ± SEM)",
    OUTDIR / "free_licking_cohort2.png",
    OUTDIR / "free_licking_cohort2.svg",
    plot_config,
    out_pdf=OUTDIR / "free_licking_cohort2.pdf"
    )
    
    plot_free_licking(
    metrics,
    "Free Licking (All Cohorts) — Total licks per day (mean ± SEM)",
    OUTDIR / "free_licking_all_cohorts.png",
    OUTDIR / "free_licking_all_cohorts.svg",
    plot_config,
    out_pdf=OUTDIR / "free_licking_all_cohorts.pdf"
    )

    print("Done. Outputs in:", OUTDIR)

if __name__ == "__main__":
    main()










