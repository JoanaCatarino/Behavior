# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 08:14:16 2026

@author: JoanaCatarino
"""

from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Choose Paths ----------

COHORT_DIRS = {
    1: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_1"),
    2: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_2"),
}

SESSIONS_XLSX = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/adaptive_sessions_to_plot.xlsx"
)

OUTDIR = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/plots/adaptive_task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# --------- Choose colors for plots ----------

STRAIN_COLORS = {
    "Tlx3": "#7ABEC6",
    "Fezf2": "#A388B1",
    "Fmr1-Fezf2": "#B1536F",
    "Fmr1-Tlx3": "#B7CB92",
}


# ---------- Choose global plot configurations ----------

plot_config = {

    # figure
    "dpi": 500,

    # fonts
    "tick_fontsize": 12,
    "label_fontsize": 14,
    "title_fontsize": 15,

    # padding    
    "ylabel_pad": 14,
    "subplot_wspace": 0.6,

    # legend
    "legend_fontsize": 12,
    "legend_title_fontsize": 13,
    "legend_loc": "lower left",
    "legend_bbox": (1.02, 0.0),

}


#%%
# ============================================================
# LOAD SPOUT–TONE MAP
# ============================================================

mapping_file = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Spout-tone map/spout_tone_generator.csv'
)

spout_map_df = pd.read_csv(mapping_file)


def get_tone_map(animal_id):
    """
    Returns dict:
    {'8KHz': 'left'/'right', '16KHz': 'left'/'right'}
    """
    row = spout_map_df[spout_map_df["Animal"] == int(animal_id)].iloc[0]

    return {
        "8KHz": row["8KHz"],
        "16KHz": row["16KHz"]
    }

#%%
# ============================================================
# HELPERS
# ============================================================

def extract_datetime_from_filename(filename: str):
    match = re.search(r"AdaptSensorimotor_(\d+)_(\d{8})_(\d{6})", filename)
    if not match:
        return None
    return datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M%S")


def extract_session_date_from_filename(filename: str):
    match = re.search(r"AdaptSensorimotor_(\d+)_(\d{8})_(\d{6})", filename)
    if not match:
        return None
    return match.group(2)


def find_adaptive_files(meta_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    unique_animals = meta_df[["animal", "cohort"]].drop_duplicates()

    for _, row in unique_animals.iterrows():
        animal = str(row["animal"]).strip()
        cohort = int(row["cohort"])

        if cohort not in COHORT_DIRS:
            print(f"Warning: cohort {cohort} not found for animal {animal}")
            continue

        behavior_dir = COHORT_DIRS[cohort] / animal / "Behavior"

        if not behavior_dir.exists():
            print(f"Warning: missing Behavior folder for animal {animal} in cohort {cohort}")
            continue

        for day_folder in sorted([d for d in behavior_dir.iterdir() if d.is_dir()]):
            for csv_path in sorted(day_folder.glob("AdaptSensorimotor*.csv")):
                filename = csv_path.name
                dt = extract_datetime_from_filename(filename)
                session_date = extract_session_date_from_filename(filename)

                if dt is None or session_date is None:
                    print(f"Could not parse filename: {filename}")
                    continue

                records.append({
                    "animal": animal,
                    "cohort": cohort,
                    "session_date": session_date,
                    "datetime": dt,
                    "path": str(csv_path),
                    "filename": filename,
                })

    if not records:
        raise FileNotFoundError("No AdaptSensorimotor csv files were found.")

    files_df = pd.DataFrame(records)
    files_df = files_df.sort_values(["animal", "session_date", "datetime"]).reset_index(drop=True)
    return files_df


_csv_cache = {}

def read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _csv_cache:
        _csv_cache[path] = pd.read_csv(path)
    return _csv_cache[path]


# ============================================================
# METRICS
# ============================================================

def compute_session_metrics(row):
    df = read_csv_cached(row["path"]).copy()

    out = {
        "animal": row["animal"],
        "sex": row["sex"],
        "strain": row["strain"],
        "cohort": row["cohort"],
        "session_date": row["session_date"],
        "stage": row["stage"],
        "filename": row["filename"],
    }

    out["total_trials"] = len(df)

    if "block" in df.columns:
        block_counts = {"sound": 0, "action-right": 0, "action-left": 0}
        prev = None

        for b in df["block"]:
            if b != prev:
                if b in block_counts:
                    block_counts[b] += 1
                prev = b

        out["n_blocks"] = sum(block_counts.values())
        out["n_sound_blocks"] = block_counts["sound"]
        out["n_action_right_blocks"] = block_counts["action-right"]
        out["n_action_left_blocks"] = block_counts["action-left"]

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        trials_per_block = (
            df.groupby(["block_index", "block"])
              .size()
              .reset_index(name="n_trials")
        )
        out["mean_trials_until_switch"] = trials_per_block["n_trials"].mean()
    else:
        out["n_blocks"] = np.nan
        out["n_sound_blocks"] = np.nan
        out["n_action_right_blocks"] = np.nan
        out["n_action_left_blocks"] = np.nan
        out["mean_trials_until_switch"] = np.nan

    if {"lick_time", "RW_start"}.issubset(df.columns):
        out["mean_latency"] = (df["lick_time"] - df["RW_start"]).mean()
    else:
        out["mean_latency"] = np.nan

    if {"omission", "catch_trial"}.issubset(df.columns):
        out["n_omissions"] = int(((df["omission"] == 1) & (df["catch_trial"] == 0)).sum())
    else:
        out["n_omissions"] = np.nan

    return out


#%% 1- Triangle Summary plots 

def make_blocktypes_df(stage_df):
    df_blocktypes = stage_df[
        [
            "animal",
            "strain",
            "stage",
            "session_date",
            "n_sound_blocks",
            "n_action_right_blocks",
            "n_action_left_blocks",
        ]
    ].copy()

    df_blocktypes = df_blocktypes.rename(columns={
        "n_sound_blocks": "sound",
        "n_action_right_blocks": "action-right",
        "n_action_left_blocks": "action-left",
    })

    return df_blocktypes.melt(
        id_vars=["animal", "strain", "stage", "session_date"],
        value_vars=["sound", "action-right", "action-left"],
        var_name="block",
        value_name="count"
    )


def plot_triangle_summary(metrics_df, stage, outdir, strain_colors):
    df_sessions = metrics_df[metrics_df["stage"] == stage].copy()

    if df_sessions.empty:
        print(f"No data for stage: {stage}")
        return

    df_blocktypes = make_blocktypes_df(df_sessions)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=plot_config["dpi"])
    
    plt.subplots_adjust(
    wspace=plot_config["subplot_wspace"])

    # Panel 1
    ax1 = axes[0]
    for _, tr in df_sessions.iterrows():
        j = (rng.random() - 0.5) * 0.20
        color = strain_colors.get(tr["strain"], "#333333")
        ax1.scatter(
            1 + j, tr["total_trials"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    ax1.errorbar(
        1.30,
        df_sessions["total_trials"].mean(),
        yerr=df_sessions["total_trials"].sem(),
        fmt="^", markersize=12, color="black", capsize=4
    )
    ax1.set_xlim(0.7, 1.5)
    ax1.set_ylim(500, 800)
    ax1.set_xticks([])
    ax1.tick_params(labelsize=plot_config["tick_fontsize"])
    ax1.set_ylabel(
    "Trials performed",
    fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"]
    )
    ax1.set_title(stage.capitalize())
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2
    ax2 = axes[1]
    for _, tr in df_sessions.iterrows():
        j = (rng.random() - 0.5) * 0.20
        color = strain_colors.get(tr["strain"], "#333333")
        ax2.scatter(
            1 + j, tr["n_blocks"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    ax2.errorbar(
        1.30,
        df_sessions["n_blocks"].mean(),
        yerr=df_sessions["n_blocks"].sem(),
        fmt="^", markersize=12, color="black", capsize=4
    )
    ax2.set_xlim(0.7, 1.5)
    ax2.set_ylim(0, 26)
    ax2.set_xticks([])
    ax2.tick_params(labelsize=plot_config["tick_fontsize"])
    ax2.set_ylabel(
    "Number of Blocks",
    fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"]
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3
    ax3 = axes[2]
    block_order = ["sound", "action-right", "action-left"]
    spacing = 2
    data_x = np.arange(3) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2

    block_positions = dict(zip(block_order, data_x))

    for _, tr in df_blocktypes.iterrows():
        j = (rng.random() - 0.5) * 0.20
        color = strain_colors.get(tr["strain"], "#333333")
        ax3.scatter(
            block_positions[tr["block"]] + j,
            tr["count"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    stats = (
        df_blocktypes.groupby("block")["count"]
        .agg(["mean", "sem"])
        .reindex(block_order)
    )

    
    ax3.errorbar(
    mean_x,
    stats["mean"].values,
    yerr=stats["sem"].values,
    fmt="^",
    markersize=12,
    color="black",
    capsize=4
    )
    
    ax3.set_xticks(label_x)
    ax3.set_xticklabels(["Sound", "Action-R", "Action-L"],
    fontsize=plot_config["label_fontsize"])
    ax3.tick_params(labelsize=plot_config["tick_fontsize"])
    ax3.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax3.set_ylim(0, 15)
    ax3.set_xticks(label_x)
    ax3.set_xticklabels(["Sound", "Action-R", "Action-L"])
    ax3.set_ylabel(
    "Blocks per type",
    fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"]
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_sessions["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    #plt.savefig(outdir / f"Adaptive_TriangleSummary_{stage}.png", dpi=600, bbox_inches="tight")
    #plt.savefig(outdir / f"Adaptive_TriangleSummary_{stage}.svg", dpi=600, bbox_inches="tight")
    #plt.savefig(outdir / f"Adaptive_TriangleSummary_{stage}.pdf", bbox_inches="tight")

#%% Performance and performance per block

# ============================================================
# PERFORMANCE TABLES
# ============================================================

def compute_performance_tables(sessions_df):
    print("\n=== Computing performance tables ===")

    performance_records = []
    perf_block_records = []

    block_types = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        # ----------------------------
        # PERFORMANCE PER SESSION
        # ----------------------------
        needed_session = ["reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()
            n_omissions = df[
                (df["omission"] == 1) & (df["catch_trial"] == 0)
            ].shape[0]

            denom = n_correct + n_incorrect + n_omissions
            performance = (n_correct / denom * 100) if denom > 0 else np.nan

            performance_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "correct": n_correct,
                "incorrect": n_incorrect,
                "omissions": n_omissions,
                "performance": performance,
            })
        else:
            print(f"⚠ Missing columns for session performance in {row['filename']}")

        # ----------------------------
        # PERFORMANCE PER BLOCK TYPE
        # ----------------------------
        needed_block = ["block", "reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]

                if df_blk.empty:
                    continue

                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()
                n_omissions = df_blk[
                    (df_blk["omission"] == 1) & (df_blk["catch_trial"] == 0)
                ].shape[0]

                denom = n_correct + n_incorrect + n_omissions
                perf = (n_correct / denom * 100) if denom > 0 else np.nan

                perf_block_records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk,
                    "performance": perf,
                })
        else:
            print(f"⚠ Missing columns for block performance in {row['filename']}")

    perf_df = pd.DataFrame(performance_records)
    perf_block_df = pd.DataFrame(perf_block_records)

    print("\n=== PERFORMANCE TABLE ===")
    print(perf_df.head())

    print("\n=== PERFORMANCE PER BLOCK TYPE ===")
    print(perf_block_df.head())

    return perf_df, perf_block_df


# ============================================================
# PERFORMANCE PER SESSION
# ============================================================

def plot_performance_per_session(perf_df, outdir, strain_colors, stage):
    df_stage = perf_df[perf_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No performance data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=plot_config["dpi"])

    data_x = 0.9
    mean_x = 1.1

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            data_x + jitter,
            row["performance"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    ax.errorbar(
        mean_x,
        df_stage["performance"].mean(),
        yerr=df_stage["performance"].sem(),
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    ax.axhline(50, linestyle="--", color="black", alpha=0.6)

    ax.set_xlim(0.75, 1.2)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Performance (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Performance per Session — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_PerformancePerSession_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_PerformancePerSession_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_PerformancePerSession_{stage}.pdf", bbox_inches="tight")


# ============================================================
# PERFORMANCE PER BLOCK TYPE
# ============================================================

def plot_performance_per_blocktype(perf_block_df, outdir, strain_colors, stage):
    df_stage = perf_block_df[perf_block_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No block performance data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    block_order = ["sound", "action-right", "action-left"]

    spacing = 2
    data_x = np.arange(len(block_order)) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2

    block_positions = dict(zip(block_order, data_x))

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            block_positions[row["block_type"]] + jitter,
            row["performance"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    stats = (
        df_stage.groupby("block_type")["performance"]
        .agg(["mean", "sem"])
        .reindex(block_order)
    )

    ax.errorbar(
        mean_x,
        stats["mean"].values,
        yerr=stats["sem"].values,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    ax.axhline(50, linestyle="--", color="gray", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax.set_xticks(label_x)
    ax.set_xticklabels(
        ["Sound", "Action-R", "Action-L"],
        fontsize=plot_config["label_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Performance (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Performance per Block Type — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_Performance_BlockTypes_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_Performance_BlockTypes_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_Performance_BlockTypes_{stage}.pdf", bbox_inches="tight")
    
    
#%% trials until switch 

# ============================================================
# TRIALS UNTIL SWITCH PER BLOCK TYPE
# ============================================================

def compute_trials_until_switch_tables(sessions_df):
    print("\n=== Computing trials until switch per block type ===")

    block_order = ["sound", "action-right", "action-left"]
    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        if "block" not in df.columns:
            print(f"⚠ Missing 'block' column in {row['filename']}")
            continue

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        for blk_idx in df["block_index"].unique():
            df_blk = df[df["block_index"] == blk_idx]

            if df_blk.empty:
                continue

            blk_type = df_blk["block"].iloc[0]
            if blk_type not in block_order:
                continue

            n_trials = len(df_blk)

            records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block_type": blk_type,
                "n_trials": n_trials,
            })

    tri_df = pd.DataFrame(records)

    if tri_df.empty:
        print("⚠ No trials-until-switch records computed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    session_means_df = (
        tri_df.groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "block_type"]
        )["n_trials"]
        .mean()
        .reset_index()
    )

    overall_stats_df = (
        tri_df.groupby(["stage", "block_type"])["n_trials"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    print("\n=== TRIALS UNTIL SWITCH TABLE ===")
    print(tri_df.head())

    print("\n=== SESSION MEANS TABLE ===")
    print(session_means_df.head())

    return tri_df, session_means_df, overall_stats_df

# ============================================================
# PLOT — TRIALS UNTIL SWITCH PER BLOCK TYPE
# ============================================================

def plot_trials_until_switch_blocktype(session_means_df, overall_stats_df, outdir, strain_colors, stage):
    df_stage = session_means_df[session_means_df["stage"] == stage].copy()
    stats_stage = overall_stats_df[overall_stats_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No trials-until-switch data for stage: {stage}")
        return

    block_order = ["sound", "action-right", "action-left"]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    spacing = 2
    data_x = np.arange(len(block_order)) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2
    block_positions = dict(zip(block_order, data_x))

    # ----------------------------------------------------
    # Per-session means
    # ----------------------------------------------------
    for _, row in df_stage.iterrows():
        blk = row["block_type"]
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            block_positions[blk] + jitter,
            row["n_trials"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    # ----------------------------------------------------
    # Overall mean ± SEM
    # ----------------------------------------------------
    stats_stage = (
        stats_stage.set_index("block_type")
        .reindex(block_order)
        .reset_index()
    )

    ax.errorbar(
        mean_x,
        stats_stage["mean"].values,
        yerr=stats_stage["sem"].values,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    # reference line at 20
    ax.axhline(20, color="black", linestyle="--", linewidth=1.5)

    # ----------------------------------------------------
    # Formatting
    # ----------------------------------------------------
    ax.set_ylim(0, 350)
    ax.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax.set_xticks(label_x)
    ax.set_xticklabels(
        ["Sound", "Action-R", "Action-L"],
        fontsize=plot_config["label_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Trials until switch",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Trials Until Block Switch — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Triangle_TrialsUntilSwitch_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Triangle_TrialsUntilSwitch_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Triangle_TrialsUntilSwitch_{stage}.pdf", bbox_inches="tight")

#%% Perseverative errors

# ============================================================
# PERSEVERATIVE ERRORS AROUND BLOCK SWITCHES
# ============================================================

def correct_side(block_type, stim, tone_map):
    if block_type == "sound":
        return tone_map[stim]
    elif block_type == "action-left":
        return "left"
    elif block_type == "action-right":
        return "right"
    return np.nan


def classify_pe_after_switch(row, new_block, prev_block, tone_map):
    """
    CLASSIFICATION RULES

    A→S : PE if habitual_side == prev_block_correct_side
    S→A : PE if habitual_side != new_block_correct_side
    """
    if row["omission"] == 1:
        return np.nan

    # determine stimulus
    if "8KHz" in row.index and row["8KHz"] == 1:
        stim = "8KHz"
    elif "16KHz" in row.index and row["16KHz"] == 1:
        stim = "16KHz"
    else:
        return np.nan

    habitual = tone_map[stim]
    new_correct = correct_side(new_block, stim, tone_map)
    prev_correct = correct_side(prev_block, stim, tone_map)

    # hit = never PE
    if row["reward"] == 1:
        return 0

    # incorrect trial
    if row["punishment"] == 1:

        # ACTION → SOUND
        if new_block == "sound":
            return 1 if habitual == prev_correct else 0

        # SOUND → ACTION
        if prev_block == "sound":
            return 1 if habitual != new_correct else 0

    return np.nan


def process_session_pe_counts_fixed(row, window=20):

    try:
        df = read_csv_cached(row["path"]).copy()
    except Exception as e:
        print(f"⚠ Could not read {row['filename']}: {e}")
        return None

    needed_cols = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
    if not needed_cols.issubset(df.columns):
        print(f"⚠ Missing columns for PE analysis in {row['filename']}")
        return None

    # --------------------------------------------------------
    # Get tone mapping from external file
    # --------------------------------------------------------
    tone_map = get_tone_map(row["animal"])

    A2S_count = 0
    S2A_count = 0

    for i in range(1, len(df)):
        prev_block = df.loc[i - 1, "block"]
        new_block = df.loc[i, "block"]

        if prev_block == new_block:
            continue

        # identify transition
        if prev_block in ["action-left", "action-right"] and new_block == "sound":
            direction = "Action→Sound"
        elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
            direction = "Sound→Action"
        else:
            continue

        # next N trials only
        next_trials = df.iloc[i:i + window].copy()

        for _, tr in next_trials.iterrows():
            pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)
            if not np.isnan(pe):
                if direction == "Action→Sound":
                    A2S_count += pe
                else:
                    S2A_count += pe

    return {
        "Action→Sound": A2S_count,
        "Sound→Action": S2A_count,
    }


def compute_pe_tables(sessions_df, window=20):
    print("\n=== Computing perseverative errors around block switches ===")

    pe_records = []

    for _, row in sessions_df.iterrows():
        res = process_session_pe_counts_fixed(row, window=window)
        if res is None:
            continue

        pe_records.append({
            "animal": row["animal"],
            "session_date": row["session_date"],
            "session_id": f"{row['animal']}_{row['session_date']}",
            "strain": row["strain"],
            "stage": row["stage"],
            "Action→Sound": res["Action→Sound"],
            "Sound→Action": res["Sound→Action"],
        })

    pe_df = pd.DataFrame(pe_records)

    if pe_df.empty:
        print("⚠ No PE data available.")
    else:
        print("\n=== PE TABLE ===")
        print(pe_df.head())

    return pe_df
    
# ============================================================
# TRIANGLE PLOT — PERSEVERATIVE ERRORS
# ============================================================

def plot_pe_triangle(pe_df, outdir, strain_colors, stage):
    df_stage = pe_df[pe_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No PE data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    x_positions = np.array([0.7, 1.25])
    mean_x = x_positions + 0.18
    label_x = (x_positions + mean_x) / 2
    x_labels = ["Action→Sound", "Sound→Action"]
    ax.set_xlim(0.55, 1.55)

    # per-session points
    for _, row in df_stage.iterrows():
        color = strain_colors.get(row["strain"], "gray")

        j1 = (rng.random() - 0.5) * 0.06
        j2 = (rng.random() - 0.5) * 0.06

        ax.scatter(
            x_positions[0] + j1,
            row["Action→Sound"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

        ax.scatter(
            x_positions[1] + j2,
            row["Sound→Action"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    # grand means
    mean_A2S = df_stage["Action→Sound"].mean()
    mean_S2A = df_stage["Sound→Action"].mean()
    
    # compute SEM
    sem_A2S = df_stage["Action→Sound"].sem()
    sem_S2A = df_stage["Sound→Action"].sem()

    # grand means ± SEM
    ax.errorbar(
        mean_x[0],
        mean_A2S,
        yerr=sem_A2S,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
        zorder=5
    )
    
    ax.errorbar(
        mean_x[1],
        mean_S2A,
        yerr=sem_S2A,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
        zorder=5
    )

    # formatting
    ax.set_xticks(label_x)
    ax.set_xticklabels(x_labels, fontsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Number of Perseverative Errors",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Perseverative Errors Around Block Switches — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]


    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_PE_TrianglePlot_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_PE_TrianglePlot_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_PE_TrianglePlot_{stage}.pdf", bbox_inches="tight")

#%%


# ============================================================
# PE TIME COURSE AFTER SWITCH
# ============================================================

def compute_pe_timecourse_tables(sessions_df, window=20):
    """
    Builds one row per trial-position after a valid block switch.
    Output columns include:
    animal, session_date, session_id, strain, stage,
    switch_type (A→S or S→A), trial_from_switch, pe
    """
    print("\n=== Computing PE time course after switch ===")

    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_cols = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
        if not needed_cols.issubset(df.columns):
            print(f"⚠ Missing columns for PE time course in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        for i in range(1, len(df)):
            prev_block = df.loc[i - 1, "block"]
            new_block = df.loc[i, "block"]

            if prev_block == new_block:
                continue

            # Identify switch type
            if prev_block in ["action-left", "action-right"] and new_block == "sound":
                switch_type = "Action→Sound"
            elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
                switch_type = "Sound→Action"
            else:
                continue

            next_trials = df.iloc[i:i + window].copy()

            for k, (_, tr) in enumerate(next_trials.iterrows(), start=1):
                pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)

                # Keep omissions as NaN, valid classified trials as 0/1
                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "switch_type": switch_type,
                    "trial_from_switch": k,
                    "pe": pe,
                })

    pe_timecourse_df = pd.DataFrame(records)

    if pe_timecourse_df.empty:
        print("⚠ No PE time course data computed.")
        return pd.DataFrame(), pd.DataFrame()

    # Mean PE rate per session, per switch, per trial position
    session_timecourse_df = (
        pe_timecourse_df
        .groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "switch_type", "trial_from_switch"],
            dropna=False
        )["pe"]
        .mean()
        .reset_index()
    )

    print("\n=== PE TIME COURSE TABLE ===")
    print(pe_timecourse_df.head())

    print("\n=== SESSION PE TIME COURSE TABLE ===")
    print(session_timecourse_df.head())

    return pe_timecourse_df, session_timecourse_df

# ============================================================
# PLOT — PE TIME COURSE AFTER SWITCH
# ============================================================

def plot_pe_timecourse(session_timecourse_df, outdir, strain_colors, stage):
    df_stage = session_timecourse_df[session_timecourse_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No PE time course data for stage: {stage}")
        return

    switch_order = ["Action→Sound", "Sound→Action"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df_stage[df_stage["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

    

        # thick mean line + SEM shading per strain
        for strain, strain_df in df_sw.groupby("strain"):
            stats_df = (
                strain_df.groupby("trial_from_switch")["pe"]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("trial_from_switch")
            )

            x = stats_df["trial_from_switch"].values
            y = stats_df["mean"].values * 100
            sem = stats_df["sem"].fillna(0).values * 100
            color = strain_colors.get(strain, "black")

            ax.plot(
                x,
                y,
                linewidth=2.8,
                color=color,
                label=strain
            )

            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=color,
                alpha=0.18
            )


        max_trial = int(df_sw["trial_from_switch"].max())

        ax.set_xlim(0, max_trial)

        # integer ticks
        ax.set_xticks(range(0, max_trial + 1, 2))   
    
        # ticks every 2 trials
        
        ax.set_ylim(0, 100)
        ax.set_xlabel("Trial after switch", fontsize=plot_config["label_fontsize"], labelpad=13)
        ax.set_title(switch_type, fontsize=plot_config["title_fontsize"], pad=15)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Perseverative errors (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D(
            [0], [0],
            color=color,
            linewidth=2.8,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    fig.suptitle(
        f"PE Time Course After Switch — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"], y=1.02
    )

    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_PE_TimeCourse_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_PE_TimeCourse_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_PE_TimeCourse_{stage}.pdf", bbox_inches="tight")


#%%

# ============================================================
# LATENCY HELPERS
# ============================================================

def correct_side(block, stim, tone_map):
    """Correct response side for this block type."""
    if block == "sound":
        return tone_map[stim]
    elif block == "action-left":
        return "left"
    elif block == "action-right":
        return "right"
    return None


def opposite(side):
    return "right" if side == "left" else "left"


    
# ============================================================
# LATENCY TABLE
# ============================================================

def compute_latency_table(sessions_df):
    print("\n=== Computing latency table ===")

    latency_records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required = [
            "block", "lick_time", "RW_start",
            "reward", "punishment", "omission",
            "8KHz", "16KHz"
        ]

        if not all(c in df.columns for c in required):
            print(f"⚠ Missing columns in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        # latency
        df["latency"] = df["lick_time"] - df["RW_start"]

        # stimulus
        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else None),
            axis=1
        )

        for _, r in df.iterrows():
            if pd.isna(r["stim"]) or r["omission"] == 1:
                continue

            blk = r["block"]
            stim = r["stim"]
            lat = r["latency"]

            if pd.isna(lat):
                continue

            cside = correct_side(blk, stim, tone_map)
            if cside is None:
                continue

            if r["reward"] == 1:
                lick_side = cside
                correctness = "correct"
            elif r["punishment"] == 1:
                lick_side = opposite(cside)
                correctness = "incorrect"
            else:
                continue

            latency_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block": blk,
                "side": lick_side,
                "correctness": correctness,
                "latency": lat
            })

    lat_df = pd.DataFrame(latency_records)

    if lat_df.empty:
        print("⚠ No latency data computed.")
        return pd.DataFrame()

    valid_blocks = ["sound", "action-left", "action-right"]
    lat_df = lat_df[lat_df["block"].isin(valid_blocks)].copy()

    print("\n=== LATENCY TABLE ===")
    print(lat_df.head())

    return lat_df


# ============================================================
# 4-PANEL LATENCY PLOT
# ============================================================

def plot_latency_panels_by_stage(lat_df, outdir, strain_colors, stage):
    df_stage = lat_df[lat_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No latency data for stage: {stage}")
        return

    categories = [
        ("sound", "correct", "Sound — Correct"),
        ("sound", "incorrect", "Sound — Incorrect"),
        ("action-right", "all", "Action-Right — All trials"),
        ("action-left", "all", "Action-Left — All trials"),
    ]

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), dpi=plot_config["dpi"])
    plt.subplots_adjust(wspace=0.45)

    for ax, (blk, cond, panel_title) in zip(axes, categories):
        x_left = 0
        x_right = 1

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"], fontsize=plot_config["tick_fontsize"])

        for (animal, sess), df_sess in df_stage.groupby(["animal", "session_date"]):
            d = df_sess[df_sess["block"] == blk].copy()

            if cond != "all":
                d = d[d["correctness"] == cond]

            if d.empty:
                continue

            strain = str(d["strain"].iloc[0])
            color = strain_colors.get(strain, "black")

            left_vals = d[d["side"] == "left"]["latency"]
            right_vals = d[d["side"] == "right"]["latency"]

            left_mean = left_vals.mean() if len(left_vals) else np.nan
            right_mean = right_vals.mean() if len(right_vals) else np.nan

            # connect if both sides exist
            if not pd.isna(left_mean) and not pd.isna(right_mean):
                ax.plot(
                    [x_left, x_right],
                    [left_mean, right_mean],
                    color=color,
                    linewidth=1.2,
                    alpha=0.35
                )

            # scatter points
            if not pd.isna(left_mean):
                ax.scatter(
                    x_left,
                    left_mean,
                    color=color,
                    s=35,
                    alpha=0.35,
                    edgecolor="none",
                    zorder=2
                )

            if not pd.isna(right_mean):
                ax.scatter(
                    x_right,
                    right_mean,
                    color=color,
                    s=35,
                    alpha=0.35,
                    edgecolor="none",
                    zorder=2
                )
                
                
        # ----------------------------------------------------
        # Thick strain mean overlays
        # ----------------------------------------------------
        d_panel = df_stage[df_stage["block"] == blk].copy()
        if cond != "all":
            d_panel = d_panel[d_panel["correctness"] == cond]
    
        for strain, d_strain in d_panel.groupby("strain"):
            color = strain_colors.get(strain, "black")
    
            left_vals = d_strain[d_strain["side"] == "left"]["latency"]
            right_vals = d_strain[d_strain["side"] == "right"]["latency"]
    
            left_mean = left_vals.mean() if len(left_vals) else np.nan
            right_mean = right_vals.mean() if len(right_vals) else np.nan
    
            if not pd.isna(left_mean) and not pd.isna(right_mean):
                ax.plot(
                    [x_left, x_right],
                    [left_mean, right_mean],
                    color=color,
                    linewidth=3.2,
                    alpha=1.0,
                    zorder=4
                )
    
            if not pd.isna(left_mean):
                ax.scatter(
                    x_left,
                    left_mean,
                    color=color,
                    s=90,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5
                )
    
            if not pd.isna(right_mean):
                ax.scatter(
                    x_right,
                    right_mean,
                    color=color,
                    s=90,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5
                )        
                
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.set_title(panel_title, fontsize=plot_config["title_fontsize"], pad=13)
        ax.set_ylabel(
            "Mean lick latency (s)",
            fontsize=plot_config["label_fontsize"],
            labelpad=plot_config["ylabel_pad"]
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.spines["top"].set_visible(False) 
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="-",
            color=color,
            markersize=6,
            linewidth=2,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    fig.suptitle(
        f"Latency per Spout Side — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"], y=1.03
    )

    plt.tight_layout()
    # plt.savefig(outdir / f"Latency_4Panels_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Latency_4Panels_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Latency_4Panels_{stage}.pdf", bbox_inches="tight")


# ============================================================
# LATENCY TIME COURSE AROUND BLOCK SWITCH
# ============================================================

def compute_latency_timecourse_tables(sessions_df, window=20):
    print("\n=== Computing latency time course after block switch ===")

    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required = [
            "block", "lick_time", "RW_start",
            "reward", "punishment", "omission",
            "8KHz", "16KHz"
        ]
        if not all(c in df.columns for c in required):
            print(f"⚠ Missing columns in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        # latency
        df["latency"] = df["lick_time"] - df["RW_start"]

        # stimulus
        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        # iterate over switches
        for i in range(1, len(df)):
            prev_block = df.loc[i - 1, "block"]
            new_block = df.loc[i, "block"]

            if prev_block == new_block:
                continue

            if prev_block in ["action-left", "action-right"] and new_block == "sound":
                switch_type = "A→S"
            elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
                switch_type = "S→A"
            else:
                continue

            next_trials = df.iloc[i:i + window].copy()

            for k, (_, tr) in enumerate(next_trials.iterrows(), start=1):
                if pd.isna(tr["stim"]) or tr["omission"] == 1:
                    continue

                stim = tr["stim"]
                blk = tr["block"]
                lat = tr["latency"]

                if pd.isna(lat):
                    continue

                cside = correct_side(blk, stim, tone_map)
                if cside is None:
                    continue

                if tr["reward"] == 1:
                    lick_side = cside
                    correctness = "correct"
                elif tr["punishment"] == 1:
                    lick_side = opposite(cside)
                    correctness = "incorrect"
                else:
                    continue

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "switch_type": switch_type,
                    "trial_from_switch": k,
                    "latency": lat,
                    "correctness": correctness,
                    "lick_side": lick_side,
                })

    lat_timecourse_df = pd.DataFrame(records)

    if lat_timecourse_df.empty:
        print("⚠ No latency time course data computed.")
        return pd.DataFrame(), pd.DataFrame()

    session_latency_timecourse_df = (
        lat_timecourse_df
        .groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "switch_type", "trial_from_switch"],
            dropna=False
        )["latency"]
        .mean()
        .reset_index()
    )

    print("\n=== LATENCY TIME COURSE TABLE ===")
    print(lat_timecourse_df.head())

    print("\n=== SESSION LATENCY TIME COURSE TABLE ===")
    print(session_latency_timecourse_df.head())

    return lat_timecourse_df, session_latency_timecourse_df

#%%

# ============================================================
# LATENCY TIME COURSE AROUND BLOCK SWITCH
# ============================================================

def compute_latency_timecourse_tables(sessions_df, window=20):
    print("\n=== Computing latency time course after block switch ===")

    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required = [
            "block", "lick_time", "RW_start",
            "reward", "punishment", "omission",
            "8KHz", "16KHz"
        ]
        if not all(c in df.columns for c in required):
            print(f"⚠ Missing columns in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        # latency
        df["latency"] = df["lick_time"] - df["RW_start"]

        # stimulus
        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        # iterate over switches
        for i in range(1, len(df)):
            prev_block = df.loc[i - 1, "block"]
            new_block = df.loc[i, "block"]

            if prev_block == new_block:
                continue

            if prev_block in ["action-left", "action-right"] and new_block == "sound":
                switch_type = "Action→Sound"
            elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
                switch_type = "Sound→Action"
            else:
                continue

            next_trials = df.iloc[i:i + window].copy()

            for k, (_, tr) in enumerate(next_trials.iterrows(), start=1):
                if pd.isna(tr["stim"]) or tr["omission"] == 1:
                    continue

                stim = tr["stim"]
                blk = tr["block"]
                lat = tr["latency"]

                if pd.isna(lat):
                    continue

                cside = correct_side(blk, stim, tone_map)
                if cside is None:
                    continue

                if tr["reward"] == 1:
                    lick_side = cside
                    correctness = "correct"
                elif tr["punishment"] == 1:
                    lick_side = opposite(cside)
                    correctness = "incorrect"
                else:
                    continue

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "switch_type": switch_type,
                    "trial_from_switch": k,
                    "latency": lat,
                    "correctness": correctness,
                    "lick_side": lick_side,
                })

    lat_timecourse_df = pd.DataFrame(records)

    if lat_timecourse_df.empty:
        print("⚠ No latency time course data computed.")
        return pd.DataFrame(), pd.DataFrame()

    session_latency_timecourse_df = (
        lat_timecourse_df
        .groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "switch_type", "trial_from_switch"],
            dropna=False
        )["latency"]
        .mean()
        .reset_index()
    )

    print("\n=== LATENCY TIME COURSE TABLE ===")
    print(lat_timecourse_df.head())

    print("\n=== SESSION LATENCY TIME COURSE TABLE ===")
    print(session_latency_timecourse_df.head())

    return lat_timecourse_df, session_latency_timecourse_df


# ============================================================
# PLOT — LATENCY TIME COURSE AFTER SWITCH
# ============================================================

def plot_latency_timecourse(session_latency_timecourse_df, outdir, strain_colors, stage):
    df_stage = session_latency_timecourse_df[
        session_latency_timecourse_df["stage"] == stage
    ].copy()

    if df_stage.empty:
        print(f"⚠ No latency time course data for stage: {stage}")
        return

    switch_order = ["Action→Sound", "Sound→Action"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(9, 4),
        dpi=plot_config["dpi"],
        sharey=True
    )

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df_stage[df_stage["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

        # ----------------------------------------------------
        # Thick mean line + SEM shading per strain
        # ----------------------------------------------------
        for strain, strain_df in df_sw.groupby("strain"):
            stats_df = (
                strain_df.groupby("trial_from_switch")["latency"]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("trial_from_switch")
            )

            # light smoothing
            stats_df["mean_smooth"] = (
                stats_df["mean"]
                .rolling(window=3, center=True, min_periods=1)
                .mean()
            )
            stats_df["sem_smooth"] = (
                stats_df["sem"]
                .rolling(window=3, center=True, min_periods=1)
                .mean()
            )

            x = stats_df["trial_from_switch"].values
            y = stats_df["mean_smooth"].values
            sem = stats_df["sem_smooth"].fillna(0).values
            color = strain_colors.get(strain, "black")

            ax.plot(
                x,
                y,
                linewidth=2,
                color=color,
                label=strain
            )

            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=color,
                alpha=0.12
            )


        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(0, 1.0)

        ax.set_xlabel("Trial after switch", fontsize=plot_config["label_fontsize"])
        ax.set_title(switch_type, fontsize=plot_config["title_fontsize"])
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Mean lick latency (s)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D(
            [0], [0],
            color=color,
            linewidth=3,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(
        handles=handles,
        title="Strain",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=max(1, len(handles))
    )

    fig.suptitle(
        f"Latency Time Course After Switch — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"],
        y=1.08
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.18,
        top=0.82,
        wspace=0.08
    )

    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_Latency_TimeCourse_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_Latency_TimeCourse_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_Latency_TimeCourse_{stage}.pdf", bbox_inches="tight")

#%%

# ============================================================
# OMISSIONS PER BLOCK TYPE
# omission == 1 AND catch_trial == 0
# ============================================================

def compute_omissions_blocktype_table(sessions_df):
    print("\n=== Computing omissions per block type ===")

    omission_records = []
    block_order = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required_cols = {"omission", "block", "catch_trial"}
        if not required_cols.issubset(df.columns):
            print(f"⚠ Missing omission/block/catch_trial columns in {row['filename']}")
            continue

        df_valid = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].copy()

        for blk in block_order:
            blk_df = df_valid[df_valid["block"] == blk]
            n_omissions = len(blk_df)

            omission_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block_type": blk,
                "n_omissions": n_omissions,
            })

    df_omissions = pd.DataFrame(omission_records)

    if df_omissions.empty:
        print("⚠ No omission data computed.")
        return pd.DataFrame()

    print("\n=== OMISSIONS PER BLOCK TYPE ===")
    print(df_omissions.head())

    return df_omissions


# ============================================================
# TRIANGLE PLOT — OMISSIONS PER BLOCK TYPE
# ============================================================

def plot_omissions_blocktype(df_omissions, outdir, strain_colors, stage):
    df_stage = df_omissions[df_omissions["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No omission data for stage: {stage}")
        return

    rng = np.random.default_rng(42)
    block_order = ["sound", "action-right", "action-left"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    spacing = 2
    data_x = np.arange(len(block_order)) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2
    block_positions = dict(zip(block_order, data_x))

    # ----------------------------------------------------
    # Per-session points
    # ----------------------------------------------------
    for _, row in df_stage.iterrows():
        blk = row["block_type"]
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            block_positions[blk] + jitter,
            row["n_omissions"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    # ----------------------------------------------------
    # Overall mean ± SEM
    # ----------------------------------------------------
    omission_stats = (
        df_stage.groupby("block_type")["n_omissions"]
        .agg(["mean", "sem"])
        .reindex(block_order)
    )

    ax.errorbar(
        mean_x,
        omission_stats["mean"].values,
        yerr=omission_stats["sem"].values,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    # ----------------------------------------------------
    # Formatting
    # ----------------------------------------------------
    ax.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax.set_ylim(0, 200)
    ax.set_xticks(label_x)
    ax.set_xticklabels(
        ["Sound", "Action-R", "Action-L"],
        fontsize=plot_config["label_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Number of omissions",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Omissions per Block Type — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    handles.append(
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            color="black",
            markerfacecolor="black",
            label="Overall mean ± SEM"
        )
    )

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_Omissions_BlockType_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_Omissions_BlockType_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_Omissions_BlockType_{stage}.pdf", bbox_inches="tight")



#%% Percencorrect and percentcorrect per block

# ============================================================
# PERFORMANCE TABLES
# ============================================================

def compute_percentcorrect_tables(sessions_df):
    print("\n=== Computing performance tables ===")

    percentcorrect_records = []
    percentcorrect_block_records = []

    block_types = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        # ----------------------------
        # PERCENT CORRECT PER SESSION
        # ----------------------------
        needed_session = ["reward", "punishment"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()
            
            denom = n_correct + n_incorrect
            percentcorrect = (n_correct / denom * 100) if denom > 0 else np.nan

            percentcorrect_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "correct": n_correct,
                "incorrect": n_incorrect,
                "percentcorrect": percentcorrect,
            })
        else:
            print(f"⚠ Missing columns for session percentcorrect in {row['filename']}")

        # ----------------------------
        # PERFORMANCE PER BLOCK TYPE
        # ----------------------------
        needed_block = ["block", "reward", "punishment"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]

                if df_blk.empty:
                    continue

                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()

                denom = n_correct + n_incorrect
                percentcorrect_blocks = (n_correct / denom * 100) if denom > 0 else np.nan

                percentcorrect_block_records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk,
                    "percentcorrect": percentcorrect_blocks,
                })
        else:
            print(f"⚠ Missing columns for block performance in {row['filename']}")

    percentcorrect_df = pd.DataFrame(percentcorrect_records)
    percentcorrect_block_df = pd.DataFrame(percentcorrect_block_records)

    print("\n=== percentcorrect TABLE ===")
    print(percentcorrect_df.head())

    print("\n=== percentcorrect PER BLOCK TYPE ===")
    print(percentcorrect_block_df.head())

    return percentcorrect_df, percentcorrect_block_df


# ============================================================
# percentcorrect PER SESSION
# ============================================================

def plot_percentcorrect_per_session(percentcorrect_df, outdir, strain_colors, stage):
    df_stage = percentcorrect_df[percentcorrect_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No percentcorrect data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=plot_config["dpi"])

    data_x = 0.9
    mean_x = 1.1

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            data_x + jitter,
            row["percentcorrect"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    ax.errorbar(
        mean_x,
        df_stage["percentcorrect"].mean(),
        yerr=df_stage["percentcorrect"].sem(),
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    ax.axhline(50, linestyle="--", color="black", alpha=0.6)

    ax.set_xlim(0.75, 1.2)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "percentcorrect (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"percentcorrect per Session — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_PerformancePerSession_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_PerformancePerSession_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_percentcorrectPerSession_{stage}.pdf", bbox_inches="tight")


# ============================================================
# PERFORMANCE PER BLOCK TYPE
# ============================================================

def plot_percentcorrect_per_blocktype(percentcorrect_block_df, outdir, strain_colors, stage):
    df_stage = percentcorrect_block_df[percentcorrect_block_df["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No block percentcorrect data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    block_order = ["sound", "action-right", "action-left"]

    spacing = 2
    data_x = np.arange(len(block_order)) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2

    block_positions = dict(zip(block_order, data_x))

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            block_positions[row["block_type"]] + jitter,
            row["percentcorrect"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
        )

    stats = (
        df_stage.groupby("block_type")["percentcorrect"]
        .agg(["mean", "sem"])
        .reindex(block_order)
    )

    ax.errorbar(
        mean_x,
        stats["mean"].values,
        yerr=stats["sem"].values,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2,
    )

    ax.axhline(50, linestyle="--", color="gray", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax.set_xticks(label_x)
    ax.set_xticklabels(
        ["Sound", "Action-R", "Action-L"],
        fontsize=plot_config["label_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "percentcorrect (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"percentcorrect per Block Type — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"Adaptive_Performance_BlockTypes_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"Adaptive_Performance_BlockTypes_{stage}.svg", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"Adaptive_percentcorrect_BlockTypes_{stage}.pdf", bbox_inches="tight")
    
    
# ============================================================
# BLOCKS PER 100 TRIALS
# ============================================================

def compute_blocks_per_100_table(metrics_df):
    df = metrics_df.copy()

    if df.empty:
        print("⚠ metrics_df is empty")
        return pd.DataFrame()

    df["blocks_per_100_trials"] = (df["n_blocks"] / df["total_trials"]) * 100

    out = df[
        ["animal", "session_date", "strain", "stage",
         "n_blocks", "total_trials", "blocks_per_100_trials"]
    ].copy()

    print("\n=== BLOCKS PER 100 TRIALS ===")
    print(out.head())

    return out


def plot_blocks_per_100(df_blocks100, outdir, strain_colors, stage):
    df_stage = df_blocks100[df_blocks100["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No blocks/100 data for stage: {stage}")
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=plot_config["dpi"])

    data_x = 0.9
    mean_x = 1.1

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            data_x + jitter,
            row["blocks_per_100_trials"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8
        )

    ax.errorbar(
        mean_x,
        df_stage["blocks_per_100_trials"].mean(),
        yerr=df_stage["blocks_per_100_trials"].sem(),
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2
    )

    ax.set_xlim(0.75, 1.2)
    ax.set_xticks([])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(
        "Blocks per 100 trials",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Blocks per 100 Trials — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"BlocksPer100_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"BlocksPer100_{stage}.svg", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"BlocksPer100_{stage}.pdf", bbox_inches="tight")


# ============================================================
# TRIALS PER BLOCK TABLE
# ============================================================

def compute_trials_per_block_table(sessions_df):
    print("\n=== Computing trials per block table ===")

    records = []
    block_order = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        if "block" not in df.columns:
            print(f"⚠ Missing block column in {row['filename']}")
            continue

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        for _, dblk in df.groupby("block_index"):
            blk_type = dblk["block"].iloc[0]
            if blk_type not in block_order:
                continue

            records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block_type": blk_type,
                "n_trials": len(dblk),
            })

    out = pd.DataFrame(records)

    print("\n=== TRIALS PER BLOCK TABLE ===")
    print(out.head())

    return out


def plot_trials_per_block(df_trials_block, outdir, strain_colors, stage):
    df_stage = df_trials_block[df_trials_block["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No trials/block data for stage: {stage}")
        return

    rng = np.random.default_rng(42)
    block_order = ["sound", "action-right", "action-left"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=plot_config["dpi"])

    spacing = 2
    data_x = np.arange(len(block_order)) * spacing
    mean_x = data_x + 0.6
    label_x = (data_x + mean_x) / 2
    block_positions = dict(zip(block_order, data_x))

    for _, row in df_stage.iterrows():
        jitter = (rng.random() - 0.5) * 0.15
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            block_positions[row["block_type"]] + jitter,
            row["n_trials"],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8
        )

    stats = (
        df_stage.groupby("block_type")["n_trials"]
        .agg(["mean", "sem"])
        .reindex(block_order)
    )

    ax.errorbar(
        mean_x,
        stats["mean"].values,
        yerr=stats["sem"].values,
        fmt="^",
        markersize=12,
        color="black",
        capsize=4,
        linewidth=2
    )

    ax.set_xlim(-0.6, mean_x[-1] + 0.1)
    ax.set_xticks(label_x)
    ax.set_xticklabels(["Sound", "Action-R", "Action-L"], fontsize=plot_config["label_fontsize"])
    ax.set_ylabel(
        "Trials per block",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )
    ax.set_title(
        f"Trials per Block — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]
    handles.append(
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            color="black",
            markerfacecolor="black",
            label="Overall mean ± SEM"
        )
    )

    fig.legend(handles=handles, title="Strain", frameon=False)
    plt.tight_layout()
    # plt.savefig(outdir / f"TrialsPerBlock_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"TrialsPerBlock_{stage}.svg", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"TrialsPerBlock_{stage}.pdf", bbox_inches="tight")


# ============================================================
# PE FIRST VS LAST BLOCKS
# ============================================================

def compute_pe_first_last_blocks(sessions_df, window=20):
    print("\n=== Computing PE first vs last blocks ===")

    records = []
    block_order = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Error loading {row['filename']}: {e}")
            continue

        needed = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
        if not needed.issubset(df.columns):
            continue

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        block_seq = (
            df.groupby("block_index")["block"]
            .first()
            .reset_index()
            .rename(columns={"block": "block_type"})
        )

        for blk_type in block_order:
            blk_indices = block_seq.loc[block_seq["block_type"] == blk_type, "block_index"].tolist()
            if len(blk_indices) == 0:
                continue

            first_blocks = blk_indices[:2]
            last_blocks = blk_indices[-2:]

            for period_name, block_list in [("first", first_blocks), ("last", last_blocks)]:
                pe_count = 0

                for blk_idx in block_list:
                    blk_df = df[df["block_index"] == blk_idx]
                    if blk_df.empty:
                        continue

                    start_i = blk_df.index[0]
                    prev_i = start_i - 1
                    if prev_i < 0:
                        continue

                    prev_block = df.loc[prev_i, "block"]
                    new_block = df.loc[start_i, "block"]
                    next_trials = df.loc[start_i:start_i + window - 1]

                    for _, tr in next_trials.iterrows():
                        pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)
                        if not pd.isna(pe):
                            pe_count += pe

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk_type,
                    "period": period_name,
                    "pe_count": pe_count
                })

    out = pd.DataFrame(records)

    print("\n=== PE FIRST/LAST BLOCKS ===")
    print(out.head())

    return out


def plot_pe_first_last_blocks(df_pe_fl, outdir, strain_colors, stage):
    df_stage = df_pe_fl[df_pe_fl["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No PE first/last data for stage: {stage}")
        return

    block_order = ["sound", "action-right", "action-left"]
    period_order = ["first", "last"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk_type in zip(axes, block_order):
        dblk = df_stage[df_stage["block_type"] == blk_type].copy()

        for _, row in dblk.iterrows():
            x = 0 if row["period"] == "first" else 1
            jitter = (rng.random() - 0.5) * 0.12
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                x + jitter,
                row["pe_count"],
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        stats = dblk.groupby("period")["pe_count"].agg(["mean", "sem"]).reindex(period_order)

        ax.errorbar(
            [0.28, 1.28],
            stats["mean"].values,
            yerr=stats["sem"].values,
            fmt="^",
            markersize=12,
            color="black",
            capsize=4,
            linewidth=2
        )

        ax.set_xticks([0.14, 1.14])
        ax.set_xticklabels(["First", "Last"], fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk_type, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Perseverative errors",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    fig.suptitle(
        f"PE in First vs Last Blocks — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    # plt.savefig(outdir / f"PE_FirstLastBlocks_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"PE_FirstLastBlocks_{stage}.svg", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"PE_FirstLastBlocks_{stage}.pdf", bbox_inches="tight")


# ============================================================
# OMISSIONS PER STIMULUS IN ACTION BLOCKS
# ============================================================

def compute_actionblock_stim_omissions(sessions_df):
    print("\n=== Computing omissions per stimulus in action blocks ===")

    records = []
    action_blocks = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Error loading {row['filename']}: {e}")
            continue

        needed = {"block", "omission", "catch_trial", "8KHz", "16KHz"}
        if not needed.issubset(df.columns):
            continue

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        for blk in action_blocks:
            dblk = df[df["block"] == blk].copy()
            if dblk.empty:
                continue

            for stim in stim_order:
                dstim = dblk[dblk["stim"] == stim].copy()
                if dstim.empty:
                    continue

                n_trials = len(dstim)
                n_omissions = len(dstim[(dstim["omission"] == 1) & (dstim["catch_trial"] == 0)])
                omission_rate = n_omissions / n_trials if n_trials > 0 else np.nan

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "action_block": blk,
                    "stim": stim,
                    "sound_rule_side": tone_map[stim],
                    "n_trials": n_trials,
                    "n_omissions": n_omissions,
                    "omission_rate": omission_rate
                })

    out = pd.DataFrame(records)

    print("\n=== ACTION BLOCK STIM OMISSIONS ===")
    print(out.head())

    return out


def plot_actionblock_stim_omissions(df_stim_omit, outdir, strain_colors, stage):
    df_stage = df_stim_omit[df_stim_omit["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No stim omission data for stage: {stage}")
        return

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk in zip(axes, block_order):
        dblk = df_stage[df_stage["action_block"] == blk].copy()

        for _, row in dblk.iterrows():
            x = 0 if row["stim"] == "8KHz" else 1
            jitter = (rng.random() - 0.5) * 0.12
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                x + jitter,
                row["omission_rate"] * 100,
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        stats = dblk.groupby("stim")["omission_rate"].agg(["mean", "sem"]).reindex(stim_order)

        ax.errorbar(
            [0.25, 1.25],
            stats["mean"].values * 100,
            yerr=stats["sem"].fillna(0).values * 100,
            fmt="^",
            markersize=12,
            color="black",
            capsize=4,
            linewidth=2
        )

        ax.set_xticks([0.12, 1.12])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Omission rate (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    fig.suptitle(
        f"Omissions by Stimulus in Action Blocks — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    # plt.savefig(outdir / f"ActionBlockStimOmissions_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"ActionBlockStimOmissions_{stage}.svg", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"ActionBlockStimOmissions_{stage}.pdf", bbox_inches="tight")


# ============================================================
# CHOICE BIAS BY STIMULUS IN ACTION BLOCKS
# ============================================================

def compute_actionblock_choice_bias(sessions_df):
    print("\n=== Computing choice bias by stimulus in action blocks ===")

    records = []
    action_blocks = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Error loading {row['filename']}: {e}")
            continue

        needed = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
        if not needed.issubset(df.columns):
            continue

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        for blk in action_blocks:
            dblk = df[df["block"] == blk].copy()
            if dblk.empty:
                continue

            for stim in stim_order:
                dstim = dblk[(dblk["stim"] == stim) & (dblk["omission"] == 0)].copy()
                if dstim.empty:
                    continue

                if blk == "action-right":
                    n_right = (dstim["reward"] == 1).sum()
                    n_left = (dstim["punishment"] == 1).sum()
                elif blk == "action-left":
                    n_left = (dstim["reward"] == 1).sum()
                    n_right = (dstim["punishment"] == 1).sum()
                else:
                    continue

                total_choices = n_left + n_right
                p_right = n_right / total_choices if total_choices > 0 else np.nan

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "action_block": blk,
                    "stim": stim,
                    "sound_rule_side": tone_map[stim],
                    "p_right": p_right
                })

    out = pd.DataFrame(records)

    print("\n=== ACTION BLOCK CHOICE BIAS ===")
    print(out.head())

    return out


def plot_actionblock_choice_bias(df_choice_bias, outdir, strain_colors, stage):
    df_stage = df_choice_bias[df_choice_bias["stage"] == stage].copy()

    if df_stage.empty:
        print(f"⚠ No choice bias data for stage: {stage}")
        return

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk in zip(axes, block_order):
        dblk = df_stage[df_stage["action_block"] == blk].copy()

        for _, row in dblk.iterrows():
            x = 0 if row["stim"] == "8KHz" else 1
            jitter = (rng.random() - 0.5) * 0.12
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                x + jitter,
                row["p_right"] * 100,
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        stats = dblk.groupby("stim")["p_right"].agg(["mean", "sem"]).reindex(stim_order)

        ax.errorbar(
            [0.25, 1.25],
            stats["mean"].values * 100,
            yerr=stats["sem"].fillna(0).values * 100,
            fmt="^",
            markersize=12,
            color="black",
            capsize=4,
            linewidth=2
        )

        ax.axhline(50, linestyle="--", color="gray", alpha=0.5)
        ax.set_xticks([0.12, 1.12])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "P(right choice) (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="^", linestyle="",
            markersize=9,
            markerfacecolor=color,
            markeredgewidth=1.8,
            markeredgecolor=color,
            color=color,
            label=strain
        )
        for strain, color in strain_colors.items()
        if strain in df_stage["strain"].astype(str).unique()
    ]

    fig.legend(handles=handles, title="Strain", frameon=False)
    fig.suptitle(
        f"Choice Bias by Stimulus in Action Blocks — {stage.capitalize()}",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    # plt.savefig(outdir / f"ActionBlockChoiceBias_{stage}.png", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"ActionBlockChoiceBias_{stage}.svg", dpi=600, bbox_inches="tight")
    # plt.savefig(outdir / f"ActionBlockChoiceBias_{stage}.pdf", bbox_inches="tight")

#%%

# ============================================================
# MAIN
# ============================================================

def main():
    meta = pd.read_excel(SESSIONS_XLSX, dtype={"animal": str, "session_date": str})
    meta.columns = [c.strip() for c in meta.columns]

    required_cols = ["animal", "sex", "strain", "cohort", "session_date", "stage"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing required column in Excel file: {col}")

    meta["animal"] = meta["animal"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    meta["session_date"] = meta["session_date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    meta["stage"] = meta["stage"].astype(str).str.strip().str.lower()
    meta["strain"] = meta["strain"].astype(str).str.strip()
    meta["cohort"] = pd.to_numeric(meta["cohort"], errors="coerce")
    meta = meta[meta["cohort"].notna()].copy()
    meta["cohort"] = meta["cohort"].astype(int)

    meta = meta[meta["stage"].isin(["naive", "trained"])].copy()

    files_df = find_adaptive_files(meta)

    sessions_df = files_df.merge(
        meta,
        on=["animal", "cohort", "session_date"],
        how="inner"
    )

    if sessions_df.empty:
        raise ValueError("No matching sessions found between metadata Excel and files.")

    print(f"Matched {len(sessions_df)} adaptive sessions")

    metrics = []
    for _, row in sessions_df.iterrows():
        try:
            metrics.append(compute_session_metrics(row))
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        raise ValueError("No metrics could be computed.")

    metrics_df = metrics_df.sort_values(
        ["stage", "strain", "animal", "session_date"]
    ).reset_index(drop=True)

    metrics_df.to_csv(OUTDIR / "adaptive_session_metrics.csv", index=False)
    sessions_df.to_csv(OUTDIR / "adaptive_sessions_matched.csv", index=False)

    print(metrics_df.head())

    perf_df, perf_block_df = compute_performance_tables(sessions_df)
    # performance table stored in variable
    performance_table = perf_df
    perf_df.to_csv(OUTDIR / "adaptive_performance_per_session.csv", index=False)
    perf_block_df.to_csv(OUTDIR / "adaptive_performance_per_blocktype.csv", index=False)

    tri_df, session_means_df, overall_stats_df = compute_trials_until_switch_tables(sessions_df)
    tri_df.to_csv(OUTDIR / "adaptive_trials_until_switch_blocks.csv", index=False)
    session_means_df.to_csv(OUTDIR / "adaptive_trials_until_switch_session_means.csv", index=False)
    overall_stats_df.to_csv(OUTDIR / "adaptive_trials_until_switch_overall_stats.csv", index=False)

    pe_df = compute_pe_tables(sessions_df, window=20)
    pe_df.to_csv(OUTDIR / "adaptive_perseverative_errors.csv", index=False)
    pe_timecourse_df, session_timecourse_df = compute_pe_timecourse_tables(sessions_df, window=20)
    pe_timecourse_df.to_csv(OUTDIR / "adaptive_pe_timecourse_all_trials.csv", index=False)
    session_timecourse_df.to_csv(OUTDIR / "adaptive_pe_timecourse_session_means.csv", index=False)

    plot_triangle_summary(metrics_df, stage="naive", outdir=OUTDIR, strain_colors=STRAIN_COLORS)
    plot_triangle_summary(metrics_df, stage="trained", outdir=OUTDIR, strain_colors=STRAIN_COLORS)

    plot_performance_per_session(perf_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_performance_per_session(perf_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_performance_per_blocktype(perf_block_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_performance_per_blocktype(perf_block_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_trials_until_switch_blocktype(
        session_means_df, overall_stats_df, OUTDIR, STRAIN_COLORS, stage="naive"
    )
    plot_trials_until_switch_blocktype(
        session_means_df, overall_stats_df, OUTDIR, STRAIN_COLORS, stage="trained"
    )

    plot_pe_triangle(pe_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_pe_triangle(pe_df, OUTDIR, STRAIN_COLORS, stage="trained")
    
    plot_pe_timecourse(session_timecourse_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_pe_timecourse(session_timecourse_df, OUTDIR, STRAIN_COLORS, stage="trained")
    
    lat_df = compute_latency_table(sessions_df)
    lat_df.to_csv(OUTDIR / "adaptive_latency_table.csv", index=False)

    plot_latency_panels_by_stage(lat_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_latency_panels_by_stage(lat_df, OUTDIR, STRAIN_COLORS, stage="trained")
    
    lat_timecourse_df, session_latency_timecourse_df = compute_latency_timecourse_tables(
    sessions_df, window=20
    )
    
    lat_timecourse_df.to_csv(OUTDIR / "adaptive_latency_timecourse_all_trials.csv", index=False)
    session_latency_timecourse_df.to_csv(
        OUTDIR / "adaptive_latency_timecourse_session_means.csv", index=False
    )

    plot_latency_timecourse(
        session_latency_timecourse_df, OUTDIR, STRAIN_COLORS, stage="naive"
    )
    plot_latency_timecourse(
        session_latency_timecourse_df, OUTDIR, STRAIN_COLORS, stage="trained"
    )
    
    df_omissions = compute_omissions_blocktype_table(sessions_df)
    df_omissions.to_csv(OUTDIR / "adaptive_omissions_per_blocktype.csv", index=False)

    plot_omissions_blocktype(df_omissions, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_omissions_blocktype(df_omissions, OUTDIR, STRAIN_COLORS, stage="trained")
    
    percentcorrect_df, percentcorrect_block_df = compute_percentcorrect_tables(sessions_df)
    percentcorrect_table = percentcorrect_df
    
    plot_percentcorrect_per_session(percentcorrect_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_percentcorrect_per_session(percentcorrect_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_percentcorrect_per_blocktype(percentcorrect_block_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_percentcorrect_per_blocktype(percentcorrect_block_df, OUTDIR, STRAIN_COLORS, stage="trained")

    # NEW TABLES
    df_blocks100 = compute_blocks_per_100_table(metrics_df)
    df_blocks100.to_csv(OUTDIR / "blocks_per_100_trials.csv", index=False)

    df_trials_block = compute_trials_per_block_table(sessions_df)
    df_trials_block.to_csv(OUTDIR / "trials_per_block.csv", index=False)

    df_pe_fl = compute_pe_first_last_blocks(sessions_df, window=20)
    df_pe_fl.to_csv(OUTDIR / "pe_first_last_blocks.csv", index=False)

    df_stim_omit = compute_actionblock_stim_omissions(sessions_df)
    df_stim_omit.to_csv(OUTDIR / "actionblock_stim_omissions.csv", index=False)

    df_choice_bias = compute_actionblock_choice_bias(sessions_df)
    df_choice_bias.to_csv(OUTDIR / "actionblock_choice_bias.csv", index=False)

    # EXISTING PLOTS
    plot_triangle_summary(metrics_df, stage="naive", outdir=OUTDIR, strain_colors=STRAIN_COLORS)
    plot_triangle_summary(metrics_df, stage="trained", outdir=OUTDIR, strain_colors=STRAIN_COLORS)

    plot_performance_per_session(perf_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_performance_per_session(perf_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_performance_per_blocktype(perf_block_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_performance_per_blocktype(perf_block_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_trials_until_switch_blocktype(
        session_means_df, overall_stats_df, OUTDIR, STRAIN_COLORS, stage="naive"
    )
    plot_trials_until_switch_blocktype(
        session_means_df, overall_stats_df, OUTDIR, STRAIN_COLORS, stage="trained"
    )

    plot_pe_triangle(pe_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_pe_triangle(pe_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_pe_timecourse(session_timecourse_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_pe_timecourse(session_timecourse_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_latency_panels_by_stage(lat_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_latency_panels_by_stage(lat_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_latency_timecourse(
        session_latency_timecourse_df, OUTDIR, STRAIN_COLORS, stage="naive"
    )
    plot_latency_timecourse(
        session_latency_timecourse_df, OUTDIR, STRAIN_COLORS, stage="trained"
    )

    plot_omissions_blocktype(df_omissions, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_omissions_blocktype(df_omissions, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_percentcorrect_per_session(percentcorrect_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_percentcorrect_per_session(percentcorrect_df, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_percentcorrect_per_blocktype(percentcorrect_block_df, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_percentcorrect_per_blocktype(percentcorrect_block_df, OUTDIR, STRAIN_COLORS, stage="trained")

    # NEW PLOTS
    plot_blocks_per_100(df_blocks100, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_blocks_per_100(df_blocks100, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_trials_per_block(df_trials_block, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_trials_per_block(df_trials_block, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_pe_first_last_blocks(df_pe_fl, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_pe_first_last_blocks(df_pe_fl, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_actionblock_stim_omissions(df_stim_omit, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_actionblock_stim_omissions(df_stim_omit, OUTDIR, STRAIN_COLORS, stage="trained")

    plot_actionblock_choice_bias(df_choice_bias, OUTDIR, STRAIN_COLORS, stage="naive")
    plot_actionblock_choice_bias(df_choice_bias, OUTDIR, STRAIN_COLORS, stage="trained")


    print("Done.")
    return metrics_df, sessions_df, performance_table, percentcorrect_table


if __name__ == "__main__":
    metrics_df, sessions_df, performance_table, percentcorrect_table = main()
    
    