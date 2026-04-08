# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:37:36 2026

@author: JoanaCatarino
"""

# get performance

from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


# ---------- Choose Paths ----------

COHORT_DIRS = {
    1: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_1"),
    2: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_2"),
}

SESSIONS_XLSX = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/all_animals_recorded.xlsx"
)

OUTDIR = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/qc_behavior"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLORS
# ============================================================

STRAIN_COLORS = {
    "Tlx3": "#32929D",
    "Fezf2": "#B1536F",
    "Fmr1-Fezf2": "#E78848",
    "Fmr1-Tlx3": "#87B663",
}

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


# ============================================================
# QC CRITERIA
# ============================================================

def evaluate_behavior_criteria(row):
    stage = str(row["stage"]).strip().lower()
    total_trials = row["total_trials"]
    percent_omissions = row["percent_omissions"]
    percent_correct = row["percent_correct"]

    fail_reasons = []

    if pd.isna(total_trials):
        fail_reasons.append("missing total_trials")
    if pd.isna(percent_omissions):
        fail_reasons.append("missing percent_omissions")

    if stage == "trained" and pd.isna(percent_correct):
        fail_reasons.append("missing percent_correct")

    if fail_reasons:
        return pd.Series({
            "qc_pass": False,
            "qc_status": "fail",
            "qc_reason": "; ".join(fail_reasons)
        })

    if stage == "naive":
        if total_trials <= 300:
            fail_reasons.append("total_trials <= 300")
        if percent_omissions > 60:
            fail_reasons.append("percent_omissions > 60")

    elif stage == "trained":
        if total_trials <= 300:
            fail_reasons.append("total_trials <= 300")
        if percent_omissions >= 30:
            fail_reasons.append("percent_omissions >= 30")
        if percent_correct <= 60:
            fail_reasons.append("percent_correct <= 60")

    else:
        fail_reasons.append(f"unknown stage: {stage}")

    passed = len(fail_reasons) == 0

    return pd.Series({
        "qc_pass": passed,
        "qc_status": "pass" if passed else "fail",
        "qc_reason": "meets criteria" if passed else "; ".join(fail_reasons)
    })


def add_include_column(behavior_df: pd.DataFrame) -> pd.DataFrame:
    """
    include = True for:
      1) ALL Fmr1-Fezf2 sessions, regardless of QC
      2) all trained sessions that passed QC
      3) all naive sessions that passed QC AND belong to animals that also
         have at least one trained session that passed QC
    """
    df = behavior_df.copy()
    df["stage"] = df["stage"].astype(str).str.lower()
    df["strain"] = df["strain"].astype(str).str.strip()

    animals_with_trained_pass_qc = set(
        df.loc[
            (df["stage"] == "trained") & (df["qc_pass"]),
            "animal"
        ].astype(str)
    )

    include_fmr1_fezf2 = df["strain"] == "Fmr1-Fezf2"

    include_trained = (df["stage"] == "trained") & (df["qc_pass"])
    include_naive = (
        (df["stage"] == "naive")
        & (df["qc_pass"])
        & (df["animal"].astype(str).isin(animals_with_trained_pass_qc))
    )

    df["include"] = include_fmr1_fezf2 | include_trained | include_naive

    include_reason = np.where(
        include_fmr1_fezf2,
        "include_all_fmr1-fezf2",
        np.where(
            include_trained,
            "trained_pass_qc",
            np.where(
                include_naive,
                "naive_pass_qc_with_trained_pass_qc_session",
                "not_included"
            )
        )
    )
    df["include_reason"] = include_reason

    return df


# ============================================================
# PERFORMANCE TABLES
# ============================================================

def compute_performance_tables(sessions_df):
    print("\n=== Computing performance tables ===")

    performance_records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_session = ["reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()
            n_omissions = df[
                (df["omission"] == 1) & (df["catch_trial"] == 0)
            ].shape[0]

            denom = n_correct + n_incorrect + n_omissions
            denom_2 = n_correct + n_incorrect

            performance = (n_correct / denom * 100) if denom > 0 else np.nan
            percentcorrect = (n_correct / denom_2 * 100) if denom_2 > 0 else np.nan
            percentomissions = (n_omissions / denom * 100) if denom > 0 else np.nan
            total_trials = len(df)

            performance_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "total_trials": total_trials,
                "correct": n_correct,
                "incorrect": n_incorrect,
                "omissions": n_omissions,
                "performance": performance,
                "percent_correct": percentcorrect,
                "percent_omissions": percentomissions
            })
        else:
            print(f"⚠ Missing columns for session performance in {row['filename']}")

    behavior_df = pd.DataFrame(performance_records)

    if behavior_df.empty:
        return behavior_df

    qc_df = behavior_df.apply(evaluate_behavior_criteria, axis=1)
    behavior_df = pd.concat([behavior_df, qc_df], axis=1)
    behavior_df = add_include_column(behavior_df)

    print(behavior_df.head())
    return behavior_df


# ============================================================
# PLOT 1: QC PANELS
# all sessions


def plot_qc_criteria_panels(behavior_df, outdir: Path):
    if behavior_df.empty:
        print("⚠ No data available for QC plot.")
        return None, None

    stage_order = ["naive", "trained"]
    stage_titles = {"naive": "Pre-Adaptive", "trained": "Post-Adaptive"}

    metric_specs = [
        {"col": "total_trials", "ylabel": "Total trials", "title": "Total trials", "ylims": None},
        {"col": "percent_omissions", "ylabel": "Percent omissions (%)", "title": "% Omissions", "ylims": (0, 100)},
        {"col": "percent_correct", "ylabel": "Percent correct (%)", "title": "% Correct", "ylims": (0, 100)},
    ]

    qc_lines = {
        "naive": {"total_trials": 300, "percent_omissions": 60, "percent_correct": None},
        "trained": {"total_trials": 300, "percent_omissions": 30, "percent_correct": 60},
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    rng = np.random.default_rng(42)
    
    # ---------- GLOBAL FONT SIZES ----------
    TITLE_SIZE = 16
    LABEL_SIZE = 13
    TICK_SIZE = 8
    LEGEND_SIZE = 12
    SUBTITLE_SIZE = 11

    for row_idx, stage in enumerate(stage_order):
        df_stage = behavior_df[behavior_df["stage"].astype(str).str.lower() == stage].copy()

        if df_stage.empty:
            for col_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue

        df_stage = df_stage.sort_values(["strain", "animal", "session_date"]).reset_index(drop=True)
        x = np.arange(len(df_stage))

        for col_idx, spec in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            y = df_stage[spec["col"]].values.astype(float)

            cutoff = qc_lines[stage][spec["col"]]

            if cutoff is not None:
                if spec["ylims"] is not None:
                    ymin, ymax = spec["ylims"]
                else:
                    ymin = min(np.nanmin(y), cutoff) * 0.9
                    ymax = max(np.nanmax(y), cutoff) * 1.1

                if spec["col"] == "total_trials":
                    ax.axhspan(cutoff, ymax, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(ymin, cutoff, color="#f8d7da", alpha=0.5, zorder=0)
                elif spec["col"] == "percent_omissions":
                    ax.axhspan(ymin, cutoff, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(cutoff, ymax, color="#f8d7da", alpha=0.5, zorder=0)
                elif spec["col"] == "percent_correct":
                    ax.axhspan(cutoff, ymax, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(ymin, cutoff, color="#f8d7da", alpha=0.5, zorder=0)

                ax.axhline(cutoff, color="black", linestyle="--", linewidth=1.5)

            # ---------- SCATTER ----------
            for i, row in df_stage.iterrows():
                jitter = rng.uniform(-0.15, 0.15)
                color = STRAIN_COLORS.get(row["strain"], "gray")

                ax.scatter(
                    i + jitter,
                    row[spec["col"]],
                    s=90,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.95
                )

            # ---------- LABELS ----------
            ax.set_title(f"{stage_titles[stage]} — {spec['title']}", fontsize=13)
            ax.set_ylabel(spec["ylabel"], fontsize=LABEL_SIZE)
            ax.set_xlabel("Sessions", fontsize=LABEL_SIZE)

            ax.set_xlim(-0.8, len(df_stage) - 0.2)

            if spec["ylims"] is not None:
                ax.set_ylim(spec["ylims"])

            ax.set_xticks(x)
            ax.set_xticklabels(df_stage["animal"].astype(str), rotation=90, fontsize=TICK_SIZE)

            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Naive percent correct note
            if stage == "naive" and spec["col"] == "percent_correct":
                ax.text(
                    0.98, 0.95,
                    "No QC cutoff",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=SUBTITLE_SIZE,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8)
                )

    # ---------- LEGEND ----------
    strain_handles = [
        plt.Line2D([0], [0],
                   marker="o",
                   linestyle="",
                   markerfacecolor=color,
                   markeredgecolor="black",
                   markersize=8,
                   label=strain)
        for strain, color in STRAIN_COLORS.items()
        if strain in behavior_df["strain"].astype(str).unique()
    ]

    fig.legend(
        handles=strain_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=LEGEND_SIZE
    )

    fig.suptitle("Behavior QC metrics by session", fontsize=TITLE_SIZE, y=1.05)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.55)

    fig.savefig(outdir / "behavior_qc_criteria_panels.png", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_criteria_panels.pdf", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_criteria_panels.svg", dpi=500, bbox_inches="tight")

    return fig, axes



# ============================================================
# 3D QC PLOT
# ============================================================


def plot_qc_3d_panels(behavior_df, outdir: Path):
    if behavior_df.empty:
        print("⚠ No data available for 3D QC plot.")
        return None, None

    stage_order = ["naive", "trained"]
    stage_titles = {"naive": "Pre-Adaptive", "trained": "Post-Adaptive"}

    fig = plt.figure(figsize=(14, 6), dpi=500)
    axes = []

    for i, stage in enumerate(stage_order, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        axes.append(ax)

        df_stage = behavior_df[behavior_df["stage"].astype(str).str.lower() == stage].copy()

        if df_stage.empty:
            ax.set_title(f"{stage_titles[stage]} — no data")
            continue

        x = df_stage["total_trials"].astype(float).values
        y = df_stage["percent_omissions"].astype(float).values
        z = df_stage["percent_correct"].astype(float).values

        # ----------------------------
        # Scatter points
        # ----------------------------
        for _, row in df_stage.iterrows():
            color = STRAIN_COLORS.get(row["strain"], "gray")
            marker = "o" if row["qc_pass"] else "X"

            ax.scatter(
                row["total_trials"],
                row["percent_omissions"],
                row["percent_correct"],
                s=80,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95
            )

        # ----------------------------
        # Axis labels
        # ----------------------------
        ax.set_xlabel("Total trials", labelpad=10)
        ax.set_ylabel("Percent omissions", labelpad=10)
        ax.set_zlabel("Percent correct", labelpad=10)
        ax.set_title(stage_titles[stage], fontsize=14)

        # ----------------------------
        # Axis limits
        # ----------------------------
        x_min = min(np.nanmin(x), 250)
        x_max = max(np.nanmax(x), 850)

        y_min, y_max = 0, 100
        z_min, z_max = 0, 100

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # ----------------------------
        # QC cutoff planes
        # ----------------------------
        yy, zz = np.meshgrid(
            np.linspace(y_min, y_max, 10),
            np.linspace(z_min, z_max, 10)
        )

        xx, zz2 = np.meshgrid(
            np.linspace(x_min, x_max, 10),
            np.linspace(z_min, z_max, 10)
        )

        xx2, yy2 = np.meshgrid(
            np.linspace(x_min, x_max, 10),
            np.linspace(y_min, y_max, 10)
        )

        # total_trials cutoff plane
        trials_cutoff = 300
        X_plane = np.full_like(yy, trials_cutoff)
        ax.plot_surface(
            X_plane, yy, zz,
            color=None,
            alpha=0.18,
            linewidth=0,
            shade=False
        )

        if stage == "naive":
            omissions_cutoff = 60

            # omissions cutoff plane
            Y_plane = np.full_like(xx, omissions_cutoff)
            ax.plot_surface(
                xx, Y_plane, zz2,
                color="#f4a6a6",
                alpha=0.16,
                linewidth=0,
                shade=False
            )

            ax.text(
                trials_cutoff, y_max, z_max,
                "trials = 300",
                fontsize=9
            )
            ax.text(
                x_max, omissions_cutoff, z_max,
                "omissions = 60",
                fontsize=9
            )

        elif stage == "trained":
            omissions_cutoff = 30
            correct_cutoff = 60

            # omissions cutoff plane
            Y_plane = np.full_like(xx, omissions_cutoff)
            ax.plot_surface(
                xx, Y_plane, zz2,
                color="#f4a6a6",
                alpha=0.16,
                linewidth=0,
                shade=False
            )

            # percent_correct cutoff plane
            Z_plane = np.full_like(xx2, correct_cutoff)
            ax.plot_surface(
                xx2, yy2, Z_plane,
                color="#a8ddb5",
                alpha=0.16,
                linewidth=0,
                shade=False
            )

            ax.text(
                trials_cutoff, y_max, z_max,
                "trials = 300",
                fontsize=9
            )
            ax.text(
                x_max, omissions_cutoff, z_max,
                "omissions = 30",
                fontsize=9
            )
            ax.text(
                x_max, y_max, correct_cutoff,
                "correct = 60",
                fontsize=9
            )

        # ----------------------------
        # View angle
        # ----------------------------
        ax.view_init(elev=22, azim=-55)

    # ----------------------------
    # Legend for strains
    # ----------------------------
    strain_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=strain
        )
        for strain, color in STRAIN_COLORS.items()
        if strain in behavior_df["strain"].astype(str).unique()
    ]

    # pass/fail legend
    pass_handle = plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label="Pass QC"
    )
    fail_handle = plt.Line2D(
        [0], [0],
        marker="X",
        linestyle="",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label="Fail QC"
    )

    fig.legend(
        handles=strain_handles + [pass_handle, fail_handle],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10
    )

    fig.suptitle("3D behavior QC space", fontsize=16, y=1.06)
    plt.tight_layout()

    fig.savefig(outdir / "behavior_qc_3d_panels.png", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_3d_panels.pdf", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_3d_panels.svg", dpi=500, bbox_inches="tight")


    return fig, axes


# ============================================================
# PLOT 3: QC PANELS
# included = full color
# excluded = same color, slightly transparent
# ============================================================

def plot_included_panels(behavior_df, outdir: Path):
    if behavior_df.empty:
        print("⚠ No data available for QC plot.")
        return None, None

    stage_order = ["naive", "trained"]
    stage_titles = {"naive": "Pre-Adaptive", "trained": "Post-Adaptive"}

    metric_specs = [
        {"col": "total_trials", "ylabel": "Total trials", "title": "Total trials", "ylims": None},
        {"col": "percent_omissions", "ylabel": "Percent omissions (%)", "title": "% Omissions", "ylims": (0, 100)},
        {"col": "percent_correct", "ylabel": "Percent correct (%)", "title": "% Correct", "ylims": (0, 100)},
    ]

    qc_lines = {
        "naive": {"total_trials": 300, "percent_omissions": 60, "percent_correct": None},
        "trained": {"total_trials": 300, "percent_omissions": 30, "percent_correct": 60},
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=500)
    rng = np.random.default_rng(42)
    
    # ---------- GLOBAL FONT SIZES ----------
    TITLE_SIZE = 16
    LABEL_SIZE = 13
    TICK_SIZE = 8
    LEGEND_SIZE = 12
    SUBTITLE_SIZE = 11

    for row_idx, stage in enumerate(stage_order):
        df_stage = behavior_df[behavior_df["stage"].astype(str).str.lower() == stage].copy()

        if df_stage.empty:
            for col_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue

        df_stage = df_stage.sort_values(["strain", "animal", "session_date"]).reset_index(drop=True)
        x = np.arange(len(df_stage))

        for col_idx, spec in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            y = df_stage[spec["col"]].values.astype(float)

            cutoff = qc_lines[stage][spec["col"]]

            if cutoff is not None:
                if spec["ylims"] is not None:
                    ymin, ymax = spec["ylims"]
                else:
                    ymin = min(np.nanmin(y), cutoff) * 0.9
                    ymax = max(np.nanmax(y), cutoff) * 1.1

                if spec["col"] == "total_trials":
                    ax.axhspan(cutoff, ymax, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(ymin, cutoff, color="#f8d7da", alpha=0.5, zorder=0)
                elif spec["col"] == "percent_omissions":
                    ax.axhspan(ymin, cutoff, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(cutoff, ymax, color="#f8d7da", alpha=0.5, zorder=0)
                elif spec["col"] == "percent_correct":
                    ax.axhspan(cutoff, ymax, color="#d8f5d0", alpha=0.5, zorder=0)
                    ax.axhspan(ymin, cutoff, color="#f8d7da", alpha=0.5, zorder=0)

                ax.axhline(cutoff, color="black", linestyle="--", linewidth=1.5)

            for i, row in df_stage.iterrows():
                jitter = rng.uniform(-0.15, 0.15)
                color = STRAIN_COLORS.get(row["strain"], "gray")
                alpha = 0.95 if bool(row.get("include", False)) else 0.35

                ax.scatter(
                    i + jitter,
                    row[spec["col"]],
                    s=90,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=alpha
                )

            ax.set_title(f"{stage_titles[stage]} — {spec['title']}", fontsize=13)
            ax.set_ylabel(spec["ylabel"], fontsize=LABEL_SIZE)
            ax.set_xlabel("Sessions", fontsize=LABEL_SIZE)
            ax.set_xlim(-0.8, len(df_stage) - 0.2)

            if spec["ylims"] is not None:
                ax.set_ylim(spec["ylims"])

            ax.set_xticks(x)
            ax.set_xticklabels(df_stage["animal"].astype(str), rotation=90, fontsize=TICK_SIZE)

            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if stage == "naive" and spec["col"] == "percent_correct":
                ax.text(
                    0.98, 0.95,
                    "No QC cutoff",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=SUBTITLE_SIZE,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8)
                )

    # ---------- LEGEND ----------
    strain_handles = [
        plt.Line2D([0], [0],
                   marker="o",
                   linestyle="",
                   markerfacecolor=color,
                   markeredgecolor="black",
                   markersize=8,
                   label=strain)
        for strain, color in STRAIN_COLORS.items()
        if strain in behavior_df["strain"].astype(str).unique()
    ]

    fig.legend(
        handles=strain_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=LEGEND_SIZE
    )

    fig.suptitle("Behavior QC metrics by session", fontsize=TITLE_SIZE, y=1.05)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.55)


    fig.savefig(outdir / "behavior_qc_included_panels.png", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_included_panels.pdf", dpi=500, bbox_inches="tight")
    fig.savefig(outdir / "behavior_qc_included_panels.svg", dpi=500, bbox_inches="tight")

    return fig, axes



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

    behavior_df = compute_performance_tables(sessions_df)

    if behavior_df.empty:
        raise ValueError("No behavior QC metrics could be computed.")

    performance_table = behavior_df
    behavior_df.to_csv(OUTDIR / "behavior_qc_metrics.csv", index=False)

    passed_df = behavior_df[behavior_df["qc_pass"]].copy()
    passed_df.to_csv(OUTDIR / "behavior_qc_metrics_pass_only.csv", index=False)

    included_df = behavior_df[behavior_df["include"]].copy()
    included_df.to_csv(OUTDIR / "behavior_qc_metrics_included.csv", index=False)

    print("\n=== QC summary by stage ===")
    print(
        behavior_df.groupby(["stage", "qc_status"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n=== Include summary by stage ===")
    print(
        behavior_df.groupby(["stage", "include"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n=== Include summary by strain ===")
    print(
        behavior_df.groupby(["strain", "include"])
        .size()
        .unstack(fill_value=0)
    )

    plot_qc_criteria_panels(behavior_df, OUTDIR)
    plot_qc_3d_panels(behavior_df, OUTDIR)
    plot_included_panels(behavior_df, OUTDIR)

    plt.show()

    print("\nDone.")
    return metrics_df, sessions_df, performance_table

# ============================================================
# EXTRA VERSION:
# same plots, excluding Fmr1-Fezf2, and saving plots only
# ============================================================

def main_exclude_fmr1_fezf2_plots_only():
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

    print(f"Matched {len(sessions_df)} adaptive sessions for extra no-Fmr1-Fezf2 plots")

    behavior_df = compute_performance_tables(sessions_df)

    if behavior_df.empty:
        raise ValueError("No behavior QC metrics could be computed for extra no-Fmr1-Fezf2 plots.")

    # Exclude Fmr1-Fezf2 only for this extra plotting version
    behavior_df_no_fmr1_fezf2 = behavior_df[
        behavior_df["strain"].astype(str).str.strip() != "Fmr1-Fezf2"
    ].copy()

    if behavior_df_no_fmr1_fezf2.empty:
        raise ValueError("No sessions left after excluding Fmr1-Fezf2.")

    # Save extra plots with different filenames so the original ones are kept
    fig1, axes1 = plot_qc_criteria_panels(behavior_df_no_fmr1_fezf2, OUTDIR)
    if fig1 is not None:
        fig1.savefig(OUTDIR / "behavior_qc_criteria_panels_no_fmr1_fezf2.png", dpi=500, bbox_inches="tight")
        fig1.savefig(OUTDIR / "behavior_qc_criteria_panels_no_fmr1_fezf2.pdf", dpi=500, bbox_inches="tight")
        fig1.savefig(OUTDIR / "behavior_qc_criteria_panels_no_fmr1_fezf2.svg", dpi=500, bbox_inches="tight")

    fig2, axes2 = plot_qc_3d_panels(behavior_df_no_fmr1_fezf2, OUTDIR)
    if fig2 is not None:
        fig2.savefig(OUTDIR / "behavior_qc_3d_panels_no_fmr1_fezf2.png", dpi=500, bbox_inches="tight")
        fig2.savefig(OUTDIR / "behavior_qc_3d_panels_no_fmr1_fezf2.pdf", dpi=500, bbox_inches="tight")
        fig2.savefig(OUTDIR / "behavior_qc_3d_panels_no_fmr1_fezf2.svg", dpi=500, bbox_inches="tight")

    fig3, axes3 = plot_included_panels(behavior_df_no_fmr1_fezf2, OUTDIR)
    if fig3 is not None:
        fig3.savefig(OUTDIR / "behavior_qc_included_panels_no_fmr1_fezf2.png", dpi=500, bbox_inches="tight")
        fig3.savefig(OUTDIR / "behavior_qc_included_panels_no_fmr1_fezf2.pdf", dpi=500, bbox_inches="tight")
        fig3.savefig(OUTDIR / "behavior_qc_included_panels_no_fmr1_fezf2.svg", dpi=500, bbox_inches="tight")

    print("\nDone extra plots without Fmr1-Fezf2.")
    return behavior_df_no_fmr1_fezf2

if __name__ == "__main__":
    metrics_df, sessions_df, performance_table = main()
    behavior_df_no_fmr1_fezf2 = main_exclude_fmr1_fezf2_plots_only()
