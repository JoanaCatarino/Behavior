# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:43:49 2025

@author: JoanaCatarino
"""
import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================
#               USER SETTINGS
# ============================================

base_dir = r"L:/dmclab/Joana/Behavior/Data"

animals_of_interest = ["986235", "999770", "986167", "986168", "986170", "986171"]

animal_strain = {
    "986235": "Tlx3",
    "999770": "Tlx3",
    "986167": "Fezf2",
    "986168": "Fezf2",
    "986170": "Fezf2",
    "986171": "Fezf2",
}

# Sessions to analyze (YYYYMMDD)
adaptive_sessions_to_analyze = {
    "986235": ["20251202"],
    "999770": ["20251111"],
    "986167": ["20251204"],
    "986168": ["20251203"],
    "986170": ["20251125", "20251201"],
}

# Build list of ID strings: e.g. "986235_20251202"
selected_session_ids = []
for animal, dates in adaptive_sessions_to_analyze.items():
    for d in dates:
        selected_session_ids.append(f"{animal}_{d}")

# Choose colors PER SESSION (editable)
session_colors = {
    "20251202": "#44B8CC",   # blue
    "20251111": "#FCC26A",   # yellow
    "20251204": "#6EC76E",   # green
    "20251203": "#FD926B",   # orange
    "20251125": "#7945B5",   # purple
    "20251201": "#C671C3",   # pink
}

OUT = r"L:/dmclab/Joana/Behavior/Data/plots/behavior_cohort_1/adaptive_task"


# =====================================================
#              HELPERS
# =====================================================

def detect_task_from_filename(filename: str) -> str:
    if filename.startswith("FreeLick"):
        return "free_licking"
    elif filename.startswith("2ChoiceAuditory"):
        return "two_choice"
    elif filename.startswith("2ChoiceBlocks"):
        return "two_choice_blocks"
    elif filename.startswith("AdaptSensorimotor"):
        return "adaptive_sensorimotor"
    return "unknown"


def extract_datetime_from_filename(filename: str):
    m = re.search(r"_(\d{8})_(\d{6})_", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def find_files_folder_structure(base_dir: str, animals: list[str]) -> pd.DataFrame:
    """Scans folder tree and finds all CSVs."""
    records = []

    for animal in animals:
        beh_dir = Path(base_dir) / animal / "Behavior"
        if not beh_dir.exists():
            print(f"⚠ No folder for {animal}")
            continue

        for day in sorted([d for d in beh_dir.iterdir() if d.is_dir()]):
            csvs = list(day.glob("*.csv"))
            if not csvs:
                continue

            csv_path = csvs[0]
            filename = csv_path.name

            records.append({
                "animal": animal,
                "task": detect_task_from_filename(filename),
                "dt": extract_datetime_from_filename(filename),
                "path": str(csv_path)
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["animal", "dt"]).reset_index(drop=True)
    return df



# =====================================================
#              LOAD ALL FILES
# =====================================================

all_files = find_files_folder_structure(base_dir, animals_of_interest)
all_files["day_in_task"] = all_files.groupby(["animal", "task"]).cumcount() + 1
all_files["strain"] = all_files["animal"].map(animal_strain)

print("\n=== ALL FILES ===")
print(all_files.head())



# =====================================================
#              EXTRACT SELECTED ADAPTIVE SESSIONS
# =====================================================

session_records = []

for animal, date_list in adaptive_sessions_to_analyze.items():
    for date_str in date_list:

        mask = (
            (all_files["animal"] == animal) &
            (all_files["task"] == "adaptive_sensorimotor") &
            (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"⚠ Missing session {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"])

        # ---- Trials ----
        total_trials = len(df)

        # ---- Count blocks (changes in df["block"]) ----
        block_counts = {"sound": 0, "action-right": 0, "action-left": 0}
        prev = None
        for b in df["block"]:
            if b != prev:
                if b in block_counts:
                    block_counts[b] += 1
                prev = b

        n_blocks = sum(block_counts.values())

        session_records.append({
            "animal": animal,
            "strain": row["strain"],
            "session": date_str,
            "total_trials": total_trials,
            "n_blocks": n_blocks,
            "n_sound_blocks": block_counts["sound"],
            "n_aright_blocks": block_counts["action-right"],
            "n_aleft_blocks": block_counts["action-left"]
        })

df_sessions = pd.DataFrame(session_records)

print("\n=== ADAPTIVE SESSIONS SUMMARY ===")
print(df_sessions)



# =====================================================
#      LONG-FORM BLOCK TYPE TABLE
# =====================================================

blocktype_records = []
for _, r in df_sessions.iterrows():
    blocktype_records += [
        {"animal": r["animal"], "session": r["session"],
         "strain": r["strain"], "block": "sound", "count": r["n_sound_blocks"]},
        {"animal": r["animal"], "session": r["session"],
         "strain": r["strain"], "block": "action-right", "count": r["n_aright_blocks"]},
        {"animal": r["animal"], "session": r["session"],
         "strain": r["strain"], "block": "action-left", "count": r["n_aleft_blocks"]},
    ]

df_blocktypes = pd.DataFrame(blocktype_records)

print("\n=== BLOCK TYPES LONG ===")
print(df_blocktypes.head())



# =====================================================
#        3-PANEL TRIANGLE SUMMARY PLOT
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=500)

# ----------------------------------------------------
# PANEL 1 – TOTAL TRIALS
# ----------------------------------------------------
ax1 = axes[0]

for _, tr in df_sessions.iterrows():
    j = (np.random.rand() - 0.5) * 0.20
    ax1.scatter(
        1 + j, tr["total_trials"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=session_colors[tr["session"]],
        linewidths=1.8
    )

mean_trials = df_sessions["total_trials"].mean()
sem_trials = df_sessions["total_trials"].sem()

ax1.errorbar(
    1.15, mean_trials, yerr=sem_trials,
    fmt="^", markersize=12, color="black", capsize=4
)

ax1.set_xlim(0.7, 1.5)
ax1.set_ylim(500, 800)
ax1.set_xticks([])
ax1.set_ylabel("Trials performed")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# ----------------------------------------------------
# PANEL 2 – TOTAL BLOCKS
# ----------------------------------------------------
ax2 = axes[1]

for _, tr in df_sessions.iterrows():
    j = (np.random.rand() - 0.5) * 0.20
    ax2.scatter(
        1 + j, tr["n_blocks"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=session_colors[tr["session"]],
        linewidths=1.8
    )

mean_blocks = df_sessions["n_blocks"].mean()
sem_blocks = df_sessions["n_blocks"].sem()

ax2.errorbar(
    1.15, mean_blocks, yerr=sem_blocks,
    fmt="^", markersize=12, color="black", capsize=4
)

ax2.set_xlim(0.7, 1.5)
ax2.set_ylim(0, 16)
ax2.set_xticks([])
ax2.set_ylabel("Number of blocks")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# ----------------------------------------------------
# PANEL 3 – BLOCKS BY TYPE
# ----------------------------------------------------
ax3 = axes[2]

block_order = ["sound", "action-right", "action-left"]
x_pos = np.arange(3)
block_positions = dict(zip(block_order, x_pos))

# session dots
for _, tr in df_blocktypes.iterrows():
    j = (np.random.rand() - 0.5) * 0.20
    ax3.scatter(
        block_positions[tr["block"]] + j,
        tr["count"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=session_colors[tr["session"]],
        linewidths=1.8
    )

# mean ± SEM
stats = df_blocktypes.groupby("block")["count"].agg(["mean", "sem"]).reindex(block_order)

ax3.errorbar(
    x_pos + 0.10,
    stats["mean"].values,
    yerr=stats["sem"].values,
    fmt="^",
    markersize=12,
    color="black",
    capsize=4
)

ax3.set_ylim(0, 10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(["Sound", "Action-R", "Action-L"])
ax3.set_ylabel("Blocks per type")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Legend
handles = [
    plt.Line2D([0], [0], marker="^", linestyle="", markersize=9,
               color=c, label=sess)
    for sess, c in session_colors.items()
]

fig.legend(handles=handles, title="Sessions",
           loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_TriangleSummary.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_TriangleSummary.svg", dpi=600)

#%%

# =====================================================
#            PREPARE LATENCY COMPUTATION
# =====================================================

adaptive_files = all_files[all_files["task"] == "adaptive_sensorimotor"].copy()

print("\n=== Adaptive Files ===")
print(adaptive_files)



# =====================================================
#       COMPUTE LICK LATENCY PER BLOCK TYPE
# =====================================================

latency_records = []

for _, row in adaptive_files.iterrows():

    filename = Path(row["path"]).name
    date_str = filename.split("_")[2]  # YYYYMMDD
    session_id = f"{row['animal']}_{date_str}"

    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(row["path"])

    if not all(col in df.columns for col in ["block", "lick_time", "RW_start"]):
        print(f"⚠ Missing columns in {row['path']}")
        continue

    # Compute latency based on respose window start
    df["latency"] = df["lick_time"] - df["RW_start"]
    

    for blk in ["sound", "action-right", "action-left"]:
        blk_df = df[df["block"] == blk]

        if blk_df.empty:
            continue

        latency_records.append({
            "animal": row["animal"],
            "session_id": session_id,
            "session": date_str,
            "block_type": blk,
            "mean_latency": blk_df["latency"].mean()
        })

latency_df = pd.DataFrame(latency_records)

print("\n=== LATENCY DATA ===")
print(latency_df)



# =====================================================
#                 LATENCY PLOT
# =====================================================

block_order = ["sound", "action-right", "action-left"]

latency_df["block_type"] = pd.Categorical(
    latency_df["block_type"],
    categories=block_order,
    ordered=True
)

latency_df = latency_df.sort_values(["session_id", "block_type"])

plt.figure(figsize=(8, 5), dpi=500)

for sess in selected_session_ids:
    df_s = latency_df[latency_df["session_id"] == sess]
    if df_s.empty:
        continue

    plt.plot(
        df_s["block_type"],
        df_s["mean_latency"],
        marker="o",
        linewidth=2.2,
        markersize=8,
        color=session_colors[df_s["session"].iloc[0]],
        alpha=0.95,
        label=sess
    )


plt.ylabel("Mean Lick Latency (s)", fontsize=11)
plt.xticks(block_order, ["Sound", "Action-R", "Action-L"], fontsize=11)
plt.ylim(0,1)
plt.title("Mean Lick Latency per Block Type")
sns.despine()

plt.legend(title="Session", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_LickLatency_SelectedSessions.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_LickLatency_SelectedSessions.svg", dpi=600)

