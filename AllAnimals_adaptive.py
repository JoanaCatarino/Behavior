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
    "999770": ["20251111", "20251206"],
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
    "20251206": "red"
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
ax2.set_ylim(0, 26)
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

ax3.set_ylim(0, 15)
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

#%% ============================================================
# 6. LICK LATENCY — FIRST vs LAST block of each block type
# ============================================================

latFL_records = []

valid_blocks = ["sound", "action-right", "action-left"]

for _, row in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    # Extract session identifier "animal_YYYYMMDD"
    filename = Path(row["path"]).name
    date_str = filename.split("_")[2]
    session_id = f"{row['animal']}_{date_str}"

    # Only analyze selected sessions
    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(row["path"]).copy()

    # Must contain required columns
    required_cols = ["block", "lick_time", "trial_start"]
    if not all(c in df.columns for c in required_cols):
        print(f"⚠ Missing required columns in {row['path']}")
        continue

    # Compute latency
    df["latency"] = df["lick_time"] - (df["trial_start"] + 1.4)

    # Identify block transitions
    df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)

    # Assign a block index per block identity
    df["block_index"] = df["block_change"].cumsum()

    # For each block type
    for blk in valid_blocks:

        df_blk = df[df["block"] == blk]
        if df_blk.empty:
            continue

        block_numbers = df_blk["block_index"].unique()
        if len(block_numbers) == 0:
            continue

        first_block = block_numbers.min()
        last_block = block_numbers.max()

        # First block latency
        first_lat = df_blk[df_blk["block_index"] == first_block]["latency"].mean()

        # Last block latency
        last_lat = df_blk[df_blk["block_index"] == last_block]["latency"].mean()

        latFL_records.append({
            "animal": row["animal"],
            "session": date_str,
            "session_id": session_id,
            "block_type": blk,
            "first_latency": first_lat,
            "last_latency": last_lat
        })

latFL_df = pd.DataFrame(latFL_records)
print("\n=== FIRST vs LAST BLOCK LATENCIES ===")
print(latFL_df)

# ============================================================
# 7. Plot FIRST vs LAST latency per block type (each session)
# ============================================================

plt.figure(figsize=(8, 5), dpi=500)

block_order = ["sound", "action-right", "action-left"]

for session_id in selected_session_ids:

    df_sess = latFL_df[latFL_df["session_id"] == session_id]
    if df_sess.empty:
        continue

    df_sess = df_sess.set_index("block_type").reindex(block_order)

    # X positions
    x = np.arange(len(block_order))

    plt.plot(
        x,
        df_sess["first_latency"],
        marker="o",
        color=session_colors[df_sess["session"].iloc[0]],
        linewidth=2,
        label=f"{session_id} (first)",
        alpha=0.8
    )

    plt.plot(
        x,
        df_sess["last_latency"],
        marker="s",
        linestyle="--",
        color=session_colors[df_sess["session"].iloc[0]],
        linewidth=2,
        label=f"{session_id} (last)",
        alpha=0.8
    )

# Formatting
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xticks(x, ["Sound", "Action-R", "Action-L"])
plt.ylabel("Lick latency (s)")
plt.title("First vs Last Block Latency per Session")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_LickLatency_First_vs_Last.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_LickLatency_First_vs_Last.svg", dpi=600)

#%%# ============================================================
# 8. TRIANGLE PLOT: First vs Last latency PER BLOCK TYPE
# ============================================================

plt.figure(figsize=(10, 5), dpi=500)

block_order = ["sound", "action-right", "action-left"]
n_blocks = len(block_order)

# X positions:
# sound → positions [0,1]
# action-R → positions [2,3]
# action-L → positions [4,5]
x_positions = []
labels = []
for i, blk in enumerate(block_order):
    x_positions.extend([2*i, 2*i + 1])
    labels.extend([f"{blk}\nfirst", f"{blk}\nlast"])

# ---------------------------
# Plot triangles
# ---------------------------
for _, row in latFL_df.iterrows():

    session = row["session"]
    sess_color = session_colors[session]

    blk = row["block_type"]
    blk_idx = block_order.index(blk)

    # First block triangle x-position
    x_first = 2 * blk_idx

    # Last block triangle x-position
    x_last = 2 * blk_idx + 1

    # Add jitter so sessions don’t overlap
    jitter_first = (np.random.rand() - 0.5) * 0.15
    jitter_last  = (np.random.rand() - 0.5) * 0.15

    # FIRST block triangle
    plt.scatter(
        x_first + jitter_first,
        row["first_latency"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=sess_color,
        linewidths=2,
        zorder=3,
        label=row["session_id"] if blk_idx == 0 else None  # only label once
    )

    # LAST block triangle
    plt.scatter(
        x_last + jitter_last,
        row["last_latency"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=sess_color,
        linewidths=2,
        zorder=3
    )

# ---------------------------
# Formatting
# ---------------------------

plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xticks(x_positions, labels, rotation=0, fontsize=11)
plt.ylabel("Lick Latency (s)")
plt.title("First vs Last Block Lick Latency — Adaptive Sensorimotor", fontsize=13)

plt.grid(axis="y", linestyle="--", alpha=0.3)

# Legend
legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="^",
        linestyle="",
        markersize=9,
        color=color,
        label=session
    )
    for session, color in session_colors.items()
]

plt.legend(
    handles=legend_handles,
    title="Sessions",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_LickLatency_Triangle_FirstLast.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_LickLatency_Triangle_FirstLast.svg", dpi=600)

#%%
# ============================================================
# 8. DOT + LINE PLOT: First vs Last latency PER BLOCK TYPE
# ============================================================

plt.figure(figsize=(10, 5), dpi=500)

block_order = ["sound", "action-right", "action-left"]
n_blocks = len(block_order)

# X positions:
# sound → 0,1
# action-right → 2,3
# action-left → 4,5
x_positions = []
labels = []
for i, blk in enumerate(block_order):
    x_positions.extend([2*i, 2*i + 1])
    labels.extend([f"{blk}\nfirst", f"{blk}\nlast"])

# ---------------------------
# Plot dots + connecting lines
# ---------------------------
for session_id in latFL_df["session_id"].unique():

    df_sess = latFL_df[latFL_df["session_id"] == session_id]
    sess_color = session_colors[df_sess["session"].iloc[0]]

    for _, row in df_sess.iterrows():

        blk = row["block_type"]
        blk_idx = block_order.index(blk)

        # First and last x positions
        x_first = 2 * blk_idx
        x_last  = 2 * blk_idx + 1

        # Add jitter so dots do not sit on top of each other
        jitter_first = (np.random.rand() - 0.5) * 0.15
        jitter_last  = (np.random.rand() - 0.5) * 0.15

        # FIRST dot
        x1 = x_first + jitter_first
        y1 = row["first_latency"]

        # LAST dot
        x2 = x_last + jitter_last
        y2 = row["last_latency"]

        # Connect FIRST → LAST
        plt.plot(
            [x1, x2], [y1, y2],
            color=sess_color,
            alpha=0.7,
            linewidth=2
        )

        # Plot dots
        plt.scatter(
            x1, y1,
            color=sess_color,
            edgecolor="white",
            s=70,
            zorder=3
        )
        plt.scatter(
            x2, y2,
            color=sess_color,
            edgecolor="white",
            s=70,
            zorder=3
        )


# ---------------------------
# Formatting
# ---------------------------

# Horizontal reference line
plt.axhline(0, linestyle="--", color="gray", linewidth=1)

# Vertical separators between block types
for i in range(1, n_blocks):
    plt.axvline(2*i - 0.5, color="lightgray", linewidth=1, alpha=0.6)

plt.xticks(x_positions, labels, fontsize=10)
plt.ylabel("Lick Latency (s)")
plt.title("First vs Last Block Latency", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.3)

# Legend: one entry per session
legend_handles = [
    plt.Line2D([0], [0], color=c, marker="o", linestyle="-", label=sess)
    for sess, c in session_colors.items()
]

plt.legend(
    handles=legend_handles,
    title="Sessions",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_LickLatency_FirstLast_DotLine.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_LickLatency_FirstLast_DotLine.svg", dpi=600)

#%% ============================================================
# 9. Hit & Perseverative Error around Block Switches
#    (animal 999770, sessions 20251111 & 20251206)
# ============================================================

from pathlib import Path  # already imported above, but safe if re-run in cell mode

# ----- SETTINGS FOR THIS ANALYSIS -----
animal_pe = "999770"
sessions_pe = ["20251111", "20251206"]   # YYYYMMDD for this animal

# Where the tone–spout mapping lives
mapping_file_path = Path(
    r"L:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv"
)

# Load tone–spout mapping for this animal
spout_map_df = pd.read_csv(mapping_file_path)
map_row = spout_map_df[spout_map_df["Animal"] == int(animal_pe)].iloc[0]

tone_map = {
    "8KHz": map_row["8KHz"],   # "left" or "right"
    "16KHz": map_row["16KHz"]
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def correct_side_for_block(block_type: str, stim: str, tone_map: dict) -> str:
    """
    Returns the correct spout ('left' or 'right') for a given
    block type and stimulus.
    """
    if block_type == "sound":
        # Sound block: stimulus is mapped to its original spout
        return tone_map[stim]

    elif block_type == "action-left":
        # Action-left block: correct = left regardless of tone
        return "left"

    elif block_type == "action-right":
        # Action-right block: correct = right regardless of tone
        return "right"

    # Fallback (should not happen if your block labels are clean)
    return tone_map[stim]


def classify_trial_pe(row: pd.Series, block_type: str, tone_map: dict):
    """
    Classify one trial as:
        - HIT (1 or 0)
        - Perseverative error (1 or 0)

    Omissions are excluded → (nan, nan).

    Perseverative error:
      incorrect trial where the animal licked the *habitual* side
      (tone's original side) while the correct side for that block
      is the *opposite* one.
    """
    # Exclude omissions
    if row["omission"] == 1:
        return np.nan, np.nan

    # Determine which stimulus was played
    if row["8KHz"] == 1:
        stim = "8KHz"
    elif row["16KHz"] == 1:
        stim = "16KHz"
    else:
        # No tone flagged → skip
        return np.nan, np.nan

    correct_side = correct_side_for_block(block_type, stim, tone_map)
    habitual_side = tone_map[stim]  # learned side from mapping

    # HIT
    if row["reward"] == 1:
        return 1.0, 0.0

    # INCORRECT
    if row["punishment"] == 1:
        # perseverative if they choose habitual side while correct side differs
        # We don't have lick-left/lick-right columns, but we know the trial
        # was *not* rewarded and that the block has a defined correct side.
        # So we treat "perseverative" as: habitual_side != correct_side
        if habitual_side != correct_side:
            return 0.0, 1.0
        else:
            return 0.0, 0.0

    # Anything else → ignore
    return np.nan, np.nan


def extract_switch_windows(df: pd.DataFrame, window: int = 20):
    """
    Find block switches and build windows of trials around each switch.

    Returns:
        windows: dict with
            windows["A→S"] = list of DataFrames
            windows["S→A"] = list of DataFrames

        Each DataFrame has:
            - all rows from last `window` trials of previous block
              and first `window` trials of next block
            - 'trial_rel' column:
                  negative = trials before switch (-N..-1)
                  positive = trials after switch (1..+N)
                  0 is reserved for the switch itself, not used.
    """
    windows = {"A→S": [], "S→A": []}

    # Find indices where block changes
    for i in range(1, len(df)):
        prev_block = df.loc[i - 1, "block"]
        next_block = df.loc[i, "block"]

        if prev_block == next_block:
            continue

        # Classify switch direction
        if prev_block in ["action-left", "action-right"] and next_block == "sound":
            direction = "A→S"
        elif prev_block == "sound" and next_block in ["action-left", "action-right"]:
            direction = "S→A"
        else:
            # Ignore other transitions
            continue

        # Compute window boundaries
        start_prev = max(0, i - window)
        end_next = min(len(df), i + window)

        prev_df = df.iloc[start_prev:i].copy()
        next_df = df.iloc[i:end_next].copy()

        # Assign trial_rel: previous block = negative indices, next block = positive
        n_prev = len(prev_df)
        n_next = len(next_df)

        prev_df["trial_rel"] = np.arange(-n_prev, 0)  # e.g. -20..-1
        next_df["trial_rel"] = np.arange(1, n_next + 1)

        combined = pd.concat([prev_df, next_df], ignore_index=True)

        windows[direction].append(combined)

    return windows


def process_pe_session(sess_date: str, window: int = 20):
    """
    For a given session (YYYYMMDD) of animal `animal_pe`, compute:

      - Hit fraction per trial_rel
      - Perseverative error fraction per trial_rel

    separately for "A→S" and "S→A" transitions.

    Returns:
      {
        "A→S": DataFrame(trial_rel, hit_mean, hit_sem, pe_mean, pe_sem),
        "S→A": DataFrame(...)
      }
      or None if no matching file.
    """
    # Use the already-loaded all_files to find the correct file
    mask = (
        (all_files["animal"] == animal_pe)
        & (all_files["task"] == "adaptive_sensorimotor")
        & (all_files["dt"].dt.strftime("%Y%m%d") == sess_date)
    )

    if not mask.any():
        print(f"❌ No adaptive file found in all_files for {animal_pe} / {sess_date}")
        return None

    row = all_files[mask].iloc[0]
    df = pd.read_csv(row["path"]).copy()

    # Basic sanity check
    required_cols = ["block", "8KHz", "16KHz", "reward", "punishment", "omission"]
    if not all(col in df.columns for col in required_cols):
        print(f"⚠ Missing required columns in {row['path']}")
        return None

    windows = extract_switch_windows(df, window=window)

    results = {}
    trial_range = [t for t in range(-window, window + 1) if t != 0]

    for direction in ["A→S", "S→A"]:
        hit_dict = {t: [] for t in trial_range}
        pe_dict = {t: [] for t in trial_range}

        for comb in windows[direction]:
            for _, tr in comb.iterrows():
                t = int(tr["trial_rel"])
                if t not in trial_range:
                    continue

                blk_type = tr["block"]
                hit, pe = classify_trial_pe(tr, blk_type, tone_map)

                if not np.isnan(hit):
                    hit_dict[t].append(hit)
                if not np.isnan(pe):
                    pe_dict[t].append(pe)

        # Aggregate across all switches
        hit_mean = []
        hit_sem = []
        pe_mean = []
        pe_sem = []

        for t in trial_range:
            hvals = hit_dict[t]
            pvals = pe_dict[t]

            if len(hvals) > 0:
                hit_mean.append(np.nanmean(hvals))
                hit_sem.append(
                    np.nanstd(hvals, ddof=1) / np.sqrt(len(hvals))
                    if len(hvals) > 1 else np.nan
                )
            else:
                hit_mean.append(np.nan)
                hit_sem.append(np.nan)

            if len(pvals) > 0:
                pe_mean.append(np.nanmean(pvals))
                pe_sem.append(
                    np.nanstd(pvals, ddof=1) / np.sqrt(len(pvals))
                    if len(pvals) > 1 else np.nan
                )
            else:
                pe_mean.append(np.nan)
                pe_sem.append(np.nan)

        out_df = pd.DataFrame({
            "trial_rel": trial_range,
            "hit_mean": hit_mean,
            "hit_sem": hit_sem,
            "pe_mean": pe_mean,
            "pe_sem": pe_sem,
        })

        results[direction] = out_df

    return results


# ------------------------------------------------------------
# Plot: one figure per session, 2 panels (A→S and S→A)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Plot: one figure per session, 2 panels (A→S and S→A)
# Hit = blue, Perseverative = red
# SEM = shaded region, no connecting lines
# ------------------------------------------------------------

hit_color = "#1f77b4"     # blue
pe_color  = "#d62728"     # red

for sess in sessions_pe:
    print(f"\n=== Processing block switches for session {sess} ===")
    res = process_pe_session(sess, window=20)
    if res is None:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=500)

    # ============================================================
    # PANEL 1 — Action → Sound
    # ============================================================
    ax = axes[0]
    df_as = res["A→S"]

    # HIT mean ± SEM (shaded)
    ax.fill_between(
        df_as["trial_rel"],
        df_as["hit_mean"] - df_as["hit_sem"],
        df_as["hit_mean"] + df_as["hit_sem"],
        color=hit_color, alpha=0.25
    )
    ax.scatter(df_as["trial_rel"], df_as["hit_mean"],
               color=hit_color, s=20, label="Hit")

    # PE mean ± SEM (shaded)
    ax.fill_between(
        df_as["trial_rel"],
        df_as["pe_mean"] - df_as["pe_sem"],
        df_as["pe_mean"] + df_as["pe_sem"],
        color=pe_color, alpha=0.25
    )
    ax.scatter(df_as["trial_rel"], df_as["pe_mean"],
               color=pe_color, s=20, label="Persev. error")

    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 21, 5))
    ax.set_xlabel("Trial from switch", fontsize=11)
    ax.set_ylabel("Fraction of trials", fontsize=11)
    ax.set_title(f"{animal_pe} {sess}   Action → Sound")
    ax.legend(frameon=False)

    # ============================================================
    # PANEL 2 — Sound → Action
    # ============================================================
    ax = axes[1]
    df_sa = res["S→A"]

    # HIT shading
    ax.fill_between(
        df_sa["trial_rel"],
        df_sa["hit_mean"] - df_sa["hit_sem"],
        df_sa["hit_mean"] + df_sa["hit_sem"],
        color=hit_color, alpha=0.25
    )
    ax.scatter(df_sa["trial_rel"], df_sa["hit_mean"],
               color=hit_color, s=20, label="Hit")

    # PE shading
    ax.fill_between(
        df_sa["trial_rel"],
        df_sa["pe_mean"] - df_sa["pe_sem"],
        df_sa["pe_mean"] + df_sa["pe_sem"],
        color=pe_color, alpha=0.25
    )
    ax.scatter(df_sa["trial_rel"], df_sa["pe_mean"],
               color=pe_color, s=20, label="Persev. error")

    ax.axvline(0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-20, 20)
    ax.set_xticks(np.arange(-20, 21, 5))
    ax.set_xlabel("Trial from switch", fontsize=11)
    ax.set_ylabel("Fraction of trials", fontsize=11)
    ax.set_title(f"{animal_pe} {sess}   Sound → Action")
    ax.legend(frameon=False)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"{OUT}/Adaptive_PER_shaded_{animal_pe}_{sess}.png", dpi=600)
    plt.savefig(f"{OUT}/Adaptive_PER_shaded_{animal_pe}_{sess}.svg", dpi=600)
   

    
    