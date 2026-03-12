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
    df["latency"] = df["lick_time"] - (df["RW_start"])

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


def classify_trial_pe(row, block_type, tone_map, prev_block):
    """
    Returns hit, perseverative_error

    RULES:
    -------
    ACTION blocks:
        PE = following habitual mapping instead of the rule-side

    SOUND blocks:
        PE = following the rule of the previous ACTION block:
             prev_block = 'action-left'  → pe if lick == left
             prev_block = 'action-right' → pe if lick == right
    """

    # Ignore omissions
    if row["omission"] == 1:
        return np.nan, np.nan

    # Determine stimulus
    stim = "8KHz" if row["8KHz"] == 1 else "16KHz"

    # Habitual mapping
    habitual = tone_map[stim]

    # Determine lick side
    lick_side = "right" if row.get("lick_right", 0) == 1 else "left"

    # --------------------
    # HIT
    # --------------------
    if row["reward"] == 1:
        return 1.0, 0.0

    # If not punished, ignore
    if row["punishment"] != 1:
        return np.nan, np.nan

    # ----------------------------
    # ACTION blocks (rule-based)
    # ----------------------------
    if block_type == "action-left":
        correct = "left"
        pe = 1.0 if habitual != correct else 0.0
        return 0.0, pe

    if block_type == "action-right":
        correct = "right"
        pe = 1.0 if habitual != correct else 0.0
        return 0.0, pe

    # ----------------------------
    # SOUND block (depends on PREVIOUS block)
    # ----------------------------
    if block_type == "sound":

        if prev_block == "action-left":
            pe = 1.0 if lick_side == "left" else 0.0

        elif prev_block == "action-right":
            pe = 1.0 if lick_side == "right" else 0.0

        else:
            pe = 0.0

        return 0.0, pe

    return np.nan, np.nan


# ============================================================
# Extract windows around block switches
# ============================================================

def extract_switch_windows(df, window=20):
    """
    Returns dict:
        windows["A→S"] = list of DataFrames with:
            - trial_rel (-20..-1, 1..20)
            - prev_block column
        windows["S→A"] = same
    """

    windows = {"A→S": [], "S→A": []}

    for i in range(1, len(df)):
        prev_b = df.loc[i - 1, "block"]
        next_b = df.loc[i, "block"]

        if prev_b == next_b:
            continue

        if prev_b in ["action-left", "action-right"] and next_b == "sound":
            direction = "A→S"

        elif prev_b == "sound" and next_b in ["action-left", "action-right"]:
            direction = "S→A"

        else:
            continue

        # Window boundaries
        start_prev = max(0, i - window)
        end_next = min(len(df), i + window)

        prev_df = df.iloc[start_prev:i].copy()
        next_df = df.iloc[i:end_next].copy()

        # Indexing
        prev_df["trial_rel"] = np.arange(-len(prev_df), 0)
        next_df["trial_rel"] = np.arange(1, len(next_df) + 1)

        # Store previous block identity for the SOUND-block PE rule
        prev_df["prev_block"] = prev_b
        next_df["prev_block"] = prev_b

        combined = pd.concat([prev_df, next_df], ignore_index=True)
        windows[direction].append(combined)

    return windows


# ============================================================
# Process a single session (using all_files)
# ============================================================

def process_pe_session(sess_string, window=20):
    """Returns dict with A→S and S→A mean/SEM curves."""

    mask = (
        (all_files["animal"] == animal_pe)
        & (all_files["task"] == "adaptive_sensorimotor")
        & (all_files["dt"].dt.strftime("%Y%m%d") == sess_string)
    )

    if not mask.any():
        print(f"❌ No file found in all_files for {animal_pe} / {sess_string}")
        return None

    row = all_files[mask].iloc[0]
    df = pd.read_csv(row["path"]).copy()

    # required columns check
    required_cols = ["block", "8KHz", "16KHz", "reward", "punishment", "omission"]
    if not all(c in df.columns for c in required_cols):
        print(f"⚠ Missing columns in {row['path']}")
        return None

    wins = extract_switch_windows(df, window=window)
    trial_range = [t for t in range(-window, window + 1) if t != 0]

    results = {}

    for direction in ["A→S", "S→A"]:
        hit_dict = {t: [] for t in trial_range}
        pe_dict = {t: [] for t in trial_range}

        for block_df in wins[direction]:

            for _, r in block_df.iterrows():
                t = int(r["trial_rel"])
                if t == 0 or t not in trial_range:
                    continue

                hit, pe = classify_trial_pe(
                    r,
                    r["block"],
                    tone_map,
                    r["prev_block"]
                )

                if not np.isnan(hit):
                    hit_dict[t].append(hit)
                if not np.isnan(pe):
                    pe_dict[t].append(pe)

        # convert dict → dataframe
        out = {
            "trial_rel": trial_range,
            "hit_mean": [np.nanmean(hit_dict[t]) if hit_dict[t] else np.nan for t in trial_range],
            "hit_sem": [
                np.nanstd(hit_dict[t], ddof=1) / np.sqrt(len(hit_dict[t])) if len(hit_dict[t]) > 1 else np.nan
                for t in trial_range
            ],
            "pe_mean": [np.nanmean(pe_dict[t]) if pe_dict[t] else np.nan for t in trial_range],
            "pe_sem": [
                np.nanstd(pe_dict[t], ddof=1) / np.sqrt(len(pe_dict[t])) if len(pe_dict[t]) > 1 else np.nan
                for t in trial_range
            ]
        }

        results[direction] = pd.DataFrame(out)

    return results


# ============================================================
#  Plotting — HIT (blue) / PE (red), SEM shading, no lines
# ============================================================

hit_color = "#5B9E9A"   # blue
pe_color = "#815D9F"    # purple

for sess in sessions_pe:
    print(f"\n=== Processing session {sess} ===")

    res = process_pe_session(sess, window=20)
    if res is None:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=500)

    # =============================
    # PANEL 1 — S → A
    # =============================
    df_sa = res["S→A"]
    ax = axes[1]
    
    # HIT
    ax.fill_between(df_sa["trial_rel"],
                    df_sa["hit_mean"] - df_sa["hit_sem"],
                    df_sa["hit_mean"] + df_sa["hit_sem"],
                    color=hit_color, alpha=0.25)
    ax.scatter(df_sa["trial_rel"], df_sa["hit_mean"], color=hit_color, s=20)
    
    # PE
    ax.fill_between(df_sa["trial_rel"],
                    df_sa["pe_mean"] - df_sa["pe_sem"],
                    df_sa["pe_mean"] + df_sa["pe_sem"],
                    color=pe_color, alpha=0.25)
    ax.scatter(df_sa["trial_rel"], df_sa["pe_mean"], color=pe_color, s=20)
    
    ax.axvline(0, color="red", linestyle="--")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-20, 20)
    ax.set_title(f"{animal_pe} {sess}   Sound → Action")
    ax.set_xlabel("Trial from switch")
    ax.set_ylabel("Fraction")
    ax.set_xticks(np.arange(-20, 21, 5))

    # =============================
    # PANEL 2 — A → S
    # =============================
    df_as = res["A→S"]
    ax = axes[0]

    # HIT
    ax.fill_between(df_as["trial_rel"],
                    df_as["hit_mean"] - df_as["hit_sem"],
                    df_as["hit_mean"] + df_as["hit_sem"],
                    color=hit_color, alpha=0.25)
    ax.scatter(df_as["trial_rel"], df_as["hit_mean"], color=hit_color, s=20)

    # PE
    ax.fill_between(df_as["trial_rel"],
                    df_as["pe_mean"] - df_as["pe_sem"],
                    df_as["pe_mean"] + df_as["pe_sem"],
                    color=pe_color, alpha=0.25)
    ax.scatter(df_as["trial_rel"], df_as["pe_mean"], color=pe_color, s=20)

    ax.axvline(0, color="red", linestyle="--")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-20, 20)
    ax.set_title(f"{animal_pe} {sess}   Action → Sound")
    ax.set_xlabel("Trial from switch")
    ax.set_ylabel("Fraction")
    ax.set_xticks(np.arange(-20, 21, 5))

    

    sns.despine()
    plt.tight_layout()

    plt.savefig(f"{OUT}/Adaptive_PE_hit_shaded_{animal_pe}_{sess}.png", dpi=600)
    plt.savefig(f"{OUT}/Adaptive_PE_hit_shaded_{animal_pe}_{sess}.svg", dpi=600)  

    
#%% =====================================================================
#  GLOBAL PERSEVERATIVE / HIT PLOT ACROSS ALL SESSIONS
# =====================================================================

print("\n=== Running GLOBAL PE/HIT analysis for ALL sessions ===")

# Global accumulators
global_results = {
    "A→S": {t: [] for t in range(-20, 21) if t != 0},
    "S→A": {t: [] for t in range(-20, 21) if t != 0}
}

global_results_pe = {
    "A→S": {t: [] for t in range(-20, 21) if t != 0},
    "S→A": {t: [] for t in range(-20, 21) if t != 0}
}

# Counters for trial totals
trial_count = {
    "A→S": {"sound": 0, "action": 0},
    "S→A": {"sound": 0, "action": 0}
}

# -------------------------------------------------------------
# PROCESS ALL ADAPTIVE SESSIONS
# -------------------------------------------------------------

adaptive_all = all_files[all_files["task"] == "adaptive_sensorimotor"]

for _, row in adaptive_all.iterrows():

    df = pd.read_csv(row["path"]).copy()

    if not {"block", "8KHz", "16KHz", "reward", "punishment", "omission"} <= set(df.columns):
        continue

    wins = extract_switch_windows(df, window=20)

    for direction in ["A→S", "S→A"]:

        for block_df in wins[direction]:

            for _, tr in block_df.iterrows():

                t = int(tr["trial_rel"])
                if t == 0 or t < -20 or t > 20:
                    continue

                hit, pe = classify_trial_pe(
                    tr,
                    tr["block"],
                    tone_map,
                    tr["prev_block"]
                )

                if not np.isnan(hit):
                    global_results[direction][t].append(hit)
                if not np.isnan(pe):
                    global_results_pe[direction][t].append(pe)

                # Count trials by block type
                if tr["block"] == "sound":
                    trial_count[direction]["sound"] += 1
                else:
                    trial_count[direction]["action"] += 1


# =====================================================================
#     Prepare GLOBAL MEAN + SEM tables
# =====================================================================

def build_df(direction, hit_dict, pe_dict):
    rows = []
    for t in sorted(hit_dict.keys()):
        hvals = hit_dict[t]
        pvals = pe_dict[t]

        rows.append({
            "trial_rel": t,
            "hit_mean": np.nanmean(hvals) if len(hvals) else np.nan,
            "hit_sem":  np.nanstd(hvals, ddof=1)/np.sqrt(len(hvals)) if len(hvals) > 1 else np.nan,
            "pe_mean":  np.nanmean(pvals) if len(pvals) else np.nan,
            "pe_sem":   np.nanstd(pvals, ddof=1)/np.sqrt(len(pvals)) if len(pvals) > 1 else np.nan
        })

    return pd.DataFrame(rows)

df_global_AS = build_df("A→S", global_results["A→S"], global_results_pe["A→S"])
df_global_SA = build_df("S→A", global_results["S→A"], global_results_pe["S→A"])


# =====================================================================
#                        PRINT TRIAL COUNTS
# =====================================================================

print("\n=== TOTAL TRIALS USED ===")
for direction in ["A→S", "S→A"]:
    print(f"\n{direction}:")
    print(f"   Sound-block trials:  {trial_count[direction]['sound']}")
    print(f"   Action-block trials: {trial_count[direction]['action']}")


# =====================================================================
#                      GLOBAL COMBINED PLOT
# =====================================================================

hit_color = "#5B9E9A"   # blue
pe_color = "#815D9F"    # purple

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=500)

# -----------------------------------
# PANEL 1 — A→S
# -----------------------------------
ax = axes[0]
df_ = df_global_AS

ax.fill_between(df_["trial_rel"], df_["hit_mean"]-df_["hit_sem"],
                df_["hit_mean"]+df_["hit_sem"], color=hit_color, alpha=0.25)
ax.scatter(df_["trial_rel"], df_["hit_mean"], color=hit_color, s=20, label="Hit")

ax.fill_between(df_["trial_rel"], df_["pe_mean"]-df_["pe_sem"],
                df_["pe_mean"]+df_["pe_sem"], color=pe_color, alpha=0.25)
ax.scatter(df_["trial_rel"], df_["pe_mean"], color=pe_color, s=20, label="PE")

ax.axvline(0, color="gray", linestyle="--")
ax.set_xlim(-20, 20)
ax.set_ylim(0, 1.1)
ax.set_title("GLOBAL  —  Action → Sound", fontsize=12)
ax.set_xlabel("Trial from switch", fontsize=11)
ax.set_ylabel("Fraction of trials", fontsize=11)
ax.legend(frameon=False)


# -----------------------------------
# PANEL 2 — S→A
# -----------------------------------
ax = axes[1]
df_ = df_global_SA

ax.fill_between(df_["trial_rel"], df_["hit_mean"]-df_["hit_sem"],
                df_["hit_mean"]+df_["hit_sem"], color=hit_color, alpha=0.25)
ax.scatter(df_["trial_rel"], df_["hit_mean"], color=hit_color, s=20, label="Hit")

ax.fill_between(df_["trial_rel"], df_["pe_mean"]-df_["pe_sem"],
                df_["pe_mean"]+df_["pe_sem"], color=pe_color, alpha=0.25)
ax.scatter(df_["trial_rel"], df_["pe_mean"], color=pe_color, s=20, label="PE")

ax.axvline(0, color="gray", linestyle="--")
ax.set_xlim(-20, 20)
ax.set_ylim(0, 1.1)
ax.set_title("GLOBAL  —  Sound → Action", fontsize=12)
ax.set_xlabel("Trial from switch", fontsize=11)
ax.set_ylabel("Fraction of trials", fontsize=11)
ax.legend(frameon=False)

sns.despine()
plt.tight_layout()

plt.savefig(f"{OUT}/Adaptive_Global_PE_Hit.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_Global_PE_Hit.svg", dpi=600)

print("\n=== Global plot saved ===")

#%%

# Block types in the order you want them plotted
block_order = ["sound", "action-right", "action-left"]

# ------------------------------------------------------------
# Step 1 — Extract "number of trials until switch" per block
# ------------------------------------------------------------

records = []   # one entry per block within each session

for _, row in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    # Extract session ID: "animal_YYYYMMDD"
    filename = Path(row["path"]).name
    date_str = filename.split("_")[2]
    session_id = f"{row['animal']}_{date_str}"

    # Skip if not part of selected adaptive session pool
    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(row["path"]).copy()

    # Determine indices where block changes occur
    df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
    df["block_index"] = df["block_change"].cumsum()

    # For each block index, compute total trials inside that block
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
            "session": date_str,
            "session_id": session_id,
            "block_type": blk_type,
            "n_trials": n_trials
        })

# Convert to DataFrame
tri_df = pd.DataFrame(records)


# ------------------------------------------------------------
# Step 2 — Compute per-session means
# ------------------------------------------------------------

session_means = (
    tri_df.groupby(["session", "block_type"])["n_trials"]
    .mean()
    .reset_index()
)

# Compute overall grand mean ± SEM per block type
overall_stats = (
    tri_df.groupby("block_type")["n_trials"]
    .agg(["mean", "sem"])
    .reindex(block_order)
)

# Map x positions
x_positions = np.arange(len(block_order))


# ------------------------------------------------------------
# Step 3 — PLOT
# ------------------------------------------------------------

plt.figure(figsize=(9, 6), dpi=500)

# --------------------------
#  Plot session-specific means (open colored triangles)
# --------------------------
for _, row in session_means.iterrows():
    blk = row["block_type"]
    sess = row["session"]
    x = block_order.index(blk)
    color = session_colors[sess]

    # small jitter in x
    jitter = (np.random.rand() - 0.5) * 0.15

    plt.scatter(
        x + jitter, row["n_trials"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=color,
        linewidths=2,
        label=sess if f"label_{sess}" not in locals() else ""
    )
    globals()[f"label_{sess}"] = True  # ensures label added only once

# Horizontal line at 20 trials
plt.axhline(20, color="black", linestyle="--", linewidth=1.5)

# --------------------------
#  Plot overall mean ± SEM (solid black triangles)
# --------------------------
plt.errorbar(
    x_positions,
    overall_stats["mean"].values,
    yerr=overall_stats["sem"].values,
    fmt="^",
    markersize=12,
    color="black",
    capsize=5,
    linewidth=2,
    label="Overall mean ± SEM"
)


# --------------------------
# Formatting
# --------------------------
plt.ylim(0, 350)
plt.xticks(x_positions, ["Sound", "Action-R", "Action-L"], fontsize=12)
plt.ylabel("Trials until switch", fontsize=13)
plt.title("Trials Until Block Switch — Triangle Plot", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()

# Legend (one entry per session)
handles = [
    plt.Line2D([0], [0], marker="^", linestyle="", 
               color=c, markerfacecolor="none", markersize=10, label=sess)
    for sess, c in session_colors.items() if sess in session_means["session"].unique()
]
handles.append(
    plt.Line2D([0], [0], marker="^", color="black", markersize=10, label="Overall mean ± SEM")
)

plt.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/Triangle_TrialsUntilSwitch_AllSessions.png", dpi=600)
plt.savefig(f"{OUT}/Triangle_TrialsUntilSwitch_AllSessions.svg", dpi=600)

#%%

# Block types in the order you want them plotted
block_order = ["sound", "action-right", "action-left"]

# ------------------------------------------------------------
# Step 1 — Extract "number of trials until switch" per block
# ------------------------------------------------------------

records = []   # one entry per block within each session

for _, row in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    # Extract session ID: "animal_YYYYMMDD"
    filename = Path(row["path"]).name
    date_str = filename.split("_")[2]
    session_id = f"{row['animal']}_{date_str}"

    # Skip if not part of selected adaptive session pool
    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(row["path"]).copy()

    # Determine indices where block changes occur
    df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
    df["block_index"] = df["block_change"].cumsum()

    # For each block index, compute total trials inside that block
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
            "session": date_str,
            "session_id": session_id,
            "block_type": blk_type,
            "n_trials": n_trials
        })

# Convert to DataFrame
tri_df = pd.DataFrame(records)


# ------------------------------------------------------------
# Step 2 — Compute per-session means
# ------------------------------------------------------------

session_means = (
    tri_df.groupby(["session", "block_type"])["n_trials"]
    .mean()
    .reset_index()
)

# Compute overall grand mean ± SEM per block type
overall_stats = (
    tri_df.groupby("block_type")["n_trials"]
    .agg(["mean", "sem"])
    .reindex(block_order)
)

# Map x positions
x_positions = np.arange(len(block_order))


# ------------------------------------------------------------
# Step 3 — PLOT
# ------------------------------------------------------------

plt.figure(figsize=(9, 6), dpi=500)

# --------------------------
#  Plot session-specific means (open colored triangles)
# --------------------------
for _, row in session_means.iterrows():
    blk = row["block_type"]
    sess = row["session"]
    x = block_order.index(blk)
    color = session_colors[sess]

    # small jitter in x
    jitter = (np.random.rand() - 0.5) * 0.15

    plt.scatter(
        x + jitter, row["n_trials"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=color,
        linewidths=2,
        label=sess if f"label_{sess}" not in locals() else ""
    )
    globals()[f"label_{sess}"] = True  # ensures label added only once

# Horizontal line at 20 trials
plt.axhline(20, color="black", linestyle="--", linewidth=1.5)

# --------------------------
#  Plot overall mean ± SEM (solid black triangles)
# --------------------------
plt.errorbar(
    x_positions,
    overall_stats["mean"].values,
    yerr=overall_stats["sem"].values,
    fmt="^",
    markersize=12,
    color="black",
    capsize=5,
    linewidth=2,
    label="Overall mean ± SEM"
)


# --------------------------
# Formatting
# --------------------------
plt.ylim(0, 100)
plt.xticks(x_positions, ["Sound", "Action-R", "Action-L"], fontsize=12)
plt.ylabel("Trials until switch", fontsize=13)
plt.title("Trials Until Block Switch — Triangle Plot", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()

# Legend (one entry per session)
handles = [
    plt.Line2D([0], [0], marker="^", linestyle="", 
               color=c, markerfacecolor="none", markersize=10, label=sess)
    for sess, c in session_colors.items() if sess in session_means["session"].unique()
]
handles.append(
    plt.Line2D([0], [0], marker="^", color="black", markersize=10, label="Overall mean ± SEM")
)

plt.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/Triangle_TrialsUntilSwitch_AllSessions_zoom.png", dpi=600)
plt.savefig(f"{OUT}/Triangle_TrialsUntilSwitch_AllSessions_zoom.svg", dpi=600)

#%%
# Omissions
# ------------------------------------------------------------
# TRIANGLE PLOT — OMISSIONS PER BLOCK TYPE
# Only omissions where: omission == 1  AND  catch_trial == 0
# ------------------------------------------------------------

print("\n=== Computing omissions (omission==1 & catch_trial==0) per block type ===")

omission_records = []

# Loop through all adaptive sessions
for _, row in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    filename = Path(row["path"]).name
    date_str = filename.split("_")[2]
    session_id = f"{row['animal']}_{date_str}"

    # Only include selected sessions
    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(row["path"])

    # Check required columns
    if not {"omission", "block", "catch_trial"}.issubset(df.columns):
        print(f"⚠ Missing omission or catch_trial columns in {row['path']}")
        continue

    # Filter to only omission trials that were NOT catch trials
    df_valid = df[(df["omission"] == 1) & (df["catch_trial"] == 0)]

    for blk in ["sound", "action-right", "action-left"]:
        blk_df = df_valid[df_valid["block"] == blk]

        # Count valid omissions
        n_omissions = len(blk_df)

        omission_records.append({
            "animal": row["animal"],
            "session": date_str,
            "session_id": session_id,
            "block_type": blk,
            "n_omissions": n_omissions
        })

df_omissions = pd.DataFrame(omission_records)

print("\n=== FILTERED OMISSION COUNTS (omission==1 & catch_trial==0) ===")
print(df_omissions)


# ------------------------------------------------------------
# Stats for overall mean ± SEM
# ------------------------------------------------------------
block_order = ["sound", "action-right", "action-left"]

omission_stats = (
    df_omissions.groupby("block_type")["n_omissions"]
    .agg(["mean", "sem"])
    .reindex(block_order)
)

x_positions = np.arange(len(block_order))


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

plt.figure(figsize=(9, 6), dpi=500)

# Open triangles = per-session omissions
for _, row in df_omissions.iterrows():
    blk = row["block_type"]
    sess = row["session"]

    x = block_order.index(blk)
    color = session_colors[sess]

    # jitter to avoid overlap
    jitter = (np.random.rand() - 0.5) * 0.15

    plt.scatter(
        x + jitter,
        row["n_omissions"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=color,
        linewidths=2,
        label=sess if f"_lab_{sess}" not in globals() else None
    )
    globals()[f"_lab_{sess}"] = True


# Solid triangles = overall mean ± SEM
plt.errorbar(
    x_positions,
    omission_stats["mean"].values,
    yerr=omission_stats["sem"].values,
    fmt="^",
    markersize=12,
    color="black",
    capsize=5,
    linewidth=2,
    label="Overall mean ± SEM"
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------
plt.ylim(0,90)
plt.xticks(x_positions, ["Sound", "Action-R", "Action-L"], fontsize=11)
plt.ylabel("Number of omissions", fontsize=11)
plt.title("Omissions per Block Type", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()

# Legend
handles = [
    plt.Line2D(
        [0], [0], marker="^", linestyle="", color=session_colors[s], 
        markerfacecolor="none", markersize=10, label=s
    )
    for s in df_omissions["session"].unique()
]

handles.append(
    plt.Line2D(
        [0], [0], marker="^", linestyle="", color="black",
        markersize=10, label="Overall mean ± SEM"
    )
)

plt.legend(
    handles=handles,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    title="Sessions"
)

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_OmissionsFiltered_TrianglePlot.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_OmissionsFiltered_TrianglePlot.svg", dpi=600)

#%% All licks together 


# ----------------------------
# SETTINGS
# ----------------------------
window = 20
valid_blocks = ["sound", "action-left", "action-right"]

animal_list = list(adaptive_sessions_to_analyze.keys())
sessions_dict = adaptive_sessions_to_analyze

# Load spout-tone mapping
mapping_file = Path(r"L:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv")
mapping_df = pd.read_csv(mapping_file)


# ============================================================
# FUNCTIONS
# ============================================================

def correct_side_for_block(block_type, stim, tone_map):
    if block_type == "sound":
        return tone_map[stim]
    if block_type == "action-left":
        return "left"
    if block_type == "action-right":
        return "right"
    return tone_map[stim]


def infer_lick_side(row, correct_side, tone_map):
    """Infer lick side using mapping + reward/punishment."""
    if row["omission"] == 1:
        return None

    # HIT → correct spout
    if row["reward"] == 1:
        return correct_side

    # INCORRECT → opposite spout
    if row["punishment"] == 1:
        return "left" if correct_side == "right" else "right"

    return None


# ============================================================
# COMPUTE LATENCIES FOR ALL SESSIONS
# ============================================================

lat_records = []

for animal in animal_list:

    # load tone mapping for this animal
    row_map = mapping_df[mapping_df["Animal"] == int(animal)].iloc[0]
    tone_map = {"8KHz": row_map["8KHz"], "16KHz": row_map["16KHz"]}

    for date_str in sessions_dict[animal]:

        # find file in all_files
        mask = (
            (all_files["animal"] == animal) &
            (all_files["task"] == "adaptive_sensorimotor") &
            (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"❌ Missing file: {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"]).copy()

        # compute latency
        df["latency"] = df["lick_time"] - df["RW_start"]

        # process trials
        for _, tr in df.iterrows():

            # determine stimulus
            if tr["8KHz"] == 1:
                stim = "8KHz"
            elif tr["16KHz"] == 1:
                stim = "16KHz"
            else:
                continue

            block_type = tr["block"]
            if block_type not in valid_blocks:
                continue

            correct_side = correct_side_for_block(block_type, stim, tone_map)
            lick_side = infer_lick_side(tr, correct_side, tone_map)

            if lick_side is None:
                continue

            lat_records.append({
                "animal": animal,
                "session": date_str,
                "block": block_type,
                "lick_side": lick_side,
                "latency": tr["latency"]
            })

lat_df = pd.DataFrame(lat_records)
print("\n=== LATENCY PER SIDE DATA ===")
print(lat_df.head())


# ============================================================
# PLOT: One dot per session, per block type, per side
# ============================================================

plt.figure(figsize=(12, 5), dpi=500)

block_order = ["sound", "action-left", "action-right"]
side_order = ["left", "right"]

# create x-ticks like:
# sound-left, sound-right, action-left-left, action-left-right, action-right-left, action-right-right
x_labels = []
x_pos = []
idx = 0
for blk in block_order:
    for side in side_order:
        x_labels.append(f"{blk}\n{side}")
        x_pos.append(idx)
        idx += 1

# plot dots
for _, row in lat_df.groupby(["animal", "session"]):
    for blk in block_order:
        for side in side_order:

            df_sub = row[(row["block"] == blk) & (row["lick_side"] == side)]
            if df_sub.empty:
                continue

            mean_lat = df_sub["latency"].mean()
            xpos = x_pos[block_order.index(blk)*2 + side_order.index(side)]

            plt.scatter(
                xpos + (np.random.rand() - 0.5) * 0.15,
                mean_lat,
                color=session_colors[row["session"].iloc[0]],
                s=70,
                edgecolor="white",
                linewidth=0.7
            )

# format
plt.xticks(x_pos, x_labels, rotation=0)
plt.ylabel("Mean Lick Latency (s)")
plt.title("Latency per spout side for each block type (one dot = one session)")
sns.despine()

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_Latency_PerSide.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_Latency_PerSide.svg", dpi=600)




#%% =====================================================================
# 4-PANEL LATENCY PLOT:
#   Panel 1: sound correct
#   Panel 2: sound incorrect
#   Panel 3: action-right all trials
#   Panel 4: action-left all trials
# ======================================================================

# Load spout mapping
spout_map_df = pd.read_csv(mapping_file_path)

def get_tone_map(animal_id):
    row = spout_map_df[spout_map_df["Animal"] == int(animal_id)].iloc[0]
    return {"8KHz": row["8KHz"], "16KHz": row["16KHz"]}

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


# ------------------------------------------------------------
# COLLECT LATENCIES (FIXED LICK-SIDE LOGIC)
# ------------------------------------------------------------

latency_records = []

for _, frow in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    animal = frow["animal"]
    tone_map = get_tone_map(animal)

    # Extract date from filename
    fname = Path(frow["path"]).name
    date_str = fname.split("_")[2]

    # Only plot selected sessions
    if f"{animal}_{date_str}" not in selected_session_ids:
        continue

    df = pd.read_csv(frow["path"]).copy()

    required = ["block", "lick_time", "trial_start",
                "reward", "punishment", "omission",
                "8KHz", "16KHz"]
    if not all(c in df.columns for c in required):
        continue

    # Compute latency
    df["latency"] = df["lick_time"] - df["RW_start"]

    # Determine stimulus
    df["stim"] = df.apply(
        lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else None),
        axis=1
    )

    for _, r in df.iterrows():

        if r["stim"] is None or r["omission"] == 1:
            continue

        blk = r["block"]
        stim = r["stim"]
        lat = r["latency"]

        if pd.isna(lat):
            continue

        # Determine CORRECT side
        cside = correct_side(blk, stim, tone_map)

        # Determine ACTUAL lick side
        if r["reward"] == 1:
            lick_side = cside
            correctness = "correct"
        elif r["punishment"] == 1:
            lick_side = opposite(cside)
            correctness = "incorrect"
        else:
            continue

        latency_records.append({
            "animal": animal,
            "session": date_str,
            "block": blk,
            "side": lick_side,
            "correctness": correctness,
            "latency": lat
        })

lat_df = pd.DataFrame(latency_records)
print("\n=== FIXED LATENCY DATA ===")
print(lat_df.head())

# Keep valid block types
valid_blocks = ["sound", "action-left", "action-right"]
lat_df = lat_df[lat_df["block"].isin(valid_blocks)]


# ------------------------------------------------------------
# SPLIT CORRECT / INCORRECT
# ------------------------------------------------------------
lat_correct = lat_df[lat_df["correctness"] == "correct"]
lat_incorrect = lat_df[lat_df["correctness"] == "incorrect"]
def plot_latency_panels(lat_df, title, filename):

    categories = [
        ("sound", "correct", "Sound — Correct"),
        ("sound", "incorrect", "Sound — Incorrect"),
        ("action-right", "all", "Action-Right — All trials"),
        ("action-left", "all", "Action-Left — All trials"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=500)

    for ax, (blk, cond, panel_title) in zip(axes, categories):

        # Build x-axis for this panel
        x_left = 0
        x_right = 1
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"], fontsize=11)

        # Loop through sessions (animal + session)
        for (animal, sess), df_sess in lat_df.groupby(["animal", "session"]):

            # Only plot requested sessions
            if f"{animal}_{sess}" not in selected_session_ids:
                continue

            color = session_colors.get(sess, "black")

            # Filter block
            d = df_sess[df_sess["block"] == blk]

            # Filter correctness
            if cond != "all":
                d = d[d["correctness"] == cond]

            # Compute average left/right latency
            left_vals = d[d["side"] == "left"]["latency"]
            right_vals = d[d["side"] == "right"]["latency"]

            left_mean = left_vals.mean() if len(left_vals) else None
            right_mean = right_vals.mean() if len(right_vals) else None

            # CONNECT dots if both sides exist
            if left_mean is not None and right_mean is not None:
                ax.plot(
                    [x_left, x_right],
                    [left_mean, right_mean],
                    color=color,
                    linewidth=2,
                    alpha=0.7
                )

            # SCATTER dots
            if left_mean is not None:
                ax.scatter(
                    x_left + (np.random.rand()*0.15 - 0.075),
                    left_mean,
                    color=color,
                    s=65,
                    edgecolor="white",
                    linewidth=0.5
                )

            if right_mean is not None:
                ax.scatter(
                    x_right + (np.random.rand()*0.15 - 0.075),
                    right_mean,
                    color=color,
                    s=65,
                    edgecolor="white",
                    linewidth=0.5
                )

        # Panel formatting
        ax.set_ylim(-0.1, 1)
        ax.set_title(panel_title, fontsize=12)
        ax.set_ylabel("Mean lick latency (s)", fontsize=11)
        sns.despine(ax=ax)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(f"{OUT}/{filename}.png", dpi=600)
    plt.savefig(f"{OUT}/{filename}.svg", dpi=600)


# RUN THE FIGURE
plot_latency_panels(
    lat_df,
    "Latency per Spout Side — 4-Category Panel Figure",
    "Latency_4Panels"
)

#%% =====================================================================
# TRIANGLE PLOT: Catch Trials per Block Type
# ======================================================================

# Block types we want
block_order = ["sound", "action-right", "action-left"]

# Collect catch trial counts
catch_records = []

for _, frow in all_files[all_files["task"] == "adaptive_sensorimotor"].iterrows():

    # Extract session ID
    fname = Path(frow["path"]).name
    date_str = fname.split("_")[2]  # e.g. 20251111
    session_id = f"{frow['animal']}_{date_str}"

    # only the 7 sessions
    if session_id not in selected_session_ids:
        continue

    df = pd.read_csv(frow["path"])

    # required columns
    if not all(c in df.columns for c in ["block", "catch_trial", "omission", "lick"]):
        continue

    for blk in block_order:

        df_blk = df[df["block"] == blk]

        # Correct catch = catch & omission==1
        correct_ct = df_blk[(df_blk["catch_trial"] == 1) & (df_blk["omission"] == 1)]

        # Incorrect catch = catch & lick==1
        incorrect_ct = df_blk[(df_blk["catch_trial"] == 1) & (df_blk["lick"] == 1)]

        catch_records.append({
            "animal": frow["animal"],
            "session": date_str,
            "session_id": session_id,
            "block": blk,
            "correct_ct": len(correct_ct),
            "incorrect_ct": len(incorrect_ct)
        })

catch_df = pd.DataFrame(catch_records)
print("\n=== CATCH TRIAL DATA ===")
print(catch_df)


#%% =====================================================================
# TRIANGLE PLOT
# ======================================================================

plt.figure(figsize=(10, 5), dpi=600)

# Two columns per block
# sound → (0,1)
# action-right → (2,3)
# action-left → (4,5)

x_positions = []
labels = []
for i, blk in enumerate(block_order):
    x_positions.extend([2*i, 2*i + 1])
    labels.extend([f"{blk}\ncorrect", f"{blk}\nincorrect"])

# ---------------------------
# Plot individual sessions
# ---------------------------
for sess, df_s in catch_df.groupby("session"):
    color = session_colors.get(sess, "black")

    for _, row in df_s.iterrows():

        blk = row["block"]
        blk_idx = block_order.index(blk)

        # correct
        x_c = 2*blk_idx + (np.random.rand()*0.12 - 0.06)
        # incorrect
        x_i = 2*blk_idx + 1 + (np.random.rand()*0.12 - 0.06)

        # Plot triangles (filled)
        plt.scatter(
            x_c, row["correct_ct"],
            marker="^",
            s=120,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=sess if blk_idx == 0 else None
        )

        plt.scatter(
            x_i, row["incorrect_ct"],
            marker="^",
            s=120,
            color=color,
            edgecolor="white",
            linewidth=0.8
        )



# ---------------------------
# Formatting
# ---------------------------
plt.ylim(-0.7,35)
plt.xticks(x_positions, labels, fontsize=10)
plt.ylabel("Number of Catch Trials", fontsize=12)
plt.title("Catch Trials — Correct vs Incorrect per Block Type", fontsize=14)

plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()

# Legend: sessions
handles = [
    plt.Line2D([0], [0], marker="^", color=c, linestyle="", markersize=9, label=s)
    for s, c in session_colors.items()
    if s in catch_df["session"].unique()
]
plt.legend(handles=handles, title="Sessions", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/Triangle_CatchTrials.png", dpi=600)
plt.savefig(f"{OUT}/Triangle_CatchTrials.svg", dpi=600)

#%% ======================================================================
# ======================================================================
#   FIXED PERSEVERATIVE ERROR COUNTS (correct logic + next 20 trials only)
# ======================================================================

def correct_side(block_type, stim, tone_map):
    if block_type == "sound":
        return tone_map[stim]
    elif block_type == "action-left":
        return "left"
    elif block_type == "action-right":
        return "right"


def classify_pe_after_switch(row, new_block, prev_block, tone_map):
    """
    CLASSIFICATION RULES:

    A→S : PE if habitual_side == prev_block_correct_side
    S→A : PE if habitual_side != new_block_correct_side
    """
    if row["omission"] == 1:
        return np.nan

    # Determine stimulus
    if row["8KHz"] == 1: stim = "8KHz"
    elif row["16KHz"] == 1: stim = "16KHz"
    else: return np.nan

    habitual = tone_map[stim]
    new_correct = correct_side(new_block, stim, tone_map)
    prev_correct = correct_side(prev_block, stim, tone_map)

    # HIT = never PE
    if row["reward"] == 1:
        return 0

    # INCORRECT trial
    if row["punishment"] == 1:

        # --- ACTION → SOUND ---
        if new_block == "sound":
            return 1 if habitual == prev_correct else 0

        # --- SOUND → ACTION ---
        if prev_block == "sound":
            return 1 if habitual != new_correct else 0

    return np.nan


def process_session_pe_counts_fixed(animal, sess_date, window=20):

    mask = (
        (all_files["animal"] == animal)
        & (all_files["task"] == "adaptive_sensorimotor")
        & (all_files["dt"].dt.strftime("%Y%m%d") == sess_date)
    )
    if not mask.any():
        return None

    row = all_files[mask].iloc[0]
    df = pd.read_csv(row["path"])
    tone_map = get_tone_map(animal)

    A2S_count = 0
    S2A_count = 0

    for i in range(1, len(df)):
        prev_block = df.loc[i-1, "block"]
        new_block  = df.loc[i,   "block"]

        if prev_block == new_block:
            continue

        # Identify transition
        if prev_block in ["action-left","action-right"] and new_block == "sound":
            direction = "A2S"
        elif prev_block == "sound" and new_block in ["action-left","action-right"]:
            direction = "S2A"
        else:
            continue

        # Take ONLY next 20 trials
        next20 = df.iloc[i:i+20].copy()

        for _, tr in next20.iterrows():
            pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)
            if not np.isnan(pe):
                if direction == "A2S":
                    A2S_count += pe
                else:
                    S2A_count += pe

    return {"A→S": A2S_count, "S→A": S2A_count}


# ------------------------------------------------------------
# RUN FOR ALL SESSIONS
# ------------------------------------------------------------

pe_records = []

for animal, sess_list in adaptive_sessions_to_analyze.items():
    for sess in sess_list:
        res = process_session_pe_counts_fixed(animal, sess)
        if res is None: continue

        pe_records.append({
            "animal": animal,
            "session": sess,
            "A→S": res["A→S"],
            "S→A": res["S→A"]
        })

pe_df = pd.DataFrame(pe_records)
print(pe_df)

# ======================================================================
#   TRIANGLE PLOT — PERSEVERATIVE ERROR COUNTS (Correct Style A)
# ======================================================================

# Use the already-built pe_df:
# columns: animal | session | A→S | S→A

if pe_df.empty:
    print("⚠️ No data available for triangle plot.")
else:
    plt.figure(figsize=(7, 5), dpi=500)

    x_positions = [0, 1]
    x_labels = ["A→S", "S→A"]

    # ---------------------------------------
    # Plot open triangles (per session)
    # ---------------------------------------
    for _, row in pe_df.iterrows():
        sess = row["session"]
        color = session_colors.get(sess, "gray")

        plt.scatter(
            x_positions[0], row["A→S"],
            marker="^", s=150,
            facecolors="none", edgecolors=color, linewidths=2,
            label=sess if f"_{sess}" not in plt.gca().get_legend_handles_labels()[1] else None
        )

        plt.scatter(
            x_positions[1], row["S→A"],
            marker="^", s=150,
            facecolors="none", edgecolors=color, linewidths=2
        )

    # ---------------------------------------
    # Plot filled black triangles (global mean)
    # ---------------------------------------
    mean_A2S = pe_df["A→S"].mean()
    mean_S2A = pe_df["S→A"].mean()

    plt.scatter(
        x_positions[0] + 0.15, mean_A2S,
        marker="^", s=180, color="black", label="Grand mean"
    )

    plt.scatter(
        x_positions[1] + 0.15, mean_S2A,
        marker="^", s=180, color="black"
    )

    # ---------------------------------------
    # Formatting
    # ---------------------------------------
    plt.xticks(x_positions, x_labels, fontsize=12)
    plt.ylabel("Number of Perseverative Errors", fontsize=13)
    plt.title("Perseverative Errors Around Block Switches", fontsize=14)

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    sns.despine()

    plt.legend(title="Sessions", frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{OUT}/Adaptive_PE_TrianglePlot.png", dpi=600)
    plt.savefig(f"{OUT}/Adaptive_PE_TrianglePlot.svg", dpi=600)

#%% ============================================================
# ------------------------------------------------------------
# Load spout mapping
# ------------------------------------------------------------
spout_map_df = pd.read_csv(mapping_file_path)

def get_tone_map(animal_id):
    """Returns dict { '8KHz': 'left'/'right', '16KHz': 'left'/'right' }."""
    row = spout_map_df[spout_map_df["Animal"] == int(animal_id)].iloc[0]
    return {"8KHz": row["8KHz"], "16KHz": row["16KHz"]}


# ------------------------------------------------------------
# Compute PE counts per switch (first 20 trials only)
# ------------------------------------------------------------
def compute_PE_postswitch(df, tone_map, window=20):
    """
    Returns:
        A2S_counts: list of PE counts (first 20 trials) for each Action→Sound switch
        S2A_counts: list of PE counts (first 20 trials) for each Sound→Action switch

    Definition of PE (per switch, first 20 trials of NEXT block):

      • S→A (sound → action):
          prev block = sound
          next block = action-left/right
          For each incorrect trial:
              prev_sound_side = tone_map[stim]
              action_side     = 'left' or 'right' (depending on next block)
              PE if prev_sound_side != action_side

      • A→S (action → sound):
          prev block = action-left/right
          next block = sound
          For each incorrect trial:
              prev_action_side = 'left' or 'right' (depending on prev block)
              sound_side       = tone_map[stim]
              PE if prev_action_side != sound_side
    """
    A2S_counts = []
    S2A_counts = []

    n_trials = len(df)

    for i in range(1, n_trials):
        prev_blk = df.loc[i - 1, "block"]
        next_blk = df.loc[i, "block"]

        # Detect direction
        if prev_blk in ["action-left", "action-right"] and next_blk == "sound":
            direction = "A→S"
        elif prev_blk == "sound" and next_blk in ["action-left", "action-right"]:
            direction = "S→A"
        else:
            continue

        # Window: first `window` trials of NEXT block
        next_df = df.iloc[i : i + window].copy()
        if next_df.empty:
            continue

        pe_count = 0

        for _, tr in next_df.iterrows():

            # Skip omissions
            if "omission" in tr.index and tr["omission"] == 1:
                continue

            # Which tone?
            if "8KHz" in tr.index and tr["8KHz"] == 1:
                stim = "8KHz"
            elif "16KHz" in tr.index and tr["16KHz"] == 1:
                stim = "16KHz"
            else:
                continue  # no tone flagged

            # Only consider incorrect trials
            if "punishment" not in tr.index or tr["punishment"] != 1:
                continue

            # ---- Sound → Action ----
            if direction == "S→A":
                prev_sound_side = tone_map[stim]  # from mapping
                action_side = "left" if next_blk == "action-left" else "right"
                if prev_sound_side != action_side:
                    pe_count += 1

            # ---- Action → Sound ----
            else:  # A→S
                prev_action_side = "left" if prev_blk == "action-left" else "right"
                sound_side = tone_map[stim]
                if prev_action_side != sound_side:
                    pe_count += 1

        if direction == "A→S":
            A2S_counts.append(pe_count)
        else:
            S2A_counts.append(pe_count)

    return A2S_counts, S2A_counts


# ------------------------------------------------------------
# MAIN LOOP – per-session PE (average per switch)
# ------------------------------------------------------------

PE_session_records = []

for animal, sess_list in adaptive_sessions_to_analyze.items():

    tone_map = get_tone_map(animal)

    for sess in sess_list:

        # Find the correct adaptive file from all_files
        mask = (
            (all_files["animal"] == animal)
            & (all_files["task"] == "adaptive_sensorimotor")
            & (all_files["dt"].dt.strftime("%Y%m%d") == sess)
        )

        if not mask.any():
            print(f"⚠ Missing adaptive session for {animal} {sess}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"]).copy()

        # Basic columns check
        needed = ["block", "8KHz", "16KHz", "reward", "punishment"]
        if not all(c in df.columns for c in needed):
            print(f"⚠ Missing required cols in {row['path']}")
            continue

        # Compute PE lists per switch
        A2S_list, S2A_list = compute_PE_postswitch(df, tone_map, window=20)

        if (len(A2S_list) == 0) and (len(S2A_list) == 0):
            print(f"⚠ No usable switches for {animal} {sess}")
            continue

        PE_session_records.append({
            "animal": animal,
            "session": sess,
            "PE_A2S": np.mean(A2S_list) if len(A2S_list) > 0 else np.nan,
            "PE_S2A": np.mean(S2A_list) if len(S2A_list) > 0 else np.nan,
            "n_switch_A2S": len(A2S_list),
            "n_switch_S2A": len(S2A_list)
        })

pe_df = pd.DataFrame(PE_session_records)

print("\n=== PERSEVERATIVE ERRORS PER SESSION (first 20 trials after each switch) ===")
print(pe_df)


# ------------------------------------------------------------
# TRIANGLE PLOT — open = per-session, black = group mean ± SEM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5), dpi=600)

x_pos = [0, 1]   # 0 = A→S, 1 = S→A

# ---- Open triangles: each session ----
for _, r in pe_df.iterrows():
    color = session_colors.get(r["session"], "gray")
    jitter = (np.random.rand() - 0.5) * 0.12

    # A→S
    if not pd.isna(r["PE_A2S"]):
        plt.scatter(
            x_pos[0] + jitter,
            r["PE_A2S"],
            s=160,
            marker="^",
            edgecolor=color,
            facecolor="none",
            linewidth=2,
            zorder=3
        )

    # S→A
    if not pd.isna(r["PE_S2A"]):
        plt.scatter(
            x_pos[1] + jitter,
            r["PE_S2A"],
            s=160,
            marker="^",
            edgecolor=color,
            facecolor="none",
            linewidth=2,
            zorder=3
        )

# ---- Group black triangle + SEM ----
mean_A2S = pe_df["PE_A2S"].mean()
mean_S2A = pe_df["PE_S2A"].mean()

sem_A2S = pe_df["PE_A2S"].std(ddof=1) / np.sqrt(pe_df["PE_A2S"].notna().sum()) \
          if pe_df["PE_A2S"].notna().sum() > 1 else np.nan
sem_S2A = pe_df["PE_S2A"].std(ddof=1) / np.sqrt(pe_df["PE_S2A"].notna().sum()) \
          if pe_df["PE_S2A"].notna().sum() > 1 else np.nan

# Slight x-offset for black triangles
plt.scatter(x_pos[0] + 0.12, mean_A2S, s=230, marker="^", color="black", zorder=5)
plt.scatter(x_pos[1] + 0.12, mean_S2A, s=230, marker="^", color="black", zorder=5)

plt.errorbar(
    x_pos[0] + 0.12, mean_A2S,
    yerr=sem_A2S,
    fmt="none",
    ecolor="black",
    capsize=5,
    linewidth=2,
    zorder=4
)
plt.errorbar(
    x_pos[1] + 0.12, mean_S2A,
    yerr=sem_S2A,
    fmt="none",
    ecolor="black",
    capsize=5,
    linewidth=2,
    zorder=4
)


plt.ylim(0,9)
plt.xticks(x_pos, ["Action → Sound", "Sound → Action"], fontsize=11)
plt.ylabel("Perseverative errors", fontsize=11)
plt.title("Perseverative Errors Across Sessions", fontsize=12)

sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT}/Triangle_PE_postswitch_First20.png", dpi=600)
plt.savefig(f"{OUT}/Triangle_PE_postswitch_First20.svg", dpi=600)

#  DEBUG SUMMARY — RAW PERSEVERATIVE ERROR COUNTS PER SESSION

print("\n================= RAW PERSEVERATIVE ERROR COUNTS =================\n")

# Recompute the full detailed counts exactly as in the plot
debug_records = []

for animal, sess_list in adaptive_sessions_to_analyze.items():

    tone_map = get_tone_map(animal)

    for sess in sess_list:

        # Find file in all_files
        mask = (
            (all_files["animal"] == animal)
            & (all_files["task"] == "adaptive_sensorimotor")
            & (all_files["dt"].dt.strftime("%Y%m%d") == sess)
        )

        if not mask.any():
            print(f"❌ No adaptive file found for {animal} {sess}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"]).copy()

        # Compute switch-wise PE (full list)
        A2S_list, S2A_list = compute_PE_postswitch(df, tone_map, window=20)

        print(f"\n--- Animal {animal} | Session {sess} ---")
        print(f"  A→S switches: {len(A2S_list)}")
        print(f"    PE counts per switch: {A2S_list}")
        if len(A2S_list) > 0:
            print(f"    Mean PE = {np.mean(A2S_list):.3f}")
        print()
        print(f"  S→A switches: {len(S2A_list)}")
        print(f"    PE counts per switch: {S2A_list}")
        if len(S2A_list) > 0:
            print(f"    Mean PE = {np.mean(S2A_list):.3f}")
        print()

        debug_records.append({
            "animal": animal,
            "session": sess,
            "n_A2S": len(A2S_list),
            "n_S2A": len(S2A_list),
            "A2S_list": A2S_list,
            "S2A_list": S2A_list,
            "mean_A2S": np.mean(A2S_list) if len(A2S_list) > 0 else np.nan,
            "mean_S2A": np.mean(S2A_list) if len(S2A_list) > 0 else np.nan
        })

debug_df = pd.DataFrame(debug_records)

print("\n================= SUMMARY TABLE =================\n")
print(debug_df)

#%% ============================================================
# PERFORMANCE PER SESSION (Adaptive Task)
# ============================================================

print("\n=== Computing performance per session (adaptive task) ===")

performance_records = []

for animal, date_list in adaptive_sessions_to_analyze.items():
    for date_str in date_list:

        # Find the correct adaptive file
        mask = (
            (all_files["animal"] == animal)
            & (all_files["task"] == "adaptive_sensorimotor")
            & (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"⚠ Missing adaptive session {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"])

        # Required columns
        needed = ["reward", "punishment", "omission", "catch_trial"]
        if not all(c in df.columns for c in needed):
            print(f"⚠ Missing columns in {row['path']}")
            continue

        # Extract counts
        n_correct = (df["reward"] == 1).sum()
        n_incorrect = (df["punishment"] == 1).sum()
        n_omissions = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].shape[0]

        denom = n_correct + n_incorrect + n_omissions
        performance = n_correct / denom if denom > 0 else np.nan

        performance_records.append({
            "animal": animal,
            "session": date_str,
            "session_id": f"{animal}_{date_str}",
            "correct": n_correct,
            "incorrect": n_incorrect,
            "omissions": n_omissions,
            "performance": performance
        })

perf_df = pd.DataFrame(performance_records)

print("\n=== PERFORMANCE TABLE ===")
print(perf_df)


#%% ============================================================
# PERFORMANCE PER SESSION (Adaptive Task)
# ============================================================

print("\n=== Computing performance per session (adaptive task) ===")

performance_records = []

for animal, date_list in adaptive_sessions_to_analyze.items():
    for date_str in date_list:

        # Find the correct adaptive file
        mask = (
            (all_files["animal"] == animal)
            & (all_files["task"] == "adaptive_sensorimotor")
            & (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"⚠ Missing adaptive session {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"])

        # Required columns
        needed = ["reward", "punishment", "omission", "catch_trial"]
        if not all(c in df.columns for c in needed):
            print(f"⚠ Missing columns in {row['path']}")
            continue

        # Extract counts
        n_correct = (df["reward"] == 1).sum()
        n_incorrect = (df["punishment"] == 1).sum()
        n_omissions = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].shape[0]

        denom = n_correct + n_incorrect + n_omissions
        performance = n_correct / denom if denom > 0 else np.nan

        performance_records.append({
            "animal": animal,
            "session": date_str,
            "session_id": f"{animal}_{date_str}",
            "correct": n_correct,
            "incorrect": n_incorrect,
            "omissions": n_omissions,
            "performance": performance
        })

perf_df = pd.DataFrame(performance_records)

print("\n=== PERFORMANCE TABLE ===")
print(perf_df)


#%% ============================================================
# PERFORMANCE PER SESSION (Adaptive Task)
# ============================================================

print("\n=== Computing performance per session (adaptive task) ===")

performance_records = []

for animal, date_list in adaptive_sessions_to_analyze.items():
    for date_str in date_list:

        # Find the correct adaptive file
        mask = (
            (all_files["animal"] == animal)
            & (all_files["task"] == "adaptive_sensorimotor")
            & (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"⚠ Missing adaptive session {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"])

        # Required columns
        needed = ["reward", "punishment", "omission", "catch_trial"]
        if not all(c in df.columns for c in needed):
            print(f"⚠ Missing columns in {row['path']}")
            continue

        # Extract counts
        n_correct = (df["reward"] == 1).sum()
        n_incorrect = (df["punishment"] == 1).sum()
        n_omissions = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].shape[0]

        denom = n_correct + n_incorrect + n_omissions
        performance = (n_correct / denom if denom > 0 else np.nan) * 100

        performance_records.append({
            "animal": animal,
            "session": date_str,
            "session_id": f"{animal}_{date_str}",
            "correct": n_correct,
            "incorrect": n_incorrect,
            "omissions": n_omissions,
            "performance": performance
        })

perf_df = pd.DataFrame(performance_records)

print("\n=== PERFORMANCE TABLE ===")
print(perf_df)


# ============================================================
# PLOT — PERFORMANCE PER SESSION (triangle-style)
# ============================================================

plt.figure(figsize=(8,5), dpi=500)

x = np.arange(len(perf_df))
y = perf_df["performance"].values

# Plot triangles (one per session)
for i, row in perf_df.iterrows():
    sess = row["session"]
    color = session_colors.get(sess, "gray")

    plt.scatter(
        i, row["performance"],
        marker="^",
        s=150,
        edgecolor=color,
        facecolor="none",
        linewidth=2,
        label=sess if f"_lab_{sess}" not in globals() else None
    )
    globals()[f"_lab_{sess}"] = True

# Horizontal line at chance = 0.5
plt.axhline(50, linestyle="--", color="black", alpha=0.6)

plt.ylim(0, 100)
plt.ylabel("Performance", fontsize=11)
plt.title("Adaptive Task — Performance per Session", fontsize=12)

sns.despine()
plt.tight_layout()

plt.savefig(f"{OUT}/Adaptive_PerformancePerSession.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_PerformancePerSession.svg", dpi=600)

#%% ============================================================
# PERFORMANCE PER BLOCK TYPE (sound, action-right, action-left)
# ============================================================

print("\n=== Computing performance per block type ===")

perf_block_records = []

block_types = ["sound", "action-right", "action-left"]

for animal, date_list in adaptive_sessions_to_analyze.items():
    for date_str in date_list:

        # Find session file
        mask = (
            (all_files["animal"] == animal) &
            (all_files["task"] == "adaptive_sensorimotor") &
            (all_files["dt"].dt.strftime("%Y%m%d") == date_str)
        )

        if not mask.any():
            print(f"⚠ Missing adaptive session {animal} {date_str}")
            continue

        row = all_files[mask].iloc[0]
        df = pd.read_csv(row["path"])

        # Required columns
        needed = ["block", "reward", "punishment", "omission", "catch_trial"]
        if not all(c in df.columns for c in needed):
            print(f"⚠ Missing columns in {row['path']}")
            continue

        for blk in block_types:

            df_blk = df[df["block"] == blk]
            if df_blk.empty:
                continue

            # Extract trial categories
            n_correct = (df_blk["reward"] == 1).sum()
            n_incorrect = (df_blk["punishment"] == 1).sum()
            n_omissions = df_blk[
                (df_blk["omission"] == 1) &
                (df_blk["catch_trial"] == 0)
            ].shape[0]

            denom = n_correct + n_incorrect + n_omissions
            perf = (n_correct / denom if denom > 0 else np.nan) * 100

            perf_block_records.append({
                "animal": animal,
                "session": date_str,
                "session_id": f"{animal}_{date_str}",
                "block_type": blk,
                "performance": perf
            })

perf_block_df = pd.DataFrame(perf_block_records)
print("\n=== PERFORMANCE PER BLOCK TYPE ===")
print(perf_block_df)

# TRIANGLE PLOT — PERFORMANCE PER BLOCK TYPE
# ============================================================

plt.figure(figsize=(9, 6), dpi=500)

block_order = ["sound", "action-right", "action-left"]
x_positions = np.arange(len(block_order))

# -----------------------
# Plot per-session dots
# -----------------------
used_labels = set()

for _, row in perf_block_df.iterrows():
    blk = row["block_type"]
    sess = row["session"]
    color = session_colors.get(sess, "black")
    x = block_order.index(blk)

    jitter = (np.random.rand() - 0.5) * 0.15

    label = sess if sess not in used_labels else None
    if label:
        used_labels.add(sess)

    plt.scatter(
        x + jitter,
        row["performance"],
        marker="^",
        s=130,
        facecolors="none",
        edgecolors=color,
        linewidths=2,
        label=label
    )

# -----------------------
# Mean ± SEM (solid black triangles)
# -----------------------
stats = perf_block_df.groupby("block_type")["performance"].agg(["mean", "sem"]).reindex(block_order)

plt.errorbar(
    x_positions + 0.12,
    stats["mean"].values,
    yerr=stats["sem"].values,
    fmt="^",
    markersize=14,
    color="black",
    capsize=5,
    linewidth=2,
    label="Mean ± SEM"
)

# -----------------------
# Formatting
# -----------------------
plt.ylim(0, 100)
plt.xticks(x_positions, ["Sound", "Action-R", "Action-L"], fontsize=12)
plt.ylabel("Performance", fontsize=11)
plt.title("Performance per Block Type", fontsize=12)
plt.axhline(50, linestyle="--", color="gray", alpha=0.5)

plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()

plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT}/Adaptive_Performance_BlockTypes.png", dpi=600)
plt.savefig(f"{OUT}/Adaptive_Performance_BlockTypes.svg", dpi=600)
