# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:39:03 2025

@author: JoanaCatarino
"""

import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# ========== User settings ==========

base_dir = r"L:/dmclab/Joana/Behavior/Data"

# Animals to include:
animals_of_interest = ["986235", "999770", "986167", "986168", "986169", "986170", "986171"]

# Strain dictionary
animal_strain = { 
    "986235" : "Tlx3", 
    "999770" : "Tlx3", 
    "986167" : "Fezf2", 
    "986168" : "Fezf2", 
    "986169" : "Fezf2", 
    "986170" : "Fezf2", 
    "986171" : "Fezf2"
    }

# Get block type per day
block_protocol = {    
    "986235": {1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:10,
               12:5,
               13:3, 14:3, 15:3},
    
    "999770": {1:10, 2:10, 3:10, 4:10, 5:10, 6:10,
               7:5, 8:5,
               9:3},
    
    "986167":{1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10,
              10:5, 11:5,
              12:3},
    
    "986168":{1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10,
              10:5,
              11:3},
    
    "986169":{1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10,
              8:5, 9:5,
              10:3},
    
    "986170":{1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10,
              10:5,
              11:3, 12:3, 13:3, 14:3},
    
    "986171":{1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:10,
              12:5, 13:5,
              14:3}
    }

# Where to save plots
OUT_1 = r"L:/dmclab/Joana/Behavior/Data/plots/behavior_cohort_1/blocks"
OUT = r"L:/dmclab/Joana/Behavior/Data/plots/behavior_cohort_1/blocks"

# ========== Task detector ==========

def detect_task_from_filename(filename: str) -> str:
    """Assigns a task type based on filename prefix."""
    if filename.startswith("FreeLick"):
        return "free_licking"
    elif filename.startswith("2ChoiceAuditory"):
        return "two_choice"
    elif filename.startswith("2ChoiceBlocks"):   # handles blocks of 10, 5, 3
        return "two_choice_blocks"
    elif filename.startswith("AdaptSensorimotor"):
        return "adaptive_sensorimotor"
    else:
        return "unknown"


# ========== Datetime extractor ==========

def extract_datetime_from_filename(filename: str):
    """
    Extract datetime from a filename of the form:
    ..._<YYYYMMDD>_<HHMMSS>_...
    """
    m = re.search(r"_(\d{8})_(\d{6})_", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

# ========== Get block size ==========

def get_block_size(animal, day):
    if animal in block_protocol and day in block_protocol[animal]:
        return block_protocol[animal][day]
    return None


# ========== Main loader ==========

def find_files_folder_structure(base_dir: str, animals: list[str]) -> pd.DataFrame:
    """
    Automatically finds CSV files inside:
    <base_dir>/<animal>/Behavior/<date_folder>/<file.csv>

    Returns DataFrame with:
        animal | task | dt | path
    """

    records = []

    for animal in animals:
        print(f"\n=== Searching data for animal {animal} ===")

        behavior_dir = Path(base_dir) / animal / "Behavior"
        if not behavior_dir.exists():
            print(f"⚠️ Behavior folder not found for {animal}")
            continue

        # list all day folders inside Behavior
        day_folders = [d for d in behavior_dir.iterdir() if d.is_dir()]
        if not day_folders:
            print(f"⚠️ No date folders found for {animal}")
            continue

        for day in sorted(day_folders):
            csv_files = list(day.glob("*.csv"))
            if not csv_files:
                continue
            if len(csv_files) > 1:
                print(f"⚠️ Multiple CSVs inside {day}, using first one.")

            csv_path = csv_files[0]
            filename = csv_path.name

            # Task type
            task = detect_task_from_filename(filename)
            
            # Date/Time from filename
            dt = extract_datetime_from_filename(filename)

            records.append({
                "animal": animal,
                "task": task,
                "dt": dt,
                "path": str(csv_path),
            })

    if not records:
        raise FileNotFoundError("❌ No CSVs found for selected animals.")

    df = pd.DataFrame(records)
    df = df.sort_values(["animal", "dt"]).reset_index(drop=True)
    return df

# ========== Run Importer ==========

all_files = find_files_folder_structure(base_dir, animals_of_interest)

all_files = all_files.sort_values(["animal", "task", "dt"]).reset_index(drop=True)

all_files["day_in_task"] = (
    all_files
    .groupby(["animal", "task"])
    .cumcount() + 1
)

all_files["strain"] = all_files["animal"].map(animal_strain)

def get_block_count(animal, day):
    if animal in block_protocol and day in block_protocol[animal]:
        return block_protocol[animal][day]
    return None

# Fill block_count only for two_choice_blocks task
all_files["block_count"] = all_files.apply(
    lambda row: get_block_count(row["animal"], row["day_in_task"])
    if row["task"] == "two_choice_blocks" else None,
    axis=1
)

print("\n=== IMPORT COMPLETED ===")
print(all_files)


# ------------------------------------------- Plotting ------------------------------------------------
#%%  Colors
# Define Task colors
task_colors = {
    "free_licking": "#B5D9CC",
    "two_choice": "#5B9E9A",
    "two_choice_blocks": "#9490B4",
    "adaptive_sensorimotor": "#815D9F"
}

# Define Strain Colors
strain_colors = {
    "Tlx3": "#020202",   # black
    "Fezf2": "#6D6A71",   # gray
}

animal_colors = {
    "986235": "#ff7f0e",
    "999770": "#2ca02c",
    "986167": "#d62728",
    "986168": "#9467bd",
    "986169": "#8c564b",
    "986170": "#e377c2",
    "986171": "#7f7f7f",
}
#%%  Plot 1

# 1) Total trials per task across animals

# Load total number of trials per csv
def get_total_trials_per_file(csv_path: str) -> int:
    """Returns the number of rows = total trials."""
    df = pd.read_csv(csv_path)
    return len(df)

# Build summary table for all animals and all tasks
summary_records = []

for _, row in all_files.iterrows():
    csv_path = row["path"]
    animal   = row["animal"]
    task     = row["task"]

    n_trials = get_total_trials_per_file(csv_path)

    summary_records.append({
        "animal": animal,
        "task": task,
        "total_trials": n_trials
    })

summary_df = pd.DataFrame(summary_records)

print("\n=== SUMMARY TABLE (total trials per session) ===")
print(summary_df)

# Compute per animal mean for dots 
animal_means = (
    summary_df
    .groupby(["animal", "task"])["total_trials"]
    .mean()
    .reset_index()
)

animal_means = animal_means.merge(
    all_files[["animal", "strain"]].drop_duplicates(),
    on="animal",
    how="left"
)

print("\n=== Animal MEANS ===")
print(animal_means)

# Compute group-level mean + SEM
task_stats = (
    summary_df
    .groupby("task")["total_trials"]
    .agg(["mean", "sem"])
    .reset_index()
)

print("\n=== GROUP MEANS ===")
print(task_stats)


# Reorder animal_means category
task_order = [
    "free_licking",
    "two_choice_blocks",
    "two_choice",
    "adaptive_sensorimotor"
]

# Reorder task_stats
task_stats = task_stats.set_index("task").loc[task_order].reset_index()

# Reorder animal_means category
animal_means["task"] = pd.Categorical(animal_means["task"],
                                      categories=task_order,
                                      ordered=True)
animal_means = animal_means.sort_values("task")


# Names for each bar
task_display_names = {
    "free_licking": "Free Licking",
    "two_choice_blocks": "Two-Choice Auditory\n Blocks",
    "two_choice": "Two-Choice Auditory",
    "adaptive_sensorimotor": "Adaptive Sensorimotor"
}


# Plot
plt.figure(figsize=(8, 6), dpi=500)

tasks = task_stats["task"].tolist()
means = task_stats["mean"].tolist()
sems  = task_stats["sem"].tolist()

x = np.arange(len(tasks))

bar_colors = [task_colors[t] for t in tasks]

# Bars
plt.bar(x, means, yerr=sems, capsize=6, color=bar_colors, alpha=0.85, width=0.5)

# Individual animal dots
for i, task in enumerate(tasks):
    dots = animal_means[animal_means["task"] == task]

    for _, row in dots.iterrows():
        strain = row["strain"]
        color = strain_colors.get(strain, "black")  # fallback to black if unknown
        jitter = (np.random.rand() - 0.5) * 0.15

        plt.scatter(
            x[i] + jitter,
            row["total_trials"],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=60,
            zorder=3
        )

custom_labels = [task_display_names[t] for t in tasks]
plt.xticks(x, custom_labels)
plt.ylabel("Total trials")
plt.title("Total Trials per Task Across Animals (mean ± SEM)")
plt.ylim(0,800)
ax = plt.gca()   # get current axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    plt.Line2D([0], [0],
               marker="o",
               color="w",
               markerfacecolor=col,
               markeredgecolor="white",
               markersize=10,
               label=strain)
    for strain, col in strain_colors.items()
]

plt.legend(handles=legend_handles, title="Strain", frameon=False)

plt.tight_layout()

plt.savefig(f"{OUT_1}/Total_Licks.png", dpi=500)
plt.savefig(f"{OUT_1}/Total_Licks.svg", dpi=500)


#%%  Plot 2

# 2) total licks per free licking day 

# Filter only Free Licking rows
free_df = all_files[all_files["task"] == "free_licking"]

lick_records = []

for _, row in free_df.iterrows():
    df = pd.read_csv(row["path"])
    n_licks = df[df["lick"] == 1].shape[0]

    lick_records.append({
        "animal": row["animal"],
        "strain": row["strain"],    # 🔥 include strain here
        "session": row["day_in_task"],
        "licks": n_licks
    })

lick_df = pd.DataFrame(lick_records)

lick_stats = (
    lick_df
    .groupby("session")["licks"]
    .agg(["mean", "sem"])
    .reset_index()
)

plt.figure(figsize=(10, 6), dpi=500)

sessions = lick_stats["session"].tolist()
means = lick_stats["mean"].tolist()
sems = lick_stats["sem"].tolist()
x = np.arange(len(sessions))

plt.bar(
    x, means,
    yerr=sems,
    capsize=6,
    color="#B5D9CC",
    width=0.6,
    alpha=0.85
)

# Dots colored by strain
for i, sess in enumerate(sessions):
    dots = lick_df[lick_df["session"] == sess]

    for _, row in dots.iterrows():
        color = strain_colors[row["strain"]]
        jitter = (np.random.rand() - 0.5) * 0.15

        plt.scatter(
            x[i] + jitter,
            row["licks"],
            color=color,
            s=60,
            edgecolor="white",
            linewidth=0.6,
            zorder=3
        )

plt.xticks(x, [f"Day {s}" for s in sessions])
plt.ylabel("Number of licks")
plt.title("Free Licking — Number of Licks per Day (mean ± SEM)")
plt.ylim(0, 800)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(handles=legend_handles, title="Strain", frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT_1}/FreeLicking_Licks.png", dpi=500)
plt.savefig(f"{OUT_1}/FreeLicking_Licks.svg", dpi=500)


#%%  Plot 3

# 3) Plot animals performance for each day within the different block types

# Extract trials until block switch
def extract_block_learning_metrics(csv_path, block_size, animal, session, strain):

    df = pd.read_csv(csv_path).copy()

    # Identify block identity
    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    # Find flip points
    block_starts = [0]
    for i in range(1, len(df)):
        if df["stim_pair"][i] != df["stim_pair"][i-1]:
            block_starts.append(i)
    block_starts.append(len(df))

    records = []

    for b in range(len(block_starts)-1):
        start = block_starts[b]
        end = block_starts[b+1]
        block_df = df.iloc[start:end]

        total_trials = len(block_df)

        records.append({
            "animal": animal,
            "strain": strain,
            "session": session,
            "block_index": b+1,
            "block_required": block_size,
            "total_trials": total_trials,
        })

    return records


# Collect all
block_learning_records = []

for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():
    block_learning_records.extend(
        extract_block_learning_metrics(
            row["path"],
            row["block_count"],
            row["animal"],
            row["day_in_task"],
            row["strain"]
        )
    )

block_learning_df = pd.DataFrame(block_learning_records)
print("\n=== BLOCK LEARNING RAW ===")
print(block_learning_df)

# Restart day count for block type

block_learning_df["block_day"] = (
    block_learning_df
    .groupby(["animal", "block_required"])["session"]
    .transform(lambda x: pd.factorize(x)[0] + 1)
)

print("\n=== BLOCK DAY ===")
print(block_learning_df)

# Per animal average (1 dot per animal per day)
animal_day_means = (
    block_learning_df
    .groupby(["animal", "strain", "block_required", "block_day"])["total_trials"]
    .mean()
    .reset_index()
)

print("\n=== ANIMAL DAY MEANS ===")
print(animal_day_means)

# Group mean + SEM
group_stats = (
    animal_day_means
    .groupby(["block_required", "block_day"])["total_trials"]
    .agg(["mean", "sem"])
    .reset_index()
)

print("\n=== GROUP STATS ===")
print(group_stats)


# Plot 

for blk in sorted(block_learning_df["block_required"].dropna().unique()):

    df_anim = animal_day_means[animal_day_means["block_required"] == blk]
    df_grp  = group_stats[group_stats["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    # --- Mean ± SEM ---
    plt.errorbar(
        df_grp["block_day"],
        df_grp["mean"],
        yerr=df_grp["sem"],
        fmt="-o",
        color="#333333",
        lw=1.8,
        markersize=6,
        capsize=4,
        label="Mean ± SEM"
    )

    # --- Animal dots ---
    for _, row in df_anim.iterrows():
        jitter = (np.random.rand() - 0.5) * 0.12
        plt.scatter(
            row["block_day"] + jitter,
            row["total_trials"],
            s=60,
            color=strain_colors[row["strain"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=3
        )

    # Formatting
    plt.title(f"{blk}-Trial Blocks — Trials to Block Switch")
    plt.xlabel("Training Day (reset to Day 1 for this block type)")
    plt.ylabel("Trials until Stimulus Switch")

    xticks = sorted(df_grp["block_day"].unique())
    plt.xticks(xticks, [f"Day {i}" for i in xticks])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Strain legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="white",
                   markersize=9, label=strain)
        for strain, c in strain_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Strain", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_TrialsPerDay.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_TrialsPerDay.svg", dpi=500)
    
#%%  Plot 4

# Overlay plot 3 with performance and %correct


# Extract trials until block switch
def extract_block_learning_full(csv_path, block_size, animal, session, strain):
    df = pd.read_csv(csv_path).copy()

    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    block_starts = [0]
    for i in range(1, len(df)):
        if df["stim_pair"][i] != df["stim_pair"][i-1]:
            block_starts.append(i)
    block_starts.append(len(df))

    recs = []
    for b in range(len(block_starts)-1):
        start = block_starts[b]
        end = block_starts[b+1]
        block_df = df.iloc[start:end]

        total = len(block_df)
        correct = (block_df["reward"]==1).sum()
        incorrect = (block_df["punishment"]==1).sum()
        omissions = (block_df["omission"]==1).sum()

        performance = 100*correct/max(correct+incorrect+omissions,1)
        pct_correct = 100*correct/max(correct+incorrect,1)

        recs.append({
            "animal": animal,
            "strain": strain,
            "session": session,
            "block_index": b+1,
            "block_required": block_size,
            "total_trials": total,
            "correct": correct,
            "incorrect": incorrect,
            "omissions": omissions,
            "performance": performance,
            "pct_correct": pct_correct
        })

    return recs


# Build full block learning DF
records = []
for _, row in all_files[all_files["task"]=="two_choice_blocks"].iterrows():
    records.extend(
        extract_block_learning_full(
            row["path"], row["block_count"],
            row["animal"], row["day_in_task"], row["strain"]
        )
    )

block_learning_df = pd.DataFrame(records)

# ADD block_day (this was the missing part)
block_learning_df["block_day"] = (
    block_learning_df
    .groupby(["animal","block_required"])["session"]
    .transform(lambda s: pd.factorize(s)[0] + 1)
)

print("\n=== BLOCK LEARNING (PLOT 4) ===")
print(block_learning_df.head())

# Per animal daily means (ONE DOT PER ANIMAL x DAY)
animal_day_means = (
    block_learning_df
    .groupby(["animal","strain","block_required","block_day"])
    .agg({
        "total_trials":"mean",
        "performance":"mean",
        "pct_correct":"mean"
    })
    .reset_index()
)

group_stats = (
    animal_day_means
    .groupby(["block_required","block_day"])
    .agg({
        "total_trials":["mean","sem"],
        "performance":["mean","sem"],
        "pct_correct":["mean","sem"]
    })
)

group_stats.columns = ["_".join(col) for col in group_stats.columns]
group_stats = group_stats.reset_index()

for blk in sorted(block_learning_df["block_required"].dropna().unique()):
    
    df_anim = animal_day_means[animal_day_means["block_required"]==blk]
    df_grp  = group_stats[group_stats["block_required"]==blk]

    fig, ax1 = plt.subplots(figsize=(10,6), dpi=500)

    # ========== LEFT AXIS: TRIALS ==========
    ax1.errorbar(
        df_grp["block_day"],
        df_grp["total_trials_mean"],
        yerr=df_grp["total_trials_sem"],
        fmt="-o", color="#333333",
        lw=2, markersize=6, capsize=4
    )

    for _, row in df_anim.iterrows():
        jitter = (np.random.rand()-0.5)*0.12
        ax1.scatter(
            row["block_day"]+jitter, row["total_trials"],
            color=strain_colors[row["strain"]],
            s=60, edgecolor="white", linewidth=0.6
        )

    ax1.set_ylabel("Trials to Stimulus Switch")
    ax1.set_xlabel("Training Day (for this block type)")
    ax1.set_title(f"{blk}-Trial Blocks — Trials, Performance, %Correct")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ========== RIGHT AXIS ==========
    ax2 = ax1.twinx()

    ax2.errorbar(
        df_grp["block_day"],
        df_grp["performance_mean"],
        yerr=df_grp["performance_sem"],
        fmt="-s", color="#5B9BD5",
        lw=2, markersize=6, capsize=4,
        label="Performance"
    )

    ax2.errorbar(
        df_grp["block_day"],
        df_grp["pct_correct_mean"],
        yerr=df_grp["pct_correct_sem"],
        fmt="-^", color="#ED7D31",
        lw=2, markersize=6, capsize=4,
        label="% Correct"
    )

    ax2.set_ylabel("Performance (%)")
    ax2.spines["top"].set_visible(False)

    # Combined legend
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc="upper right", frameon=False)

    plt.tight_layout()

    plt.savefig(f"{OUT}/Block_{blk}_Trials_Performance.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_Trials_Performance.svg", dpi=500)
    
#%%  Plot 5 — Number of block switches per day (per block type)

print("\n=== PLOT 5: BLOCK SWITCHES ===")

# Count switches per animal per day for each block type
switch_records = []

for (animal, blk, day), df_day in block_learning_df.groupby(
        ["animal", "block_required", "block_day"]):

    # number of blocks that day for that animal
    n_blocks = df_day["block_index"].nunique()

    # switches = blocks - 1
    n_switches = max(n_blocks - 1, 0)

    strain = df_day["strain"].iloc[0]

    switch_records.append({
        "animal": animal,
        "strain": strain,
        "block_required": blk,
        "block_day": day,
        "switches": n_switches
    })

switch_df = pd.DataFrame(switch_records)
print(switch_df.head())

# Compute mean ± SEM across animals
switch_stats = (
    switch_df
    .groupby(["block_required", "block_day"])["switches"]
    .agg(["mean", "sem"])
    .reset_index()
)

# ========== PLOTTING ==========

for blk in sorted(switch_df["block_required"].unique()):

    df_anim = switch_df[switch_df["block_required"] == blk]
    df_grp  = switch_stats[switch_stats["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    # --- Group mean ± SEM ---
    plt.errorbar(
        df_grp["block_day"],
        df_grp["mean"],
        yerr=df_grp["sem"],
        fmt="-o",
        color="#333333",
        lw=1.8,
        markersize=6,
        capsize=4,
        label="Mean ± SEM"
    )

    # --- Animal dots ---
    for _, row in df_anim.iterrows():
        jitter = (np.random.rand() - 0.5) * 0.12
        plt.scatter(
            row["block_day"] + jitter,
            row["switches"],
            s=60,
            color=strain_colors[row["strain"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=3
        )

    # Formatting
    plt.title(f"{blk}-Trial Blocks — Number of Block Switches per Day")
    plt.xlabel("Training Day (reset to Day 1 for this block type)")
    plt.ylabel("Number of Switches")

    xticks = sorted(df_grp["block_day"].unique())
    plt.xticks(xticks, [f"Day {i}" for i in xticks])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Strain legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="white",
                   markersize=9, label=strain)
        for strain, c in strain_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Strain", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_SwitchesPerDay.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_SwitchesPerDay.svg", dpi=500)

#%%  Plot 6 — Block Switch Success (correct logic)
print("\n=== PLOT 6: BLOCK SWITCH SUCCESS ===")

switch_success_records = []

for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():

    df = pd.read_csv(row["path"]).copy()
    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    # ==== Find block switches ====
    switch_indices = []
    for i in range(1, len(df)):
        if df["stim_pair"][i] != df["stim_pair"][i-1]:
            switch_indices.append(i)  # i = first trial of new block

    n_switches = len(switch_indices)

    n_success = 0

    # ==== Evaluate first trial after each switch ====
    for idx in switch_indices:
        first_trial = df.iloc[idx]

        if first_trial["reward"] == 1:
            n_success += 1

    if n_switches > 0:
        success_rate = n_success / n_switches
    else:
        success_rate = np.nan

    animal  = row["animal"]
    strain  = row["strain"]
    blk_req = row["block_count"]
    session = row["day_in_task"]

    # Find block_day
    block_day = block_learning_df[
        (block_learning_df["animal"] == animal) &
        (block_learning_df["session"] == session) &
        (block_learning_df["block_required"] == blk_req)
    ]["block_day"].iloc[0]

    switch_success_records.append({
        "animal": animal,
        "strain": strain,
        "block_required": blk_req,
        "session": session,
        "block_day": block_day,
        "n_switches": n_switches,
        "n_success": n_success,
        "success_rate": success_rate
    })

switch_df = pd.DataFrame(switch_success_records)
print("\n=== SWITCH SUCCESS RAW ===")
print(switch_df.head())

# ==== group mean ± SEM ====
switch_stats = (
    switch_df
    .groupby(["block_required", "block_day"])["success_rate"]
    .agg(["mean", "sem"])
    .reset_index()
)

# ========== PLOT ==========

for blk in sorted(switch_df["block_required"].unique()):

    df_anim = switch_df[switch_df["block_required"] == blk]
    df_grp  = switch_stats[switch_stats["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    # Mean ± SEM
    plt.errorbar(
        df_grp["block_day"],
        df_grp["mean"],
        yerr=df_grp["sem"],
        fmt="-o",
        color="#333333",
        lw=1.8,
        markersize=6,
        capsize=4,
        label="Mean ± SEM"
    )

    # Animal dots
    for _, row in df_anim.iterrows():
        jitter = (np.random.rand() - 0.5) * 0.12
        plt.scatter(
            row["block_day"] + jitter,
            row["success_rate"],
            s=60,
            color=strain_colors[row["strain"]],
            edgecolor="white",
            linewidth=0.6,
            zorder=3
        )

    plt.title(f"{blk}-Trial Blocks — Switch Success Rate\n(First Trial After EACH Block Switch)")
    plt.xlabel("Training Day (reset for this block type)")
    plt.ylabel("Switch Success Rate (0–1)")

    xticks = sorted(df_grp["block_day"].unique())
    plt.xticks(xticks, [f"Day {i}" for i in xticks])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="white",
                   markersize=9, label=strain)
        for strain, c in strain_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Strain", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_SwitchSuccess.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_SwitchSuccess.svg", dpi=500)

#%%  Plot 7 — Number of Switches in FIRST vs LAST session per block type
print("\n=== PLOT 7: FIRST vs LAST session — Number of Switches ===")

switch_summary = []

# Filter only block sessions
block_sessions = all_files[all_files["task"] == "two_choice_blocks"]

for animal in animals_of_interest:

    df_animal = block_sessions[block_sessions["animal"] == animal]

    for blk in sorted(df_animal["block_count"].dropna().unique()):

        df_blk = df_animal[df_animal["block_count"] == blk]

        # First and last session numbers
        sessions = sorted(df_blk["day_in_task"].unique())
        if len(sessions) == 0:
            continue

        first_session = sessions[0]
        last_session  = sessions[-1]

        for sess_label, sess in zip(["first", "last"], [first_session, last_session]):

            row = df_blk[df_blk["day_in_task"] == sess].iloc[0]
            csv_path = row["path"]

            df = pd.read_csv(csv_path).copy()
            df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

            # Count block switches
            switch_count = 0
            for i in range(1, len(df)):
                if df["stim_pair"][i] != df["stim_pair"][i-1]:
                    switch_count += 1

            switch_summary.append({
                "animal": animal,
                "strain": row["strain"],
                "block_required": blk,
                "session_type": sess_label,
                "n_switches": switch_count
            })

switch_summary_df = pd.DataFrame(switch_summary)
print("\n=== SWITCH SUMMARY (FIRST vs LAST) ===")
print(switch_summary_df.head())

# ---- Plotting ----
for blk in sorted(switch_summary_df["block_required"].unique()):

    df_blk = switch_summary_df[switch_summary_df["block_required"] == blk]

    df_blk["session_type"] = pd.Categorical(df_blk["session_type"],
                                            categories=["first", "last"],
                                            ordered=True)

    plt.figure(figsize=(4,6), dpi=500)

    # Means for bars
    means = df_blk.groupby("session_type")["n_switches"].mean()
    sems  = df_blk.groupby("session_type")["n_switches"].sem()

    # Move bars closer together
    x = np.array([0, 0.25])      # <<< spacing reduced
    bar_width = 0.20

    plt.bar(x, means, yerr=sems, capsize=6,
            color=["#6D6A71", "#020202"], width=bar_width, alpha=0.7)

    # Individual animal dots
    for _, row in df_blk.iterrows():
        jitter = (np.random.rand() - 0.5) * 0.06
        xpos = x[0] if row["session_type"] == "first" else x[1]
        plt.scatter(
            xpos + jitter,
            row["n_switches"],
            color=strain_colors[row["strain"]],
            edgecolor="white",
            s=70,
            linewidth=0.6,
            zorder=3
        )

    plt.xticks(x, ["First Session", "Last Session"], fontsize=11)
    plt.ylabel("Number of Block Switches", fontsize=11)
    plt.title(f"Block of {blk}-Trials — First vs Last Session", fontsize=12)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)

    # Strain legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="white",
                   markersize=9, label=strain)
        for strain, c in strain_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Strain", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_SwitchCount_FirstLast.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_SwitchCount_FirstLast.svg", dpi=500)

#%%  Plot 8 — Final version: per-animal colored triangles + SEM + total block switches

print("\n=== PLOT 8: colored triangles + SEM + total switches ===")

triangle_summary = []

# ---- Extract per-session metrics ----
for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():

    df = pd.read_csv(row["path"]).copy()
    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    # Count block switches
    switches = sum(df["stim_pair"].shift() != df["stim_pair"]) - 1

    total_trials = len(df)

    triangle_summary.append({
        "animal": row["animal"],
        "strain": row["strain"],
        "block_required": row["block_count"],
        "session": row["day_in_task"],
        "trials": total_trials,
        "switches": switches
    })

triangle_df = pd.DataFrame(triangle_summary)

# ---- Average PER ANIMAL PER BLOCK TYPE (numeric cols only) ----
triangle_avg = (
    triangle_df
    .groupby(["animal", "block_required"])[["trials", "switches"]]
    .agg(["mean", "sem"])
    .reset_index()
)

triangle_avg.columns = [
    "animal", "block_required",
    "trials_mean", "trials_sem",
    "switches_mean", "switches_sem"
]

print("\n=== TRIANGLE AVERAGES ===")
print(triangle_avg.head())


# ===========================
#     PLOTTING
# ===========================
for blk in sorted(triangle_avg["block_required"].dropna().unique()):

    df_blk = triangle_avg[triangle_avg["block_required"] == blk]

    # Group means
    mean_trials = df_blk["trials_mean"].mean()
    sem_trials  = df_blk["trials_mean"].sem()

    mean_switches = df_blk["switches_mean"].mean()
    sem_switches  = df_blk["switches_mean"].sem()

    fig, axes = plt.subplots(1, 2, figsize=(5, 4), dpi=500)

    #-----------------------------------------------------------------------
    # LEFT — Trials performed
    #-----------------------------------------------------------------------
    ax = axes[0]

    for _, r in df_blk.iterrows():
        ax.scatter(
            1, r["trials_mean"],
            marker="^",
            facecolors="none",
            edgecolors=animal_colors[r["animal"]],   # <--- use your real colors
            s=130,
            linewidths=1.5,
            zorder=2
        )

    ax.errorbar(
        1.25, mean_trials,
        yerr=sem_trials,
        fmt="^",
        markersize=10,
        color="black",
        capsize=4,
        elinewidth=1.4,
        zorder=3
    )

    ax.set_ylabel("Trials performed")
    ax.set_xlim(0.5, 1.8)
    ax.set_ylim(0,800)
    ax.set_xticks([])
   

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    #-----------------------------------------------------------------------
    # RIGHT — Total block switches
    #-----------------------------------------------------------------------
    ax = axes[1]

    for _, r in df_blk.iterrows():
        ax.scatter(
            1, r["switches_mean"],
            marker="^",
            facecolors="none",
            edgecolors=animal_colors[r["animal"]],   # <--- use your real colors
            s=130,
            linewidths=1.5,
            zorder=2
        )

    ax.errorbar(
        1.25, mean_switches,
        yerr=sem_switches,
        fmt="^",
        markersize=10,
        color="black",
        capsize=4,
        elinewidth=1.4,
        zorder=3
    )

    ax.set_ylabel("Total block switches")
    ax.set_xlim(0.5, 1.8)
    ax.set_xticks([])
    ax.set_ylim(0, max(df_blk["switches_mean"].max() * 1.3, 2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Blocks of {blk} trials", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_TrianglePlot_TotalSwitches_SEM.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_TrianglePlot_TotalSwitches_SEM.svg", dpi=500)

#%%  Plot 9 — Two-Choice Auditory: Performance and %Correct Across Sessions
print("\n=== PLOT 9: Two-Choice Auditory — Performance and %Correct Across Sessions ===")

# Filter only 2-choice auditory sessions
two_choice_df = all_files[all_files["task"] == "two_choice"].copy()

if two_choice_df.empty:
    print("⚠️ No 2-choice auditory sessions found.")
else:

    perf_records = []

    # ---- Extract performance metrics per session ----
    for _, row in two_choice_df.iterrows():
        df = pd.read_csv(row["path"]).copy()

        correct = (df["reward"] == 1).sum()
        incorrect = (df["punishment"] == 1).sum()
        omissions = (df["omission"] == 1).sum()

        # performance: correct / (correct + incorrect + omissions)
        if (correct + incorrect + omissions) > 0:
            performance = 100 * correct / (correct + incorrect + omissions)
        else:
            performance = np.nan

        # %correct: correct / (correct + incorrect)
        if (correct + incorrect) > 0:
            pct_correct = 100 * correct / (correct + incorrect)
        else:
            pct_correct = np.nan

        perf_records.append({
            "animal": row["animal"],
            "strain": row["strain"],
            "session": row["day_in_task"],
            "performance": performance,
            "pct_correct": pct_correct
        })

    perf_df = pd.DataFrame(perf_records)

    # ---- Group mean ± SEM across animals ----
    perf_stats = (
        perf_df
        .groupby("session")
        .agg({
            "performance": ["mean", "sem"],
            "pct_correct": ["mean", "sem"]
        })
        .reset_index()
    )

    # Flatten column names
    perf_stats.columns = [
        "session", 
        "perf_mean", "perf_sem",
        "corr_mean", "corr_sem"
    ]

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(10, 6), dpi=500)

    # Plot mean ± SEM for performance
    ax.errorbar(
        perf_stats["session"],
        perf_stats["perf_mean"],
        yerr=perf_stats["perf_sem"],
        fmt="-o",
        color="#5B9BD5",
        lw=2, capsize=4,
        label="Performance (mean ± SEM)"
    )

    # Plot mean ± SEM for % correct
    ax.errorbar(
        perf_stats["session"],
        perf_stats["corr_mean"],
        yerr=perf_stats["corr_sem"],
        fmt="-s",
        color="#ED7D31",
        lw=2, capsize=4,
        label="% Correct (mean ± SEM)"
    )

    # Plot individual animals
    for animal in perf_df["animal"].unique():
        df_an = perf_df[perf_df["animal"] == animal]
        color = strain_colors[df_an["strain"].iloc[0]]

        ax.scatter(
            df_an["session"],
            df_an["performance"],
            color=color,
            s=55,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.7,
            label=None
        )

    ax.set_xlabel("Training Session (Day in Task)")
    ax.set_ylabel("Performance (%)")
    ax.set_title("Two-Choice Auditory: Performance & %Correct Across Sessions")

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Strain legend
    legend_handles = [
        plt.Line2D([0], [0],
                   marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="white",
                   markersize=8, label=strain)
        for strain, c in strain_colors.items()
    ]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0],
              frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/TwoChoiceAuditory_Performance.png", dpi=500)
    plt.savefig(f"{OUT}/TwoChoiceAuditory_Performance.svg", dpi=500)
    
#%% PLOT A — Two-Choice Auditory: Performance + %Correct (Mean ± SEM)
print("\n=== PLOT A: Overall Learning Curve ===")

two_choice = all_files[all_files["task"] == "two_choice"]

if two_choice.empty:
    print("⚠ No two-choice sessions found.")
else:
    perf_records = []

    for _, row in two_choice.iterrows():
        df = pd.read_csv(row["path"])

        correct = (df["reward"] == 1).sum()
        incorrect = (df["punishment"] == 1).sum()
        omissions = (df["omission"] == 1).sum()

        # Performance = correct / all trials
        perf = 100 * correct / (correct + incorrect + omissions)

        # %Correct = correct / (correct + incorrect)
        pct = 100 * correct / (correct + incorrect) if (correct+incorrect)>0 else np.nan

        perf_records.append({
            "session": row["day_in_task"],
            "performance": perf,
            "pct_correct": pct
        })

    perf_df = pd.DataFrame(perf_records)

    # Mean ± SEM per session
    perf_stats = perf_df.groupby("session").agg(["mean","sem"]).reset_index()
    perf_stats.columns = [
        "session",
        "perf_mean", "perf_sem",
        "corr_mean", "corr_sem"
    ]

    plt.figure(figsize=(9,6), dpi=500)

    # Performance curve
    plt.errorbar(
        perf_stats["session"], perf_stats["perf_mean"],
        yerr=perf_stats["perf_sem"],
        fmt="-o", lw=2, color="#5B9BD5", capsize=4,
        label="Performance (mean ± SEM)"
    )

    # %Correct curve
    plt.errorbar(
        perf_stats["session"], perf_stats["corr_mean"],
        yerr=perf_stats["corr_sem"],
        fmt="-o", lw=2, color="#ED7D31", capsize=4,
        label="% Correct (mean ± SEM)"
    )

    plt.xlabel("Session")
    plt.ylabel("Percentage (%)")
    plt.title("Two-Choice Auditory — Learning Curve")
    plt.ylim(0, 100)
    plt.legend(frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/TwoChoice_LearningCurve.png", dpi=500)
    plt.savefig(f"{OUT}/TwoChoice_LearningCurve.svg", dpi=500)

#%% PLOT B — Two-Choice Auditory: %Correct for 8kHz vs 16kHz
print("\n=== PLOT B: Stimulus-Specific Accuracy ===")

stim_records = []

for _, row in two_choice.iterrows():
    df = pd.read_csv(row["path"])

    df8  = df[df["8KHz"]  == 1]
    df16 = df[df["16KHz"] == 1]

    pct_8 = np.nan
    pct_16 = np.nan

    if len(df8) > 0:
        c = (df8["reward"]==1).sum()
        ic = (df8["punishment"]==1).sum()
        pct_8 = 100 * c / (c + ic)

    if len(df16) > 0:
        c = (df16["reward"]==1).sum()
        ic = (df16["punishment"]==1).sum()
        pct_16 = 100 * c / (c + ic)

    stim_records.append({
        "session": row["day_in_task"],
        "pct_8": pct_8,
        "pct_16": pct_16
    })

stim_df = pd.DataFrame(stim_records)

# Mean ± SEM
stim_stats = stim_df.groupby("session").agg(["mean","sem"]).reset_index()
stim_stats.columns = [
    "session",
    "m8", "sem8",
    "m16", "sem16"
]

plt.figure(figsize=(9,6), dpi=500)

# 8 kHz
plt.errorbar(
    stim_stats["session"], stim_stats["m8"],
    yerr=stim_stats["sem8"],
    fmt="-o", lw=2, color="#3A7BD5", capsize=4,
    label="8 kHz (mean ± SEM)"
)

# 16 kHz
plt.errorbar(
    stim_stats["session"], stim_stats["m16"],
    yerr=stim_stats["sem16"],
    fmt="-o", lw=2, color="#D55E00", capsize=4,
    label="16 kHz (mean ± SEM)"
)

plt.xlabel("Session")
plt.ylabel("% Correct")
plt.title("Two-Choice Auditory — 8 kHz vs 16 kHz Accuracy")
plt.ylim(0, 100)
plt.legend(frameon=False)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_StimulusAccuracy.png", dpi=500)
plt.savefig(f"{OUT}/TwoChoice_StimulusAccuracy.svg", dpi=500)

#%%  Plot 6 new — Per-animal Switch Success Curves (transparent lines + vertical day markers)
print("\n=== PLOT 6 (UPDATED): PER-ANIMAL SWITCH SUCCESS CURVES ===")

switch_success_records = []

for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():

    df = pd.read_csv(row["path"]).copy()
    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    # ---- Find block switches ----
    switch_indices = []
    for i in range(1, len(df)):
        if df["stim_pair"][i] != df["stim_pair"][i-1]:
            switch_indices.append(i)  # i = first trial after switch

    n_switches = len(switch_indices)
    n_success = 0

    # ---- Evaluate first trial after each switch ----
    for idx in switch_indices:
        first_trial = df.iloc[idx]
        if first_trial["reward"] == 1:
            n_success += 1

    success_rate = n_success / n_switches if n_switches > 0 else np.nan

    animal  = row["animal"]
    blk_req = row["block_count"]
    session = row["day_in_task"]

    # Retrieve aligned block_day
    block_day = block_learning_df[
        (block_learning_df["animal"] == animal) &
        (block_learning_df["session"] == session) &
        (block_learning_df["block_required"] == blk_req)
    ]["block_day"].iloc[0]

    switch_success_records.append({
        "animal": animal,
        "block_required": blk_req,
        "session": session,
        "block_day": block_day,
        "success_rate": success_rate
    })

switch_df = pd.DataFrame(switch_success_records)

# ---- Aggregate per animal per block_day ----
switch_animal_mean = (
    switch_df
    .groupby(["animal", "block_required", "block_day"])["success_rate"]
    .mean()
    .reset_index()
)


# ========== PLOT (PER BLOCK TYPE) ==========
for blk in sorted(switch_df["block_required"].unique()):

    df_blk = switch_animal_mean[switch_animal_mean["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    all_days = sorted(df_blk["block_day"].unique())

    # === Vertical dashed lines for each training day ===
    for d in all_days:
        plt.axvline(
            x=d,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.4,
            zorder=0
        )

    # === per-animal curves ===
    for animal in sorted(df_blk["animal"].unique()):
        df_a = df_blk[df_blk["animal"] == animal].sort_values("block_day")

        # Transparent line (same color but alpha)
        plt.plot(
            df_a["block_day"],
            df_a["success_rate"],
            "-",
            color=animal_colors[animal],
            alpha=0.3,            # <-- line transparency
            linewidth=2
        )

        # Solid dots
        plt.scatter(
            df_a["block_day"],
            df_a["success_rate"],
            color=animal_colors[animal],
            edgecolor="white",
            linewidth=0.6,
            s=70,
            zorder=5
        )

    # Formatting
    plt.title(f"Blocks of {blk} trials — Switch Success per Animal")
    plt.xlabel("Training Day")
    plt.ylabel("Switch Success Rate")

    plt.xticks(all_days, [f"Day {d}" for d in all_days])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylim(-0.05, 1.05)

    # Animal legend
    handles = [
        plt.Line2D([0], [0],
                   marker="o", color=animal_colors[a],
                   markerfacecolor=animal_colors[a],
                   markersize=8, linewidth=2,
                   label=a)
        for a in sorted(df_blk["animal"].unique())
    ]
    plt.legend(handles=handles, title="Animal", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_SwitchSuccess_PerAnimal_new.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_SwitchSuccess_PerAnimal_new.svg", dpi=500)

#%%  Plot 6B — Per-animal Performance Curves (same style as Switch Success)
print("\n=== PLOT 6B: PERFORMANCE PER ANIMAL ACROSS BLOCK DAYS ===")

performance_records = []

for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():

    df = pd.read_csv(row["path"]).copy()

    # Performance calculation for whole session
    correct    = (df["reward"] == 1).sum()
    incorrect  = (df["punishment"] == 1).sum()
    omissions  = (df["omission"] == 1).sum()

    if (correct + incorrect + omissions) > 0:
        performance = 100 * correct / (correct + incorrect + omissions)
    else:
        performance = np.nan

    animal  = row["animal"]
    blk_req = row["block_count"]
    session = row["day_in_task"]

    # Find block-day
    block_day = block_learning_df[
        (block_learning_df["animal"] == animal) &
        (block_learning_df["session"] == session) &
        (block_learning_df["block_required"] == blk_req)
    ]["block_day"].iloc[0]

    performance_records.append({
        "animal": animal,
        "block_required": blk_req,
        "session": session,
        "block_day": block_day,
        "performance": performance
    })

performance_df = pd.DataFrame(performance_records)

# ---- Per-animal means ----
perf_animal_mean = (
    performance_df
    .groupby(["animal", "block_required", "block_day"])["performance"]
    .mean()
    .reset_index()
)

# ---------- PLOTTING ----------
for blk in sorted(performance_df["block_required"].unique()):

    df_blk = perf_animal_mean[perf_animal_mean["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    all_days = sorted(df_blk["block_day"].unique())

    # === Vertical dashed lines per training day ===
    for d in all_days:
        plt.axvline(
            x=d,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.4,
            zorder=0
        )

    # === Horizontal threshold line at 75% ===
    plt.axhline(
        y=75,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        zorder=0
    )

    # === Per-animal line + dots ===
    for animal in sorted(df_blk["animal"].unique()):
        df_a = df_blk[df_blk["animal"] == animal].sort_values("block_day")

        # Transparent line
        plt.plot(
            df_a["block_day"],
            df_a["performance"],
            "-",
            color=animal_colors[animal],
            alpha=0.40,
            linewidth=2
        )

        # Solid dots
        plt.scatter(
            df_a["block_day"],
            df_a["performance"],
            color=animal_colors[animal],
            edgecolor="white",
            linewidth=0.6,
            s=70,
            zorder=5
        )

    # Formatting
    plt.title(f"{blk}-Trial Blocks — Performance per Animal")
    plt.xlabel("Training Day (reset per block type)")
    plt.ylabel("Performance (%)")

    plt.xticks(all_days, [f"Day {d}" for d in all_days])
    plt.ylim(0, 100)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [
        plt.Line2D([0], [0],
                   marker="o", color=animal_colors[a],
                   markerfacecolor=animal_colors[a],
                   markersize=8, linewidth=2,
                   label=a)
        for a in sorted(df_blk["animal"].unique())
    ]
    plt.legend(handles=handles, title="Animal", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_Performance_PerAnimal.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_Performance_PerAnimal.svg", dpi=500)

#%% Plot 9 — Triangles with safe jitter and fixed x-limits
print("\n=== PLOT 9 SAFE VERSION ===")

session_counts = []

df_blocks = all_files[all_files["task"] == "two_choice_blocks"]

for animal in animals_of_interest:

    df_a = df_blocks[df_blocks["animal"] == animal]

    for blk in sorted(df_a["block_count"].dropna().unique()):

        n_sessions = df_a[df_a["block_count"] == blk]["day_in_task"].nunique()

        print(f"Animal {animal}, block {blk}, sessions = {n_sessions}")

        session_counts.append({
            "animal": animal,
            "block_required": blk,
            "n_sessions": n_sessions
        })

sessions_df = pd.DataFrame(session_counts)


# ---------- PLOT ----------
plt.figure(figsize=(8,6), dpi=500)

# Desired order on the axis
desired_order = [10, 5, 3]

# Keep block types that appear in data *in this exact order*
block_types = [blk for blk in desired_order 
               if blk in sessions_df["block_required"].unique()]

# Assign x positions according to desired order
x_positions = np.arange(len(block_types))
blk_to_x = {blk: x for x, blk in enumerate(block_types)}

JITTER = 0.2  # smaller jitter


# ----- Triangles -----
for _, row in sessions_df.iterrows():

    x_base = blk_to_x[row["block_required"]]
    jitter = (np.random.rand() - 0.5) * 2 * JITTER

    plt.scatter(
        x_base + jitter,
        row["n_sessions"],
        marker="^",
        s=200,
        color=animal_colors[row["animal"]],
        edgecolor="white",
        linewidth=0.9,
        zorder=3
    )


# Explicit x-limit padding so jitter never hides points
plt.xlim(-0.5, len(block_types) - 0.5)
plt.ylim(0,13)

plt.xticks(x_positions, [f"Blocks of {blk}" for blk in block_types])
plt.ylabel("Number of Sessions")
plt.title("Sessions per block type")

# Clean axis
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# Legend
legend_handles = [
    plt.Line2D([0],[0],
               marker="^",
               color=col,
               markerfacecolor=col,
               markersize=10,
               linewidth=0,
               label=animal)
    for animal, col in animal_colors.items()
]
plt.legend(handles=legend_handles, title="Animal", frameon=False, ncol=2)

plt.tight_layout()
plt.savefig(f"{OUT}/SessionsPerBlock_Triangles_NoAvg_SAFE.png", dpi=500)
plt.savefig(f"{OUT}/SessionsPerBlock_Triangles_NoAvg_SAFE.svg", dpi=500)

#%%  Plot 3 (Updated)
# ======================================================
# INDIVIDUAL ANIMALS ONLY — NO GROUP MEAN
# One color per animal, dots connected
# ======================================================

print("\n=== PLOT 3 (Updated): individual animals connected ===")

# ---- Extract block learning metrics (unchanged) ----
def extract_block_learning_metrics(csv_path, block_size, animal, session, strain):

    df = pd.read_csv(csv_path).copy()

    # Identify block identity
    df["stim_pair"] = list(zip(df["8KHz"], df["16KHz"]))

    # Find flip points
    block_starts = [0]
    for i in range(1, len(df)):
        if df["stim_pair"][i] != df["stim_pair"][i-1]:
            block_starts.append(i)
    block_starts.append(len(df))

    records = []

    for b in range(len(block_starts)-1):
        start = block_starts[b]
        end = block_starts[b+1]
        block_df = df.iloc[start:end]

        total_trials = len(block_df)

        records.append({
            "animal": animal,
            "strain": strain,
            "session": session,
            "block_index": b+1,
            "block_required": block_size,
            "total_trials": total_trials,
        })

    return records


# ---- Collect all sessions for two-choice blocks ----
block_learning_records = []

for _, row in all_files[all_files["task"] == "two_choice_blocks"].iterrows():
    block_learning_records.extend(
        extract_block_learning_metrics(
            row["path"],
            row["block_count"],
            row["animal"],
            row["day_in_task"],
            row["strain"]
        )
    )

block_learning_df = pd.DataFrame(block_learning_records)
print(block_learning_df.head())

# ---- Compute block_day (restart day numbering per block type per animal) ----
block_learning_df["block_day"] = (
    block_learning_df
    .groupby(["animal", "block_required"])["session"]
    .transform(lambda x: pd.factorize(x)[0] + 1)
)

# ---- Compute per animal per block_day mean (1 point per day) ----
animal_day_means = (
    block_learning_df
    .groupby(["animal", "strain", "block_required", "block_day"])["total_trials"]
    .mean()
    .reset_index()
)

print("\n=== ANIMAL DAY MEANS ===")
print(animal_day_means)


# ======================================================
#            PLOTTING — ONE FIGURE PER BLOCK TYPE
# ======================================================

# Custom Y-limits per block
block_ylim = {
    10: (0, 250),   # <-- change these!
    5:  (0, 10),
    3:  (0, 6)
}


for blk in sorted(animal_day_means["block_required"].dropna().unique()):

    df_blk = animal_day_means[animal_day_means["block_required"] == blk]

    plt.figure(figsize=(9,6), dpi=500)

    # ---- ONE LINE PER ANIMAL ----
    for animal in df_blk["animal"].unique():

        df_a = df_blk[df_blk["animal"] == animal].sort_values("block_day")

        x = df_a["block_day"]
        y = df_a["total_trials"]

        # Connect with line (transparent)
        plt.plot(
            x, y,
            color=animal_colors.get(animal, "black"),
            linewidth=2,
            alpha=0.4,
            zorder=2
        )

        # Plot dots with jitter
        for _, row in df_a.iterrows():
            jitter = (np.random.rand() - 0.5) * 0.15
            plt.scatter(
                row["block_day"] + jitter,
                row["total_trials"],
                s=70,
                color=animal_colors.get(animal, "black"),
                edgecolor="white",
                linewidth=0.6,
                zorder=3
            )

    # ---- Formatting ----
    plt.title(f"Block of {blk} trials — Trials to Block Switch")
    plt.ylabel("Trials until Stimulus Switch")

    xticks = sorted(df_blk["block_day"].unique())
    plt.xticks(xticks, [f"Day {i}" for i in xticks])
    
    if blk in block_ylim:
        plt.ylim(block_ylim[blk])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Legend: one color per animal ----
    legend_handles = [
        plt.Line2D([0], [0],
                   marker="o",
                   linestyle="",
                   markerfacecolor=animal_colors[a],
                   markeredgecolor="white",
                   markersize=9,
                   label=a)
        for a in df_blk["animal"].unique()
    ]

    plt.legend(handles=legend_handles, title="Animal", frameon=False)

    plt.tight_layout()
    plt.savefig(f"{OUT}/Block_{blk}_TrialsPerDay_IndividualsOnly.png", dpi=500)
    plt.savefig(f"{OUT}/Block_{blk}_TrialsPerDay_IndividualsOnly.svg", dpi=500)
