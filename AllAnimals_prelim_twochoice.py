# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 02:49:04 2025

@author: JoanaCatarino
"""

import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# ============================================================
# ========== User settings ===================================
# ============================================================

base_dir = r"L:/dmclab/Joana/Behavior/Data"

animals_of_interest = ["986235", "999770", "986167", "986168", "986169", "986170", "986171"]

animal_strain = { 
    "986235" : "Tlx3", 
    "999770" : "Tlx3", 
    "986167" : "Fezf2", 
    "986168" : "Fezf2", 
    "986169" : "Fezf2", 
    "986170" : "Fezf2", 
    "986171" : "Fezf2"
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

OUT = r"L:/dmclab/Joana/Behavior/Data/plots/behavior_cohort_1/two_choice"


# ============================================================
# Helpers
# ============================================================

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


# ============================================================
# Find files
# ============================================================

def find_files_folder_structure(base_dir: str, animals: list[str]) -> pd.DataFrame:

    records = []
    for animal in animals:
        behavior_dir = Path(base_dir) / animal / "Behavior"
        if not behavior_dir.exists():
            continue

        for day in sorted([d for d in behavior_dir.iterdir() if d.is_dir()]):
            csvs = list(day.glob("*.csv"))
            if not csvs:
                continue

            filename = csvs[0].name
            task = detect_task_from_filename(filename)
            dt   = extract_datetime_from_filename(filename)

            records.append({
                "animal": animal,
                "task": task,
                "dt": dt,
                "path": str(csvs[0]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["animal", "dt"]).reset_index(drop=True)
    return df


# ============================================================
# Run Importer
# ============================================================

all_files = find_files_folder_structure(base_dir, animals_of_interest)

# Add day in task
all_files["day_in_task"] = (
    all_files.groupby(["animal", "task"]).cumcount() + 1
)

# Add strain
all_files["strain"] = all_files["animal"].map(animal_strain)

print("\n=== IMPORT COMPLETED ===")
print(all_files)


# ============================================================
# ========== Filter ONLY two-choice auditory =================
# ============================================================

tc_files = all_files[all_files["task"] == "two_choice"].copy()

print("\n=== TWO-CHOICE SESSIONS ONLY ===")
print(tc_files)


# ============================================================
# ========== Compute session-level performance ===============
# ============================================================

session_records = []

for _, row in tc_files.iterrows():

    df = pd.read_csv(row["path"])

    required = ["reward", "punishment"]
    if not all(c in df.columns for c in required):
        continue

    has_om = "omission" in df.columns
    has_ct = "catch_trial" in df.columns

    total_trials = len(df)
    n_correct = (df["reward"] == 1).sum()
    n_incorrect = (df["punishment"] == 1).sum()

    if has_om and has_ct:
        n_omissions = ((df["omission"] == 1) & (df["catch_trial"] == 0)).sum()
    elif has_om:
        n_omissions = (df["omission"] == 1).sum()
    else:
        n_omissions = 0

    denom = n_correct + n_incorrect + n_omissions
    performance = n_correct / denom if denom > 0 else np.nan

    session_records.append({
        "animal": row["animal"],
        "strain": row["strain"],
        "day_in_task": row["day_in_task"],
        "date": row["dt"].strftime("%Y%m%d") if row["dt"] is not None else None,
        "path": row["path"],
        "total_trials": total_trials,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_omissions": n_omissions,
        "performance": performance,
    })

tc_summary = pd.DataFrame(session_records)

print("\n=== TWO-CHOICE PERFORMANCE SUMMARY ===")
print(tc_summary)



#%% =======================================================
# TONE-SPECIFIC PERFORMANCE
# =======================================================

tone_records = []

for _, row in tc_files.iterrows():

    df = pd.read_csv(row["path"])

    # identify 8kHz vs 16kHz trials
    df["stim"] = df.apply(
        lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else None),
        axis=1
    )

    # compute per-tone performance
    for tone in ["8KHz", "16KHz"]:

        dft = df[df["stim"] == tone]

        if len(dft) == 0:
            continue

        n_correct = (dft["reward"] == 1).sum()
        n_incorrect = (dft["punishment"] == 1).sum()

        if "omission" in df.columns and "catch_trial" in df.columns:
            n_om = ((dft["omission"] == 1) & (dft["catch_trial"] == 0)).sum()
        elif "omission" in df.columns:
            n_om = (dft["omission"] == 1).sum()
        else:
            n_om = 0

        denom = n_correct + n_incorrect + n_om
        perf = n_correct / denom if denom > 0 else np.nan

        tone_records.append({
            "animal": row["animal"],
            "session": row["dt"].strftime("%Y%m%d"),
            "day_in_task": row["day_in_task"],
            "tone": tone,
            "performance": perf
        })

tone_df = pd.DataFrame(tone_records)
print("\n=== TONE PERFORMANCE ===")
print(tone_df.head())


# =======================================================
# CLEAN MULTI-PANEL TONE PERFORMANCE PLOT
# One subplot per animal
# =======================================================

animals = tone_df["animal"].unique()
n_animals = len(animals)

fig, axes = plt.subplots(n_animals, 1, figsize=(8, 3*n_animals), dpi=500, sharex=True)

if n_animals == 1:
    axes = [axes]

for ax, animal in zip(axes, animals):

    df_a = tone_df[tone_df["animal"] == animal].sort_values("day_in_task")

    # 8kHz
    df_8 = df_a[df_a["tone"] == "8KHz"]
    ax.plot(df_8["day_in_task"], df_8["performance"],
            marker="o", markersize=6, linewidth=2,
            label="8 kHz", color="#5B9E9A")

    # 16kHz
    df_16 = df_a[df_a["tone"] == "16KHz"]
    ax.plot(df_16["day_in_task"], df_16["performance"],
            marker="o", markersize=6, linewidth=2,
            label="16 kHz", color="#815D9F")

    ax.axhline(0.5, linestyle="--", color="gray")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Animal {animal}")
    ax.set_ylabel("Performance")
    sns.despine()

    ax.legend(frameon=False)

plt.xlabel("Day in Two-Choice Task")
plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_TonePerformance_ByAnimal.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_TonePerformance_ByAnimal.svg", dpi=600)

plt.show()


#%% =======================================================
# GROUP AVERAGE TONE PERFORMANCE (VERY CLEAN)
# =======================================================

mean_df = (
    tone_df.groupby(["day_in_task", "tone"])["performance"]
           .mean()
           .reset_index()
)

plt.figure(figsize=(8,5), dpi=500)

for tone, df_t in mean_df.groupby("tone"):
    plt.plot(
        df_t["day_in_task"],
        df_t["performance"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=tone
    )


plt.ylim(0,1.05)
plt.xlabel("Day in Two-Choice Task")
plt.ylabel("Performance")
plt.title("Mean Tone-Specific Performance Across Animals")
plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_TonePerformance_GroupMean.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_TonePerformance_GroupMean.svg", dpi=600)

#%% ==========================================================
# ==========================================================
# TRIANGLE PLOT: Mean Performance per Animal
#     + black filled triangle = group mean ± SEM
# ==========================================================


mean_perf_animal = (
    tone_df.groupby("animal")["performance"]
           .mean()
           .reset_index()
)

# convert to percent
mean_perf_animal["performance_pct"] = mean_perf_animal["performance"] * 100

animals_sorted = mean_perf_animal["animal"].unique()
x_positions = np.arange(len(animals_sorted))

plt.figure(figsize=(10, 5), dpi=500)

# --- individual animals (open triangles) ---
for i, row in mean_perf_animal.iterrows():
    animal = row["animal"]
    perf = row["performance_pct"]
    color = animal_colors.get(animal, "black")

    plt.scatter(
        x_positions[i],
        perf,
        marker="^",
        s=150,
        facecolors="none",
        edgecolors=color,
        linewidths=2
    )

# --- GROUP MEAN & SEM ---
group_mean = mean_perf_animal["performance_pct"].mean()
group_sem  = mean_perf_animal["performance_pct"].sem()

plt.errorbar(
    [np.mean(x_positions)],
    [group_mean],
    yerr=[group_sem],
    fmt="^",
    markersize=14,
    color="black",
    capsize=5,
    label="Group mean ± SEM"
)

plt.axhline(75, linestyle="--", color="gray")   # 50% chance line

plt.xticks(x_positions, animals_sorted, fontsize=10)
plt.ylabel("Mean Performance (%)")
plt.ylim(0, 105)
plt.title("Mean Performance per Animal — With Group Mean ± SEM", fontsize=14)

sns.despine()
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_Triangle_MeanPerformancePerAnimal_pct.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_Triangle_MeanPerformancePerAnimal_pct.svg", dpi=600)


#%% ==========================================================
# TRIANGLE PLOT: Mean Performance per Tone per Animal (%)
#      + black filled triangles = group mean ± SEM
# ==========================================================

mean_tone = (
    tone_df.groupby(["animal", "tone"])["performance"]
           .mean()
           .reset_index()
)

# convert to percent
mean_tone["performance_pct"] = mean_tone["performance"] * 100

tones = ["8KHz", "16KHz"]
colors = {"8KHz": "#1f77b4", "16KHz": "#ff7f0e"}

animals_sorted = mean_tone["animal"].unique()
x_positions = np.arange(len(animals_sorted))

plt.figure(figsize=(12, 5), dpi=500)

# --- Individual animal values (open triangles) ---
for tone in tones:
    df_t = mean_tone[mean_tone["tone"] == tone]

    for i, animal in enumerate(animals_sorted):
        vals = df_t[df_t["animal"] == animal]["performance_pct"]
        if len(vals) == 0:
            continue

        plt.scatter(
            x_positions[i] + (0.1 if tone == "16KHz" else -0.1),
            vals.iloc[0],
            marker="^",
            s=150,
            facecolors="none",
            edgecolors=colors[tone],
            linewidths=2,
            label=tone if i == 0 else None
        )

# --- GROUP MEAN ± SEM per tone ---
for j, tone in enumerate(tones):
    df_t = mean_tone[mean_tone["tone"] == tone]["performance_pct"]

    gmean = df_t.mean()
    gsem  = df_t.sem()

    xpos = (len(animals_sorted) - 1) + 0.4 + j*0.3  # place next to animals

    plt.errorbar(
        xpos,
        gmean,
        yerr=gsem,
        fmt="^",
        markersize=14,
        color="black",
        capsize=5,
        label=f"{tone} mean ± SEM"
    )

# Formatting
plt.axhline(50, linestyle="--", color="gray")   # 50% chance line

plt.xticks(
    list(x_positions) + [len(animals_sorted)+0.4, len(animals_sorted)+0.7],
    list(animals_sorted) + ["8kHz Mean", "16kHz Mean"],
    rotation=45
)

plt.ylabel("Performance (%)")
plt.ylim(0, 105)
plt.title("Mean Tone-Specific Performance per Animal — With Group Means ± SEM", fontsize=14)

plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_Triangle_MeanPerformancePerTone_pct.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_Triangle_MeanPerformancePerTone_pct.svg", dpi=600)

#%% ============================================================
#                TWO-CHOICE ANALYSIS PIPELINE
# ============================================================

import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# USER SETTINGS
# ============================================================

base_dir = r"L:/dmclab/Joana/Behavior/Data"

animals_of_interest = ["986235", "999770", "986167", "986168", "986169", "986170", "986171"]

animal_colors = {
    "986235": "#ff7f0e",
    "999770": "#2ca02c",
    "986167": "#d62728",
    "986168": "#9467bd",
    "986169": "#8c564b",
    "986170": "#e377c2",
    "986171": "#7f7f7f",
}

OUT = r"L:/dmclab/Joana/Behavior/Data/plots/behavior_cohort_1/two_choice"


# ============================================================
# HELPERS — DETECT TASK, LOAD FILES
# ============================================================

def detect_task_from_filename(filename: str) -> str:
    if filename.startswith("2ChoiceAuditory"):
        return "two_choice"
    elif filename.startswith("FreeLick"):
        return "free_licking"
    elif filename.startswith("2ChoiceBlocks"):
        return "two_choice_blocks"
    elif filename.startswith("AdaptSensorimotor"):
        return "adaptive_sensorimotor"
    return "unknown"


def extract_datetime_from_filename(filename: str):
    """
    Extract timestamp from ..._<YYYYMMDD>_<HHMMSS>_...
    """
    m = re.search(r"_(\d{8})_(\d{6})_", filename)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def find_files_folder_structure(base_dir: str, animals: list[str]) -> pd.DataFrame:
    records = []

    for animal in animals:
        beh_dir = Path(base_dir) / animal / "Behavior"
        if not beh_dir.exists():
            continue

        for day in sorted(d for d in beh_dir.iterdir() if d.is_dir()):
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

    return pd.DataFrame(records)


# ============================================================
# LOAD ALL FILES
# ============================================================

all_files = find_files_folder_structure(base_dir, animals_of_interest)
all_files = all_files.sort_values(["animal", "dt"]).reset_index(drop=True)

print("\n=== ALL FILES FOUND ===")
print(all_files)


# ============================================================
# EXTRACT TWO-CHOICE ONLY
# ============================================================

twochoice_files = all_files[all_files["task"] == "two_choice"].copy()

print(f"\nFound {len(twochoice_files)} two-choice sessions.")


# ============================================================
# COMPUTE OMISSIONS PER TONE
# ============================================================

omission_records = []

for _, row in twochoice_files.iterrows():
    df = pd.read_csv(row["path"])

    if not {"omission", "8KHz", "16KHz"}.issubset(df.columns):
        continue

    fname = Path(row["path"]).name
    date_str = fname.split("_")[2]

    omission_8  = len(df[(df["omission"] == 1) & (df["8KHz"] == 1)])
    omission_16 = len(df[(df["omission"] == 1) & (df["16KHz"] == 1)])

    omission_records.append({
        "animal": row["animal"],
        "session": date_str,
        "om_8": omission_8,
        "om_16": omission_16
    })

omit_df = pd.DataFrame(omission_records)

print("\n=== OMISSIONS PER TONE ===")
print(omit_df)



# ============================================================
# COMPUTE LICK LATENCY PER TONE
# ============================================================

lat_records = []

for _, row in twochoice_files.iterrows():
    df = pd.read_csv(row["path"])

    if not {"lick_time", "RW_start", "8KHz", "16KHz", "omission"}.issubset(df.columns):
        continue

    df["latency"] = df["lick_time"] - df["RW_start"]

    fname = Path(row["path"]).name
    date_str = fname.split("_")[2]

    # Only non-omission trials
    lat_8  = df[(df["8KHz"] == 1) & (df["omission"] == 0)]["latency"].mean()
    lat_16 = df[(df["16KHz"] == 1) & (df["omission"] == 0)]["latency"].mean()

    lat_records.append({
        "animal": row["animal"],
        "session": date_str,
        "lat_8": lat_8,
        "lat_16": lat_16
    })

lat_df = pd.DataFrame(lat_records)
print("\n=== LATENCY PER TONE ===")
print(lat_df)


# ============================================================
# TRIANGLE PLOT — LICK LATENCY PER TONE
# ============================================================

plt.figure(figsize=(8, 5), dpi=500)

x_pos = [0, 1]
labels = ["8 kHz", "16 kHz"]

# session dots
for _, r in lat_df.iterrows():
    jitter = (np.random.rand() - 0.5) * 0.12
    plt.scatter(
        x_pos[0] + jitter, r["lat_8"],
        marker="o", s=90,
        color=animal_colors[r["animal"]],
        edgecolor="white", linewidth=0.7
    )
    plt.scatter(
        x_pos[1] + jitter, r["lat_16"],
        marker="o", s=90,
        color=animal_colors[r["animal"]],
        edgecolor="white", linewidth=0.7
    )

# group mean + SEM
mean_vals = [lat_df["lat_8"].mean(), lat_df["lat_16"].mean()]
sem_vals  = [lat_df["lat_8"].sem(),  lat_df["lat_16"].sem()]



#%%# -------------------------------------------------------------------
# TWO-CHOICE ANALYSIS: Omissions & Lick Latency per Tone
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# INPUT: all_files from your main loader
# animals_of_interest
# animal_colors
# OUT folder
# -----------------------------------------------------------

# Ensure all required columns exist in two-choice files
required_cols = ["8KHz", "16KHz", "reward", "punishment",
                 "omission", "lick_time", "RW_start"]


# -----------------------------------------------------------
# COLLECT DATA ACROSS ALL TWO-CHOICE SESSIONS
# -----------------------------------------------------------

records_omit = []
records_lat  = []

for _, row in all_files[all_files["task"] == "two_choice"].iterrows():

    animal = row["animal"]
    df = pd.read_csv(row["path"])

    if not all(c in df.columns for c in required_cols):
        continue

    date_str = Path(row["path"]).name.split("_")[2]  # YYYYMMDD

    # -------------------------
    # OMISSIONS per tone
    # -------------------------
    om_8  = df[(df["8KHz"] == 1)  & (df["omission"] == 1)].shape[0]
    om_16 = df[(df["16KHz"] == 1) & (df["omission"] == 1)].shape[0]

    records_omit.append({
        "animal": animal,
        "session": date_str,
        "om_8": om_8,
        "om_16": om_16
    })

    # -------------------------
    # LATENCY per tone
    # -------------------------
    df["latency"] = df["lick_time"] - df["RW_start"]

    lat_8  = df[(df["8KHz"] == 1)  & (df["omission"] == 0)]["latency"].mean()
    lat_16 = df[(df["16KHz"] == 1) & (df["omission"] == 0)]["latency"].mean()

    records_lat.append({
        "animal": animal,
        "session": date_str,
        "lat_8": lat_8,
        "lat_16": lat_16
    })


omit_df = pd.DataFrame(records_omit)
lat_df  = pd.DataFrame(records_lat)

print("\n=== Raw omissions ===")
print(omit_df)
print("\n=== Raw latencies ===")
print(lat_df)


# -----------------------------------------------------------
# AVERAGE ACROSS SESSIONS (per ANIMAL)
# -----------------------------------------------------------

omit_animal = (
    omit_df.groupby("animal")
           .agg({"om_8": "mean", "om_16": "mean"})
           .reset_index()
)

lat_animal = (
    lat_df.groupby("animal")
          .agg({"lat_8": "mean", "lat_16": "mean"})
          .reset_index()
)

print("\n=== Omissions per animal ===")
print(omit_animal)
print("\n=== Latency per animal ===")
print(lat_animal)


# -----------------------------------------------------------
# PLOT 1 — OMISSIONS TRIANGLE PLOT
# -----------------------------------------------------------

plt.figure(figsize=(8, 5), dpi=500)

x_pos = [0, 1]
labels = ["8 kHz", "16 kHz"]

for _, r in omit_animal.iterrows():
    animal = r["animal"]
    color = animal_colors.get(animal, "gray")
    jit = (np.random.rand() - 0.5) * 0.12

    # 8 kHz
    plt.scatter(
        x_pos[0] + jit, r["om_8"],
        marker="^", s=140,
        facecolors="none",
        edgecolors=color, linewidths=2
    )

    # 16 kHz
    plt.scatter(
        x_pos[1] + jit, r["om_16"],
        marker="^", s=140,
        facecolors="none",
        edgecolors=color, linewidths=2
    )

# ---- Group mean ± SEM ----
mean_vals = [omit_animal["om_8"].mean(), omit_animal["om_16"].mean()]
sem_vals  = [omit_animal["om_8"].sem(),  omit_animal["om_16"].sem()]

plt.errorbar(
    [x_pos[0] + 0.15, x_pos[1] + 0.15],
    mean_vals,
    yerr=sem_vals,
    fmt="^",
    markersize=12,
    color="black",
    capsize=6,
    label="Group mean ± SEM"
)

plt.xticks(x_pos, labels, fontsize=11)
plt.ylabel("Omissions", fontsize=11)
plt.title("Omissions per Tone — averaged over sessions", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_OmissionsPerTone_ANIMALS.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_OmissionsPerTone_ANIMALS.svg", dpi=600)


# -----------------------------------------------------------
# PLOT 2 — LATENCY TRIANGLE PLOT
# -----------------------------------------------------------

plt.figure(figsize=(8, 5), dpi=500)

for _, r in lat_animal.iterrows():
    animal = r["animal"]
    color = animal_colors.get(animal, "gray")
    jit = (np.random.rand() - 0.5) * 0.12

    plt.scatter(
        x_pos[0] + jit, r["lat_8"],
        marker="^", s=110,
        color=color, edgecolor="white", linewidth=0.7
    )

    plt.scatter(
        x_pos[1] + jit, r["lat_16"],
        marker="^", s=110,
        color=color, edgecolor="white", linewidth=0.7
    )

# ---- Group mean ± SEM ----
mean_vals = [lat_animal["lat_8"].mean(), lat_animal["lat_16"].mean()]
sem_vals  = [lat_animal["lat_8"].sem(),  lat_animal["lat_16"].sem()]

plt.errorbar(
    [x_pos[0] + 0.15, x_pos[1] + 0.15],
    mean_vals,
    yerr=sem_vals,
    fmt="^",
    markersize=12,
    color="black",
    capsize=6,
    label="Group mean ± SEM"
)

plt.ylim(0, 0.5)
plt.xticks(x_pos, labels, fontsize=11)
plt.ylabel("Lick latency (s)", fontsize=11)
plt.title("Lick Latency per Tone — averaged over sessions", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.3)
sns.despine()
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT}/TwoChoice_LatencyPerTone_ANIMALS.png", dpi=600)
plt.savefig(f"{OUT}/TwoChoice_LatencyPerTone_ANIMALS.svg", dpi=600)
