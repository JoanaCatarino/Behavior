# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:10:06 2025

@author: JoanaCatarino
"""

import os
import math
import re
import glob
from pathlib import Path
from datetime import datetime
from math import isnan

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

# ==== USER SETTINGS ==========================================
base_dir = r"L:\dmclab\Joana\Behavior\Data"   
animals_of_interest = ['956700']  
animal = 956700               
recursive_search = True
save_formats = ("png", "pdf", "svg")
patterns = ["AdaptSensorimotor*.csv"]
DPI = 500
# =============================================================

# Example filename:
# 2ChoiceAuditory_956700_20240910_103000_something.csv
fname_rx = re.compile(
    r"^(?:2ChoiceAuditory|2ChoiceBlocks)_(?P<animal>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_.*\.csv$",
    re.IGNORECASE,
)

# Try to detect "Box 3" or "Box3" in folder names
box_rx = re.compile(r"box[\s_-]?(\d+)", re.IGNORECASE)


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def find_files() -> pd.DataFrame:
    files = []
    for pat in patterns:
        if recursive_search:
            files.extend(glob.glob(str(Path(base_dir) / "**" / pat), recursive=True))
        else:
            files.extend(glob.glob(str(Path(base_dir) / pat), recursive=False))

    records = []
    for f in files:
        p = Path(f)
        name = p.name
        m = fname_rx.match(name)
        if not m:
            continue
        animal = m.group("animal")
        if animals_of_interest and animal not in animals_of_interest:
            continue

        date, time = m.group("date"), m.group("time")
        dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
        records.append({
            "animal": animal,
            "dt": dt,
            "path": str(p),
        })

    if not records:
        raise FileNotFoundError("No matching CSVs found for selected animals.")

    df = pd.DataFrame(records).sort_values(["animal", "dt"]).reset_index(drop=True)
    return df


def extract_metadata(file_path: str) -> tuple[datetime, str | None]:
    """Returns (datetime, box_str)."""
    p = Path(file_path)
    m = fname_rx.match(p.name)
    if m:
        dt = datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    else:
        dt = None

    # walk parents to find something like “Box 3” or “Box3”
    box = None
    for parent in p.parents:
        mm = box_rx.search(parent.name)
        if mm:
            box = mm.group(1)
            break
    return dt, box


def load_tone_mapping(animal_id: str) -> str:
    mapping_file_path = Path(r"L:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv")
    if not mapping_file_path.exists():
        return "Tone-spout mapping: (mapping file not found)"

    spout_mapping_df = pd.read_csv(mapping_file_path)
    # robust cast to int if possible
    try:
        row = spout_mapping_df[spout_mapping_df["Animal"] == int(animal_id)].iloc[0]
        pair_8khz = f"8KHz → {row['8KHz']} spout"
        pair_16khz = f"16KHz → {row['16KHz']} spout"
        return f"Tone-spout mapping: {pair_8khz}, {pair_16khz}"
    except Exception:
        return "Tone-spout mapping: (not found for this animal)"
    

def load_trial_counts(file_path:str) -> dir:
    
    df = pd.read_csv(file_path).fillna(0)
    
    # Compute lick latency
    df["RW_start"] = df["trial_start"] + 1.4 # 1 second of waiting window + 400ms of sound
    df["lick_latency"] = df["lick_time"] - df["RW_start"]

    valid_licks = df[df["lick_latency"].notna()]
    licks_8KHz = valid_licks[valid_licks["8KHz"] == 1]
    licks_16KHz = valid_licks[valid_licks["16KHz"] == 1]

    total_trials = len(df)
    correct_8KHz = df[(df["8KHz"] == 1) & (df["reward"] == 1)].shape[0]
    correct_16KHz = df[(df["16KHz"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect_8KHz = df[(df["8KHz"] == 1) & (df["punishment"] == 1)].shape[0]
    incorrect_16KHz = df[(df["16KHz"] == 1) & (df["punishment"] == 1)].shape[0]
    early = df[df["early_lick"] == 1].shape[0] if "early_lick" in df.columns else 0
    omissions = df[df['omission'] == 1].shape[0]
    omission_8KHz = df[(df["omission"] == 1) & (df["8KHz"] == 1)].shape[0]
    omission_16KHz = df[(df["omission"] == 1) & (df["16KHz"] == 1)].shape[0]
    dprime = df["d_prime"].iloc[0] if "d_prime" in df.columns else None
    hit_rate = df["hit_rate"].iloc[0] if "hit_rate" in df.columns else None
    false_alarm = df["false_alarm"].iloc[0] if "false_alarm" in df.columns else None
    latency_8KHz = licks_8KHz["lick_latency"].mean() if not licks_8KHz.empty else None
    latency_16KHz = licks_16KHz["lick_latency"].mean() if not licks_16KHz.empty else None
    latency_general = latency_8KHz + latency_16KHz
    latency_8KHz_std = licks_8KHz["lick_latency"].std() if not licks_8KHz.empty else None
    latency_16KHz_std = licks_16KHz["lick_latency"].std() if not licks_16KHz.empty else None
    latency_general_std = latency_8KHz_std + latency_16KHz_std
    

    # Compute hit rate, false alarm, d'
    correct = correct_8KHz + correct_16KHz
    incorrect = incorrect_8KHz + incorrect_16KHz
    total = len(df)
    
    hr = (correct + 0.5) / (total + 1)
    fa = (incorrect + 0.5) / (total + 1)
    hr = min(max(hr, 0.01), 0.99)
    fa = min(max(fa, 0.01), 0.99)
    dprime = norm.ppf(hr) - norm.ppf(fa)
    
    
    # Compute overall Performance
    performance = (correct / (correct + incorrect + omissions))*100
    
    # Compute performance in 8KHz and 16KHz trials
    performance_8KHz = (correct_8KHz / (correct_8KHz + incorrect_8KHz + omission_8KHz))*100
    performance_16KHz = (correct_16KHz / (correct_16KHz + incorrect_16KHz + omission_16KHz))*100
    
    # Compute overall %correct (excludes omissions)
    percent_correct = (correct / (correct + incorrect))*100
    
    # Compute %correct in 8KHz and 16KHz trials
    percent_correct_8KHz = (correct_8KHz / (correct_8KHz + incorrect_8KHz))*100
    percent_correct_16KHz = (correct_16KHz / (correct_16KHz + incorrect_16KHz))*100
    

    if "QW" in df.columns and not df["QW"].isna().all():
        qw_value = df["QW"].mode()[0]
    else:
        qw_value = "NA"

    autom_reward_dominant = False
    if "autom_reward" in df.columns:
        autom_reward_dominant = (df["autom_reward"] == 1).sum() > len(df) / 2

    return dict(
        total_trials = total_trials,
        correct_8KHz = correct_8KHz,
        correct_16KHz = correct_16KHz,
        incorrect_8KHz = incorrect_8KHz,
        incorrect_16KHz = incorrect_16KHz,
        early_licks = early,
        omissions = omissions,
        omission_8KHz = omission_8KHz,
        omission_16KHz = omission_16KHz,
        dprime = dprime,
        hit_rate = hit_rate,
        false_alarm = false_alarm,
        latency_general = latency_general,
        latency_8KHz = latency_8KHz,
        latency_16KHz = latency_16KHz,
        latency_8KHz_std = latency_8KHz_std,
        latency_16KHz_std = latency_16KHz_std,
        latency_general_std = latency_general_std,
        QW=qw_value,
        autom_reward=autom_reward_dominant,
        performance = performance,
        performance_8KHz = performance_8KHz,
        performance_16KHz = performance_16KHz,
        percent_correct = percent_correct,
        percent_correct_8KHz = percent_correct_8KHz,
        percent_correct_16KHz = percent_correct_16KHz
    )



def annotate_bars(ax, bars, offset=1, fmt="{:.1f}%"):
    """Put percentage labels on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_across_days(animal: str, session_summaries: list[dict]) -> None:
    if not session_summaries:
        print(f"No valid data to plot for animal {animal}.")
        return

    # sort by date
    session_summaries.sort(key=lambda x: x["date"])

    df = pd.DataFrame(session_summaries)
    
    
    # ===== Session → real date mapping (add this) ============================
    # Protocol from filename prefix (e.g., 2ChoiceAuditory / 2ChoiceBlocks)
    proto = df["file"].map(lambda p: os.path.basename(p).split("_")[0] if isinstance(p, str) else None)
    base  = df["file"].map(lambda p: os.path.basename(p) if isinstance(p, str) else None)

    session_table = pd.DataFrame({
        "Session": np.arange(1, len(df) + 1, dtype=int),
        "Date":    pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "Time":    pd.to_datetime(df["date"]).dt.strftime("%H:%M:%S"),
        "Protocol": proto,
        "Box":      df.get("box"),
        "Filename": base,
    })

    # Show a small view in the console
    print("\nSession → real date mapping:")
    print(session_table[["Session", "Date", "Time"]].to_string(index=False))

    tone_mapping_str = load_tone_mapping(animal)


    # Figure 1: total trials across sessions ---------------------------------------------
    # x = session numbers (start at 1), y = total trials per session
    x = list(range(1, len(df) + 1))
    y = y = df["total_trials"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi = DPI)
    ax.scatter(x, y, s=30, color="#876EA6", zorder=3)

    # add dashed stems from each dot to x-axis
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [0, yi], linestyle="--", color="gray", linewidth=1, zorder=2)

    day_labels = [f"{i}" for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=0)
    ax.set_xlabel("Sessions")
    ax.set_ylabel("Total trials")
    ax.set_title(f"Total Trials per Session — Animal {animal}", pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.tight_layout()

def run_for_animals(animal_ids: list[str]) -> None:
    all_files = find_files()
    for animal in animal_ids:
        animal_df = all_files[all_files["animal"] == animal]
        if animal_df.empty:
            print(f"⚠️ No files found for animal {animal}")
            continue

        summaries = []
        for _, row in animal_df.iterrows():
            file_path = row["path"]
            try:
                date, box = extract_metadata(file_path)
                tdat = load_trial_counts(file_path)
                tdat.update({"date": date, "box": box, "file": file_path})
                summaries.append(tdat)
            except Exception as e:
                print(f"⚠️ Skipping file due to error: {file_path}\n{e}")

        plot_across_days(animal, summaries)


def cli():
    parser = argparse.ArgumentParser(description="Across-days plots for 2-Choice tasks.")
    parser.add_argument("--animals", nargs="*", default=animals_of_interest,
                        help="Animal IDs to process (default: from script).")
    args = parser.parse_args()
    run_for_animals(args.animals)


if __name__ == "__main__":
    cli()
