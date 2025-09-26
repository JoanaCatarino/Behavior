# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 18:59:01 2025

@author: JoanaCatarino

overall plots across sessions for each animal 
"""

import os, re
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
animals_of_interest = ["956700"]                  
recursive_search = True
save_formats = ("png", "pdf", "svg")
patterns = ["2ChoiceAuditory*.csv", "2ChoiceBlocks*.csv"]
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
    df["tone_time"] = df["trial_start"] + 1.2
    df["lick_latency"] = df["lick_time"] - df["tone_time"]

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
    latency_8KHz_std = licks_8KHz["lick_latency"].std() if not licks_8KHz.empty else None
    latency_16KHz_std = licks_16KHz["lick_latency"].std() if not licks_16KHz.empty else None
    

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
    
    # Compute %correct (excludes omissions)
    

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
        early = early,
        omissions = omissions,
        omission_8KHz = omission_8KHz,
        omission_16KHz = omission_16KHz,
        dprime = dprime,
        hit_rate = hit_rate,
        false_alarm = false_alarm,
        latency_8KHz = latency_8KHz,
        latency_16KHz = latency_16KHz,
        latency_8KHz_std = latency_8KHz_std,
        latency_16KHz_std = latency_16KHz_std,
        QW=qw_value,
        autom_reward=autom_reward_dominant,
        performance = performance,
        performance_8KHz = performance_8KHz,
        performance_16KHz = performance_16KHz
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
    dates = [s["date"] for s in session_summaries]
    x = list(range(len(dates)))
    day_labels = [f"Day {i+1}" for i in x]
    boxes = [s.get("box") for s in session_summaries]

    # Colors per QW and autom_reward override
    qw_colors = {
        0: "#FFFFFF",
        1: "#FAD4D4",
        2: "#FCE5CD",
        3: "#FFF2CC",
        'NA': "#F5F5F5"
    }

    tone_mapping_str = load_tone_mapping(animal)


    # === FIGURE 1: total trials dot plot with stems ===
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
    plt.tight_layout()

    
    # Figure 2 - Performance across sessions
    x = list(range(1, len(df) + 1))
    y1 = df["performance"].tolist()
    y2 = df["performance_8KHz"].tolist()
    y3 = df["performance_16KHz"].tolist()
    day_labels = [f"{i}" for i in x]
    bw = 0.6
    spacing = 2
    x_spaced = [xi *spacing for xi in x]
    x2 = [xi - bw/2 for xi in x_spaced]
    x3 = [xi + bw/2 for xi in x_spaced]
    

    fig, (ax1, ax2)= plt.subplots(2,1, figsize=(20,14), sharex=False, constrained_layout=False, gridspec_kw={'hspace': 0.5}, dpi=DPI)
    ax1.plot(x, y1, linewidth=1, color='gray') 
    ax1.scatter(x, y1, s=30, color="#876EA6", zorder=3)

    
    ax1.set_xticks(x)
    ax1.set_ylim(0,100)
    ax1.set_xticklabels(day_labels, rotation=0)
    ax1.set_xlabel("Sessions")
    ax1.set_ylabel("Performance (%)")
    ax1.set_title(f"Performance per Session — Animal {animal}", pad=20)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    
    
    # Bottom: performance in 8KHz trials versus 16KHz trials
    bars1a = ax2.bar(x2, y2, width=bw, color="#ADD3D1", label="8Khz")
    bars1b = ax2.bar(x3, y3, width=bw, color="#8E7FAD", label="16KHz")

    ax2.set_xticks(x_spaced) 
    ax2.set_ylim(0,100)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title(f"Performance per Session — Animal {animal}", pad=20)
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticklabels(day_labels, rotation=0)  # show 'Day N' labels on top plot too
    pad = spacing * 0.7  # optional padding
    ax2.set_xlim(x_spaced[0] - pad, x_spaced[-1] + pad)
    annotate_bars(ax2, bars1a)
    annotate_bars(ax2, bars1b)
    
    
    

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















