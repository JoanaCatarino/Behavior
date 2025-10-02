# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 18:59:01 2025

@author: JoanaCatarino

overall plots across sessions for each animal 
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
    
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_Total_Trials.png', dpi=500)
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_Total_Trials.svg', dpi=500)
    
    
    # Figure 2 - Performance across sessions ---------------------------------------------
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
    #plt.tight_layout()
    
    
    # Bottom: performance in 8KHz trials versus 16KHz trials
    bars1a = ax2.bar(x2, y2, width=bw, color="#ADD3D1", label="8Khz")
    bars1b = ax2.bar(x3, y3, width=bw, color="#8E7FAD", label="16KHz")

    ax2.set_xticks(x_spaced) 
    ax2.set_ylim(0,100)
    ax2.set_xlabel("Sessions")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title(f"Performance per tone", pad=20)
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticklabels(day_labels, rotation=0)  # show 'Day N' labels on top plot too
    pad = spacing * 0.7  # optional padding
    ax2.set_xlim(x_spaced[0] - pad, x_spaced[-1] + pad)
    
    # Legend under the bottom plot
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    # leave room for the legend below the axes
    fig.subplots_adjust(bottom=0.20)
    
    annotate_bars(ax2, bars1a)
    annotate_bars(ax2, bars1b)
   
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_Performance.png', dpi=500)
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_Performance.svg', dpi=500)
        
   
    # Figure 3 - %correct across sessions -------------------------------------------------------
    x = list(range(1, len(df) + 1))
    y1 = df["percent_correct"].tolist()
    y2 = df["percent_correct_8KHz"].tolist()
    y3 = df["percent_correct_16KHz"].tolist()
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
    ax1.set_ylabel("% Correct")
    ax1.set_title(f"% Correct per Session — Animal {animal}", pad=20)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #plt.tight_layout()
    
    
    # Bottom: performance in 8KHz trials versus 16KHz trials
    bars1a = ax2.bar(x2, y2, width=bw, color="#ADD3D1", label="8Khz")
    bars1b = ax2.bar(x3, y3, width=bw, color="#8E7FAD", label="16KHz")

    ax2.set_xticks(x_spaced) 
    ax2.set_ylim(0,100)
    ax2.set_xlabel("Sessions")
    ax2.set_ylabel("% Correct")
    ax2.set_title(f"% Correct per tone", pad=30)
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticklabels(day_labels, rotation=0)  # show 'Day N' labels on top plot too
    pad = spacing * 0.7  # optional padding
    ax2.set_xlim(x_spaced[0] - pad, x_spaced[-1] + pad)
    
    # Legend under the bottom plot
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    # leave room for the legend below the axes
    fig.subplots_adjust(bottom=0.20)
    
    annotate_bars(ax2, bars1a)
    annotate_bars(ax2, bars1b)    
    
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_%Correct.png', dpi=500)
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_%Correct.svg', dpi=500)
    
    
    # Figure 4 - Early licks and Omissions across sessions -------------------------------------------------------
    x = list(range(1, len(df) + 1))
    y1 = df["early_licks"].tolist()
    y2 = df["omission_8KHz"].tolist()
    y3 = df["omission_16KHz"].tolist()
    day_labels = [f"{i}" for i in x]


    fig, (ax1, ax2)= plt.subplots(2,1, figsize=(20,14), sharex=False, constrained_layout=False, gridspec_kw={'hspace': 0.5}, dpi=DPI)
    fig.subplots_adjust(bottom=0.20)
    
    ax1.plot(x, y1, linewidth=1, color='gray') 
    ax1.scatter(x, y1, s=30, color="#BC556F", label="Early Licks", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(day_labels, rotation=0)
    ax1.set_xlabel("Sessions")
    ax1.set_ylabel("Total Number")
    ax1.set_title(f"Early Licks per Session — Animal {animal}", pad=20)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    
    ax2.plot(x, y2, linewidth=1, color='gray') 
    ax2.scatter(x, y2, s=30, color="#5B9E9A", label="Omission 8KHz", zorder=3)
    ax2.plot(x, y3, linewidth=1, color='gray') 
    ax2.scatter(x, y3, s=30, color="#9AA1BB", label="Omission 16KHz", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels, rotation=0)
    ax2.set_xlabel("Sessions")
    ax2.set_ylabel("Total Number")
    ax2.set_title(f"omissions per tone", pad=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    #plt.tight_layout()
    
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_EarlyLicks_Omissions.png', dpi=500)
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_EarlyLicks_Omissions.svg', dpi=500)
    
    
    # Figure X — d′ (d-prime) across sessions -----------------------------------
    # Uses the dprime you already compute in load_trial_counts()
    
    # 1) Per-file session (default)
    x = list(range(1, len(df) + 1))
    y = pd.to_numeric(df["dprime"], errors="coerce").tolist()
    day_labels = [f"{i}" for i in x]
    
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=DPI)
    ax.plot(x, y, linewidth=1, color="gray", zorder=2)          # connect dots
    ax.scatter(x, y, s=30, color="#CC6FA8", zorder=3)           # dots
    
    # optional dashed stems like your Fig. 1
    for xi, yi in zip(x, y):
        if not (isinstance(yi, float) and math.isnan(yi)):
            ax.plot([xi, xi], [0, yi], linestyle="--", color="gray", linewidth=1, zorder=1)
    
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)          # reference line at 0
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=0)
    ax.set_xlabel("Sessions")
    ax.set_ylabel("d′")
    ax.set_title(f"d′ per Session — Animal {animal}", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_d.png', dpi=500)
    plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_d.svg', dpi=500)
    
    
    # Figure 5 - Lick Latency across sessions -------------------------------------------------------
        
    spacing  = 1.6                 # >1 spreads violins apart
    violin_w = 0.9                 # violin width (in data units)

    datasets, means, labels = [], [], []
    
    # Use the file paths stored in df["file"] (added in run_for_animals)
    for fp in df["file"].tolist():
        name = os.path.basename(fp)

        ddf = safe_read_csv(Path(fp)).copy()
        # compute latency exactly like load_trial_counts()
        if ("trial_start" not in ddf.columns) or ("lick_time" not in ddf.columns):
            continue
        ddf["start_RW"] = ddf["trial_start"] + 1.4
        ddf["lick_latency"] = ddf["lick_time"] - ddf["start_RW"]

        lat = ddf["lick_latency"].dropna().astype("float64")
        
        if lat.empty:
            continue

        arr = lat.to_numpy()
        n   = arr.size
        m   = float(np.mean(arr))

        datasets.append(arr)
        means.append(m)
        labels.append(str(len(labels) + 1))   # 1..K for selected sessions

    if datasets:
        x_idx = np.arange(1, len(datasets) + 1, dtype=float)
        x_pos = x_idx * spacing

        fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)

        parts = ax.violinplot(
            datasets,
            positions=x_pos,
            widths=violin_w,
            showmeans=True,
            showmedians=False,
            showextrema=True
        )
         
        # optional light fill
        for pc in parts["bodies"]:
            pc.set_facecolor("#9AC1CD")
            pc.set_alpha(0.5)

        # overlay mean ± SEM
        means = np.array(means, dtype=float)
        ax.errorbar(x_pos, means, fmt='o', color= "#815D9F", capsize=3, linewidth=1, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_ylim(0,3.5)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Sessions")
        ax.set_ylabel("Lick latency (s)")
        ax.set_title(f"Lick Latency — Animal {animal}")

        # padding so ends aren’t cramped
        pad = spacing * 0.7
        ax.set_xlim(x_pos[0] - pad, x_pos[-1] + pad)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.tight_layout()  
        
        plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_LickLatency_violin.png', dpi=500)
        plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_LickLatency_violin.svg', dpi=500)
        
        
        # Figure 6 - Lick Latency across sessions — CONNECTED DOT PLOT (same as previous plot)----------------
        
        spacing = 1.6   # >1 spreads points farther apart on x-axis
        
        datasets, means, sems, labels = [], [], [], []
        
        # Build per-session arrays (same latency calc you used)
        for fp in df["file"].tolist():
            ddf = safe_read_csv(Path(fp)).copy()
            if ("trial_start" not in ddf.columns) or ("lick_time" not in ddf.columns):
                continue
        
            ddf["start_RW"]     = ddf["trial_start"] + 1.4
            ddf["lick_latency"] = ddf["lick_time"] - ddf["start_RW"]
        
            lat = ddf["lick_latency"].dropna().astype("float64")
            if lat.empty:
                continue
        
            arr = lat.to_numpy()
            datasets.append(arr)
        
            # mean ± SEM for dot plot
            means.append(float(np.mean(arr)))
            sems.append(float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan)
        
            labels.append(str(len(labels) + 1))  # 1..K for selected sessions
        
        if datasets:
            x_idx = np.arange(1, len(datasets) + 1, dtype=float)
            x_pos = x_idx * spacing
        
            fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
        
            # connected line + dots
            ax.plot(x_pos, means, linewidth=1, color='gray', zorder=2)                 # connecting line
            ax.scatter(x_pos, means, s=30, color="#815D9F", zorder=3)                  # dots
        
            # optional: show SEM as error bars
            ax.errorbar(x_pos, means, yerr=sems, fmt='none', ecolor="#815D9F",
                        elinewidth=1, capsize=3, zorder=2)
        
            # axes / labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylim(0, 3.5)
            ax.set_xlabel("Sessions")
            ax.set_ylabel("Lick latency (s)")
            ax.set_title(f"Lick Latency — Animal {animal}")
        
            # padding so ends aren’t cramped
            pad = spacing * 0.7
            ax.set_xlim(x_pos[0] - pad, x_pos[-1] + pad)
        
            # clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.tight_layout()
        
            # save (new filenames to distinguish from violin version)
            outdir = f"L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis"
            Path(outdir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{outdir}/{animal}_LickLatency_dot.png", dpi=500)
            plt.savefig(f"{outdir}/{animal}_LickLatency_dot.svg", dpi=500)
            # plt.show()  # uncomment if running interactively
        else:
            print("No valid lick latency data to plot for Figure 5.")
                
                
        
        # Figure 7 - Lick Latency across sessions - correct vs incorrect - Violin plot -------------------------------------------------------
        
        spacing   = 2.6    # >1 spreads days farther apart on x-axis
        violin_w  = 0.7    # width of each violin (in data units)
        group_sep = 0.8    # horizontal separation between the two violins inside a day
        
        # Collect per-session arrays
        center_positions = []   # centers for each session tick
        tick_labels      = []
        inc_positions, inc_data, inc_means, inc_sems = [], [], [], []
        cor_positions, cor_data, cor_means, cor_sems = [], [], [], []
        
        day_index = 0
        for fp in df["file"].tolist():
            ddf = safe_read_csv(Path(fp)).copy()
            if ("trial_start" not in ddf.columns) or ("lick_time" not in ddf.columns):
                continue
        
            # latency (match your pipeline)
            ddf["start_RW"]    = ddf["trial_start"] + 1.4
            ddf["lick_latency"] = ddf["lick_time"] - ddf["start_RW"]
            lat = ddf["lick_latency"].astype("float64")
        
            # per-group arrays
            inc_arr = lat[(ddf["punishment"] == 1) & lat.notna()].to_numpy()
            cor_arr = lat[(ddf["reward"]     == 1) & lat.notna()].to_numpy()
        
            # skip day if no data at all
            if (inc_arr.size == 0) and (cor_arr.size == 0):
                continue
        
            day_index += 1
            x_center = day_index * spacing
            center_positions.append(x_center)
            tick_labels.append(str(day_index))
        
            # incorrect (punishment==1)
            if inc_arr.size > 0:
                inc_positions.append(x_center - group_sep/2)
                inc_data.append(inc_arr)
                inc_means.append(float(np.mean(inc_arr)))
                inc_sems.append(float(np.std(inc_arr, ddof=1) / np.sqrt(inc_arr.size)) if inc_arr.size > 1 else np.nan)
        
            # correct (reward==1)
            if cor_arr.size > 0:
                cor_positions.append(x_center + group_sep/2)
                cor_data.append(cor_arr)
                cor_means.append(float(np.mean(cor_arr)))
                cor_sems.append(float(np.std(cor_arr, ddof=1) / np.sqrt(cor_arr.size)) if cor_arr.size > 1 else np.nan)
        
        if center_positions:
            fig, ax = plt.subplots(figsize=(16, 6), dpi=DPI)
        
            # Plot violins (two calls so we can style each group differently)
            parts_inc = ax.violinplot(
                inc_data, positions=inc_positions, widths=violin_w,
                showmeans=False, showmedians=False, showextrema=True
            )
            parts_cor = ax.violinplot(
                cor_data, positions=cor_positions, widths=violin_w,
                showmeans=False, showmedians=False, showextrema=True
            )
        
            # Colors (edit to taste)
            col_inc = "#BC556F"  # incorrect (punishment==1)
            col_cor = "#5B9E9A"  # correct   (reward==1)
        
            # Style violins
            for pc in parts_inc["bodies"]:
                pc.set_facecolor(col_inc)
                pc.set_edgecolor(col_inc)
                pc.set_alpha(0.5)
            for pc in parts_cor["bodies"]:
                pc.set_facecolor(col_cor)
                pc.set_edgecolor(col_cor)
                pc.set_alpha(0.5)
        
            # Style extrema lines
            for key in ("cmins", "cmaxes", "cbars"):
                if key in parts_inc: parts_inc[key].set_color(col_inc)
                if key in parts_cor: parts_cor[key].set_color(col_cor)
        
            # Overlay means ± SEM for each group
            if inc_positions:
                ax.errorbar(inc_positions, inc_means, yerr=inc_sems, fmt='o',
                            ms=4, color=col_inc, ecolor=col_inc, capsize=3, lw=1, zorder=3)
            if cor_positions:
                ax.errorbar(cor_positions, cor_means, yerr=cor_sems, fmt='o',
                            ms=4, color=col_cor, ecolor=col_cor, capsize=3, lw=1, zorder=3)
        
            # Axes/labels
            ax.set_ylim(0,3.5)
            ax.set_xticks(center_positions)
            ax.set_xticklabels(tick_labels, rotation=0)
            ax.set_xlabel("Sessions")
            ax.set_ylabel("Lick latency (s)")
            ax.set_title(f"Lick Latency per Session — Incorrect vs Correct — Animal {animal}")
        
            # Y-limits (optional): uncomment to fix range
            # ax.set_ylim(0, 3)
        
            # Horizontal padding so edge violins aren’t cramped
            pad = spacing * 0.7
            ax.set_xlim(center_positions[0] - pad, center_positions[-1] + pad)
        
            # Legend (two proxy patches)
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=col_inc, edgecolor=col_inc, alpha=0.5, label="Incorrect (punishment=1)"),
                Patch(facecolor=col_cor, edgecolor=col_cor, alpha=0.5, label="Correct (reward=1)")
            ]
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=2, frameon=False)
            fig.subplots_adjust(bottom=0.18)
        
            # Clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.tight_layout()
            #plt.show()
        else:
            print("No valid latency data to plot for Figure 5 (no reward/punishment latencies found).")


        # Figure 8 - Same as before but as a dot plot
        
        spacing   = 2.6   # >1 spreads days farther apart on x-axis
        group_sep = 0.8   # horizontal separation between the two series inside a day
        
        # Colors
        col_inc = "#BC556F"  # incorrect (punishment==1)
        col_cor = "#5B9E9A"  # correct   (reward==1)
        
        # Collect per-session means/SEMs
        center_positions = []   # x centers (one per plotted day)
        tick_labels      = []   # "1","2",...
        
        inc_means, inc_sems = [], []
        cor_means, cor_sems = [], []
        
        day_index = 0
        for fp in df["file"].tolist():
            ddf = safe_read_csv(Path(fp)).copy()
            if ("trial_start" not in ddf.columns) or ("lick_time" not in ddf.columns):
                continue
        
            # latency (match your pipeline)
            ddf["start_RW"]     = ddf["trial_start"] + 1.4
            ddf["lick_latency"] = ddf["lick_time"] - ddf["start_RW"]
            lat = ddf["lick_latency"].astype("float64")
        
            # groups
            inc_arr = lat[(ddf["punishment"] == 1) & lat.notna()].to_numpy()
            cor_arr = lat[(ddf["reward"]     == 1) & lat.notna()].to_numpy()
        
            # skip day if no data at all
            if (inc_arr.size == 0) and (cor_arr.size == 0):
                continue
        
            # new plotted day
            day_index += 1
            x_center = day_index * spacing
            center_positions.append(x_center)
            tick_labels.append(str(day_index))
        
            # per-day stats (mean ± SEM); use NaN if missing
            if inc_arr.size > 0:
                inc_means.append(float(np.mean(inc_arr)))
                inc_sems.append(float(np.std(inc_arr, ddof=1) / np.sqrt(inc_arr.size)) if inc_arr.size > 1 else np.nan)
            else:
                inc_means.append(np.nan)
                inc_sems.append(np.nan)
        
            if cor_arr.size > 0:
                cor_means.append(float(np.mean(cor_arr)))
                cor_sems.append(float(np.std(cor_arr, ddof=1) / np.sqrt(cor_arr.size)) if cor_arr.size > 1 else np.nan)
            else:
                cor_means.append(np.nan)
                cor_sems.append(np.nan)
        
        if center_positions:
            fig, ax = plt.subplots(figsize=(16, 6), dpi=DPI)
        
            centers = np.array(center_positions, dtype=float)
            x_inc   = centers - (group_sep / 2.0)
            x_cor   = centers + (group_sep / 2.0)
        
            inc_means = np.array(inc_means, dtype=float)
            cor_means = np.array(cor_means, dtype=float)
            inc_sems  = np.array(inc_sems, dtype=float)
            cor_sems  = np.array(cor_sems, dtype=float)
        
            # --- connected dot plots ---
            ax.plot(x_inc, inc_means, linewidth=1, marker='o', ms=4, color=col_inc, label="Incorrect (pun=1)")
            ax.plot(x_cor, cor_means, linewidth=1, marker='o', ms=4, color=col_cor, label="Correct (rew=1)")
        
            # optional: SEM error bars
            ax.errorbar(x_inc, inc_means, yerr=inc_sems, fmt='none', ecolor=col_inc, elinewidth=1, capsize=3, zorder=2)
            ax.errorbar(x_cor, cor_means, yerr=cor_sems, fmt='none', ecolor=col_cor, elinewidth=1, capsize=3, zorder=2)
        
            # Axes/labels
            ax.set_ylim(0, 3.5)  # adjust if you want a different range
            ax.set_xticks(centers)
            ax.set_xticklabels(tick_labels, rotation=0)
            ax.set_xlabel("Sessions")
            ax.set_ylabel("Lick latency (s)")
            ax.set_title(f"Lick Latency per Session — Incorrect vs Correct — Animal {animal}")
        
            # padding so edge points aren’t cramped
            pad = spacing * 0.7
            ax.set_xlim(centers[0] - pad, centers[-1] + pad)
        
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
            fig.subplots_adjust(bottom=0.18)
        
            # clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #plt.tight_layout()
            #plt.show()
        else:
            print("No valid latency data to plot (no reward/punishment latencies found).")
            

        # Figure 9 — Lick Latency per tone & outcome (connected dot plots) --------------------
        # Two subplots:
        #   Top  : 8 kHz — Incorrect (pun=1) vs Correct (rew=1)
        #   Bottom: 16 kHz — Incorrect (pun=1) vs Correct (rew=1)
    
        # Colors (match your palette)
        col_inc = "#BC556F"  # incorrect (punishment==1)
        col_cor = "#5B9E9A"  # correct   (reward==1)
    
        n_days = len(df["file"])
        x_idx = np.arange(1, n_days + 1, dtype=float)  # 1..N sessions for labeling
    
        # Pre-allocate with NaN so lines naturally break when a day has no data
        mean8_inc = np.full(n_days, np.nan, dtype=float)
        sem8_inc  = np.full(n_days, np.nan, dtype=float)
        mean8_cor = np.full(n_days, np.nan, dtype=float)
        sem8_cor  = np.full(n_days, np.nan, dtype=float)
    
        mean16_inc = np.full(n_days, np.nan, dtype=float)
        sem16_inc  = np.full(n_days, np.nan, dtype=float)
        mean16_cor = np.full(n_days, np.nan, dtype=float)
        sem16_cor  = np.full(n_days, np.nan, dtype=float)
    
        # Build per-day stats
        for day_i, fp in enumerate(df["file"].tolist()):
            ddf = safe_read_csv(Path(fp)).copy()
            if ("trial_start" not in ddf.columns) or ("lick_time" not in ddf.columns):
                continue
    
            # Latency as in your pipeline (last version you shared used +1.4)
            ddf["start_RW"]     = ddf["trial_start"] + 1.4
            ddf["lick_latency"] = ddf["lick_time"] - ddf["start_RW"]
            lat = ddf["lick_latency"].astype("float64")
    
            # 8 kHz
            inc8 = lat[(ddf["8KHz"] == 1) & (ddf["punishment"] == 1) & lat.notna()].to_numpy()
            cor8 = lat[(ddf["8KHz"] == 1) & (ddf["reward"]     == 1) & lat.notna()].to_numpy()
            if inc8.size > 0:
                mean8_inc[day_i] = float(np.mean(inc8))
                if inc8.size > 1:
                    sem8_inc[day_i] = float(np.std(inc8, ddof=1) / np.sqrt(inc8.size))
            if cor8.size > 0:
                mean8_cor[day_i] = float(np.mean(cor8))
                if cor8.size > 1:
                    sem8_cor[day_i] = float(np.std(cor8, ddof=1) / np.sqrt(cor8.size))
    
            # 16 kHz
            inc16 = lat[(ddf["16KHz"] == 1) & (ddf["punishment"] == 1) & lat.notna()].to_numpy()
            cor16 = lat[(ddf["16KHz"] == 1) & (ddf["reward"]     == 1) & lat.notna()].to_numpy()
            if inc16.size > 0:
                mean16_inc[day_i] = float(np.mean(inc16))
                if inc16.size > 1:
                    sem16_inc[day_i] = float(np.std(inc16, ddof=1) / np.sqrt(inc16.size))
            if cor16.size > 0:
                mean16_cor[day_i] = float(np.mean(cor16))
                if cor16.size > 1:
                    sem16_cor[day_i] = float(np.std(cor16, ddof=1) / np.sqrt(cor16.size))
    
        # Plot
        fig, (ax8, ax16) = plt.subplots(2, 1, figsize=(16, 10), dpi=DPI, gridspec_kw={'hspace': 0.4})
    
        labels = [str(i) for i in x_idx.astype(int)]
    
        # --- Top: 8 kHz (Incorrect vs Correct) ---
        # connect the dots; NaNs create gaps automatically
        ax8.plot(x_idx, mean8_inc, linewidth=1, marker='o', ms=4, color=col_inc, label="Incorrect")
        ax8.plot(x_idx, mean8_cor, linewidth=1, marker='o', ms=4, color=col_cor, label="Correct")
    
        # optional: show errorbars (SEM) at each point
        ax8.errorbar(x_idx, mean8_inc, yerr=sem8_inc, fmt='none', ecolor=col_inc, elinewidth=1, capsize=3, zorder=2)
        ax8.errorbar(x_idx, mean8_cor, yerr=sem8_cor, fmt='none', ecolor=col_cor, elinewidth=1, capsize=3, zorder=2)
    
        ax8.set_xticks(x_idx)
        ax8.set_ylim(0,3)
        ax8.set_xticklabels(labels, rotation=0)
        ax8.set_xlabel("Sessions")
        ax8.set_ylabel("Lick latency (s)")
        ax8.set_title(f"8 kHz — Lick Latency per Session (Incorrect vs Correct) — Animal {animal}")
        ax8.spines['top'].set_visible(False); ax8.spines['right'].set_visible(False)
    
    
        # --- Bottom: 16 kHz (Incorrect vs Correct) ---
        ax16.plot(x_idx, mean16_inc, linewidth=1, marker='o', ms=4, color=col_inc, label="Incorrect")
        ax16.plot(x_idx, mean16_cor, linewidth=1, marker='o', ms=4, color=col_cor, label="Correct")
    
        ax16.errorbar(x_idx, mean16_inc, yerr=sem16_inc, fmt='none', ecolor=col_inc, elinewidth=1, capsize=3, zorder=2)
        ax16.errorbar(x_idx, mean16_cor, yerr=sem16_cor, fmt='none', ecolor=col_cor, elinewidth=1, capsize=3, zorder=2)
    
        ax16.set_xticks(x_idx)
        ax16.set_ylim(0,3)
        ax16.set_xticklabels(labels, rotation=0)
        ax16.set_xlabel("Sessions")
        ax16.set_ylabel("Lick latency (s)")
        ax16.set_title(f"16 kHz — Lick Latency per Session (Incorrect vs Correct) — Animal {animal}")
        ax16.spines['top'].set_visible(False); ax16.spines['right'].set_visible(False)
        ax16.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.subplots_adjust(bottom=0.12)
    
        plt.tight_layout()
        
        #plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_LickLatency_correct_incorrect_violin.png', dpi=500)
        #plt.savefig(f'L:/dmclab/Joana/Behavior/Data/{animal}/Analysis/Overall_Analysis/{animal}_LickLatency_correct_incorrect_violin.svg', dpi=500)
        

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















