# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:21:24 2025

@author: JoanaCatarino
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import re
from scipy.stats import norm

# Regex to extract date and box from filename
filename_regex = re.compile(
    r'(?P<protocol>[^_]+)_(?P<animal>\d+)_(?P<date>\d{8})_\d+_box(?P<box>\w+)',
    re.IGNORECASE
)

def extract_metadata(file_path):
    match = filename_regex.match(Path(file_path).stem)
    if not match:
        raise ValueError(f"Filename format incorrect: {file_path}")
    return match.group("date"), match.group("box")


def load_tone_mapping(animal_id):
    mapping_file_path = Path(r"L:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv")
    spout_mapping_df = pd.read_csv(mapping_file_path)
    row = spout_mapping_df[spout_mapping_df["Animal"] == int(animal_id)].iloc[0]
    pair_5khz = f"5KHz → {row['5KHz']} spout"
    pair_10khz = f"10KHz → {row['10KHz']} spout"
    return f"Tone-spout mapping: {pair_5khz}, {pair_10khz}"

def load_trial_counts(file_path):
    
    df = pd.read_csv(file_path).fillna(0)
    
    # Compute lick latency
    df["tone_time"] = df["trial_start"] + 1.2
    df["lick_latency"] = df["lick_time"] - df["tone_time"]

    valid_licks = df[df["lick_latency"].notna()]
    left_licks = valid_licks[valid_licks["left_spout"] == 1]
    right_licks = valid_licks[valid_licks["right_spout"] == 1]

    total_trials = len(df)
    correct_left = df[(df["left_spout"] == 1) & (df["reward"] == 1)].shape[0]
    correct_right = df[(df["right_spout"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect_left = df[(df["left_spout"] == 1) & (df["punishment"] == 1)].shape[0]
    incorrect_right = df[(df["right_spout"] == 1) & (df["punishment"] == 1)].shape[0]
    early = df[df["early_lick"] == 1].shape[0] if "early_lick" in df.columns else 0
    omission_5khz = df[(df["omission"] == 1) & (df["5KHz"] == 1)].shape[0]
    omission_10khz = df[(df["omission"] == 1) & (df["10KHz"] == 1)].shape[0]
    dprime = df["d_prime"].iloc[0] if "d_prime" in df.columns else None
    hit_rate = df["hit_rate"].iloc[0] if "hit_rate" in df.columns else None
    false_alarm = df["false_alarm"].iloc[0] if "false_alarm" in df.columns else None
    latency_left = left_licks["lick_latency"].mean() if not left_licks.empty else None
    latency_right = right_licks["lick_latency"].mean() if not right_licks.empty else None
    latency_left_std = left_licks["lick_latency"].std() if not left_licks.empty else None
    latency_right_std = right_licks["lick_latency"].std() if not right_licks.empty else None
    

    # Compute hit rate, false alarm, d'
    correct = correct_left + correct_right
    incorrect = incorrect_left + incorrect_right
    total = len(df)
    
    hr = (correct + 0.5) / (total + 1)
    fa = (incorrect + 0.5) / (total + 1)
    hr = min(max(hr, 0.01), 0.99)
    fa = min(max(fa, 0.01), 0.99)
    dprime = norm.ppf(hr) - norm.ppf(fa)


    if "QW" in df.columns and not df["QW"].isna().all():
        qw_value = df["QW"].mode()[0]
    else:
        qw_value = "NA"

    autom_reward_dominant = False
    if "autom_reward" in df.columns:
        autom_reward_dominant = (df["autom_reward"] == 1).sum() > len(df) / 2

    return dict(
        correct_left=correct_left,
        correct_right=correct_right,
        incorrect_left=incorrect_left,
        incorrect_right=incorrect_right,
        early=early,
        omission_5khz=omission_5khz,
        omission_10khz=omission_10khz,
        d_prime=dprime,
        hit_rate=hr,
        false_alarm=fa,
        latency_left=latency_left,
        latency_right=latency_right,
        latency_left_std=latency_left_std,
        latency_right_std=latency_right_std,
        total=total_trials,
        QW=qw_value,
        autom_reward=autom_reward_dominant
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    summary = []

    for file_path in args.files:
        try:
            date, box = extract_metadata(file_path)
            trial_data = load_trial_counts(file_path)
            trial_data.update({"date": date, "box": box})
            summary.append(trial_data)
        except Exception as e:
            print(f"⚠️ Skipping file due to error: {file_path}\n{e}")

    if not summary:
        print("No valid data to plot.")
        return

    # Sort by date
    summary.sort(key=lambda x: x["date"])

    df = pd.DataFrame(summary)
    x = range(len(df))
    day_labels = [f"Day {i+1}" for i in x]
    
    # Extract data
    dates = [s["date"] for s in summary]
    correct_right = [s["correct_right"] for s in summary]
    correct_left = [s["correct_left"] for s in summary]
    incorrect_right = [s["incorrect_right"] for s in summary]
    incorrect_left = [s["incorrect_left"] for s in summary]
    boxes = [s["box"] for s in summary]
    qws = [s["QW"] for s in summary]
    day_labels = [f"Day {i+1}" for i in range(len(dates))]
    x = range(len(dates))

    qw_colors = {
        0: "#FFFFFF",
        1: "#FAD4D4",
        2: "#FCE5CD",
        3: "#FFF2CC",
        'NA': "#F5F5F5"
    }

    tone_mapping_str = load_tone_mapping(args.animal)

    fig, axs = plt.subplots(5, 1, figsize=(16, 20))

    # Add background (Autom_reward overrides QW)
    for ax in axs:
        for i in x:
            if df.loc[i, 'autom_reward']:
                color = "#E0CCFF"  # light purple for auto reward
            else:
                color = qw_colors.get(df.loc[i, 'QW'], "#F5F5F5")
            ax.axvspan(i - 0.4, i + 0.4, color=color, alpha=0.3, zorder=0)

    axs[0].plot(x, df['correct_left'], '-o', label="Correct Left", color='green')
    axs[0].plot(x, df['correct_right'], '-o', label="Correct Right", color='limegreen')
    axs[0].plot(x, df['incorrect_left'], '-o', label="Incorrect Left", color='red')
    axs[0].plot(x, df['incorrect_right'], '-o', label="Incorrect Right", color='darkred')
    axs[0].set_title("Trial Outcomes - correct and incorrect")
    axs[0].set_ylabel("Count")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(day_labels)
    for i, box in enumerate(boxes):
        y = max(correct_right[i], incorrect_right[i], correct_left[i], incorrect_left[i]) + 6  # extra space
        axs[0].text(i, y, f"Box {box}", ha='center', fontsize=8)
   
    # Add background (Autom_reward overrides QW)
    for ax in axs:
        for i in x:
            if df.loc[i, 'autom_reward']:
                color = "#E0CCFF"  # light purple for auto reward
            else:
                color = qw_colors.get(df.loc[i, 'QW'], "#F5F5F5")
            ax.axvspan(i - 0.4, i + 0.4, color=color, alpha=0.3, zorder=0)


    axs[1].plot(x, df['early'], '-o', label="Early Licks", color='orange')
    axs[1].plot(x, df['omission_5khz'], '-o', label="Omission 5KHz", color='#91BAD6')
    axs[1].plot(x, df['omission_10khz'], '-o', label="Omission 10KHz", color='#D69F7E')
    axs[1].set_title("Trial Outcomes - early licks and omissions")
    axs[1].set_ylabel("Count")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(day_labels)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)

    axs[2].plot(x, df['hit_rate'], '-o', label="Hit Rate", color='blue')
    axs[2].plot(x, df['false_alarm'], '-o', label="False Alarm", color='purple')
    axs[2].plot(x, df['d_prime'], '-o', label="d'", color='black')
    axs[2].set_title("Performance Metrics")
    axs[2].set_ylabel("Value")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(day_labels)
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)

    axs[3].errorbar(x, df['latency_left'], yerr=df['latency_left_std'], fmt='-o', label="Latency Left", color='#1f77b4', capsize=4)
    axs[3].errorbar(x, df['latency_right'], yerr=df['latency_right_std'], fmt='-o', label="Latency Right", color='#ff7f0e', capsize=4)
    axs[3].set_title("Mean Lick Latency ± SEM")
    axs[3].set_ylabel("Seconds")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(day_labels)
    axs[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)

    axs[4].plot(x, df['total'], '-o', label="Total Trials", color='black')
    axs[4].set_title("Total Trials")
    axs[4].set_ylabel("Trials")
    axs[4].set_xticks(x)
    axs[4].set_xticklabels(day_labels)
    axs[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, fontsize=8)

    handles = [
        Patch(facecolor=color, edgecolor='k', label=f"QW {qw}")
        for qw, color in qw_colors.items() if qw != 'NA'
    ] + [
        Patch(facecolor="#E0CCFF", edgecolor='k', label="Autom Reward Majority")
    ]
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + handles,
                  loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=9, frameon=False, fontsize=8)


    plt.subplots_adjust(hspace=0.7)
    plt.suptitle(f"Animal {args.animal} — 2-Choice Auditory data across days", fontsize=12, fontweight='bold')
    plt.figtext(0.5, 0.95, tone_mapping_str, ha='center', fontsize=10)

    base_dir = Path(r"L:/dmclab/Joana/Behavior/Data") / args.animal / "Analysis" / "Across-days"
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_filename = base_dir / f"{args.animal}_2ChoiceAuditory_across_days"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_filename.with_suffix(f".{ext}"), dpi=500)

    df.to_csv(base_dir / f"{args.animal}_2ChoiceAuditory_across_days.csv", index=False)
    print(f"✅ Analysis complete and saved for animal {args.animal}")

if __name__ == "__main__":
    main()
