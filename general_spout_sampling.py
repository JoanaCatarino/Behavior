# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 09:21:21 2025

@author: JoanaCatarino
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import re

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

def load_trial_counts(file_path):
    df = pd.read_csv(file_path)

    total_trials = len(df)
    correct = df[(df["lick"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect = df[(df["lick"] == 1) & (df["reward"] == 0)].shape[0]
    incorrect_left = df[(df["left_spout"] == 1) & (df["reward"] == 0)].shape[0]
    incorrect_right = df[(df["right_spout"] == 1) & (df["reward"] == 0)].shape[0]

    if 'QW' in df.columns and not df['QW'].isna().all():
        qw_value = df['QW'].mode()[0]
    else:
        qw_value = 'NA'

    return correct, incorrect, incorrect_left, incorrect_right, qw_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    summary = []

    for file_path in args.files:
        try:
            date, box = extract_metadata(file_path)
            correct, incorrect, inc_left, inc_right, qw = load_trial_counts(file_path)
            summary.append({
                "date": date,
                "box": box,
                "correct": correct,
                "incorrect": incorrect,
                "incorrect_left": inc_left,
                "incorrect_right": inc_right,
                "QW": qw
            })
        except Exception as e:
            print(f"⚠️ Skipping file due to error: {file_path}\n{e}")

    if not summary:
        print("No valid data to plot.")
        return

    # Sort by date
    summary.sort(key=lambda x: x["date"])

    # Extract data
    dates = [s["date"] for s in summary]
    corrects = [s["correct"] for s in summary]
    incorrects = [s["incorrect"] for s in summary]
    inc_lefts = [s["incorrect_left"] for s in summary]
    inc_rights = [s["incorrect_right"] for s in summary]
    boxes = [s["box"] for s in summary]
    qws = [s["QW"] for s in summary]
    day_labels = [f"Day {i+1}" for i in range(len(dates))]
    x = range(len(dates))

    # QW colors
    qw_colors = {
        0: "#FFFFFF",
        1: "#FAD4D4",
        2: "#FCE5CD",
        3: "#FFF2CC",
        'NA': "#F5F5F5"
    }

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.8)

    # Add QW background
    for ax in axs:
        for i, qw in enumerate(qws):
            color = qw_colors.get(qw, "#F5F5F5")
            ax.axvspan(i - 0.4, i + 0.4, color=color, alpha=0.3, zorder=0)

    # Plot 1: Correct vs Incorrect
    axs[0].plot(x, corrects, '-o', label="Correct Trials", color='green')
    axs[0].plot(x, incorrects, '-o', label="Incorrect Trials", color='red')
    axs[0].set_title("Correct and Incorrect Trials Across Days", pad=20)
    axs[0].set_ylabel("Count")
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    for i, box in enumerate(boxes):
        y = max(corrects[i], incorrects[i]) + 4
        axs[0].text(i, y, f"Box {box}", ha='center', fontsize=8)
    axs[0].legend(frameon=False)

    # Plot 2: Incorrect Left and Right
    axs[1].plot(x, inc_lefts, '-o', label="Incorrect Left", color='#D95F02')
    axs[1].plot(x, inc_rights, '-o', label="Incorrect Right", color='#7570B3')
    axs[1].set_title("Incorrect Left vs Right Across Days", pad=20)
    axs[1].set_ylabel("Count")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(day_labels)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    for i, box in enumerate(boxes):
        y = max(inc_lefts[i], inc_rights[i]) + 4
        axs[1].text(i, y, f"Box {box}", ha='center', fontsize=8)
    axs[1].legend(frameon=False)

    # QW Legend
    qw_handles = [
        Patch(facecolor=color, edgecolor='k', label=f"QW {qw}")
        for qw, color in qw_colors.items() if qw != 'NA'
    ]
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + qw_handles, frameon=False)

    plt.suptitle(f"Animal {args.animal} — Spout Sampling data across days", fontsize=14, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save outputs
    base_dir = Path(r"L:/dmclab/Joana/Behavior/Data") / args.animal / "Analysis" / "Across-days"
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_filename = base_dir / f"{args.animal}_SpoutSamp_across_days"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_filename.with_suffix(f".{ext}"), dpi=500)

    # Export CSV
    df_export = pd.DataFrame(summary)
    df_export.to_csv(base_dir / f"{args.animal}_SpoutSamp_across_days.csv", index=False)
    print(f"✅ Analysis complete and saved for animal {args.animal}")

if __name__ == "__main__":
    main()