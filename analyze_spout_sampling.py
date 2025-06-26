# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 17:26:54 2025

@author: JoanaCatarino
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from pathlib import Path

def analyze(file_path, animal, date, box, output_dir):
    print(f"Starting Spout Sampling analysis for: {file_path}")
    df = pd.read_csv(file_path)
    
    # Get info from file name 
    df = pd.read_csv(file_path).fillna(0)
    protocol = "Spout Sampling"
    fig_title = f"{protocol} | Animal: {animal} | Date: {date} | Box {box}"
    

    required_columns = {'trial_number', 'left_spout', 'right_spout', 'reward', 'omission', 'lick'}
    if not required_columns.issubset(df.columns):
        print(f"Skipping file (missing columns): {file_path}")
        return

    df = df.sort_values("trial_number")
    df = df.fillna(0)

    # Define outcomes (excluding omission)
    df["outcome"] = ""
    df.loc[(df["left_spout"] == 1) & (df["omission"] == 0), "outcome"] = "left"
    df.loc[(df["right_spout"] == 1) & (df["omission"] == 0), "outcome"] = "right"

    outcome_to_y = {"left": 1, "right": 0}
    df = df[df["outcome"] != ""]
    df["y_pos"] = df["outcome"].map(outcome_to_y)
    df["color"] = df["reward"].apply(lambda r: "green" if r == 1 else "gray")

    # Compute bar plot values
    total_trials = len(df)
    correct_trials = df[(df["lick"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect_trials = df[(df["lick"] == 1) & (df["reward"] == 0)].shape[0]
    correct_left = df[(df["left_spout"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect_left = df[(df["left_spout"] == 1) & (df["reward"] == 0)].shape[0]
    correct_right = df[(df["right_spout"] == 1) & (df["reward"] == 1)].shape[0]
    incorrect_right = df[(df["right_spout"] == 1) & (df["reward"] == 0)].shape[0]

    bar_labels = [
        "Total Trials", "Correct Trials", "Incorrect Trials",
        "Correct Left", "Incorrect Left", "Correct Right", "Incorrect Right"
    ]
    bar_values = [
        total_trials, correct_trials, incorrect_trials,
        correct_left, incorrect_left, correct_right, incorrect_right
    ]
    bar_colors = [
        "#A9CCE3", "green", "red",
        "green", "red", "green", "red"
    ]

    # Plot
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(nrows=5, ncols=1, height_ratios=[0.2, 2, 0.2, 0.2, 1], hspace=0.7)

    # Spacer above plot
    ax_title_spacer = fig.add_subplot(gs[0])
    ax_title_spacer.axis('off')

    ax0 = fig.add_subplot(gs[1])
    for outcome, y_val in outcome_to_y.items():
        subset = df[df["outcome"] == outcome]
        ax0.scatter(subset["trial_number"], [y_val]*len(subset), c=subset["color"], label=outcome, s=20, zorder=3)

    ax0.set_yticks(list(outcome_to_y.values()))
    ax0.set_yticklabels(["Right Spout", "Left Spout"][::-1])
    ax0.set_xlabel("Trial Number")
    ax0.set_title(f" Performance over trials")
    ax0.set_xlim(df["trial_number"].min() - 1, df["trial_number"].max() + 1)
    ax0.set_ylim(-0.5, 1.5)
    ax0.grid(True, zorder=1)

    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')
    ax_legend.legend(handles=[
        Patch(color="green", label="Rewarded Trial"),
        Patch(color="gray", label="Non-rewarded Trial")
    ], loc='center', ncol=2, fontsize=10, frameon=False)

    # Spacer
    ax_spacer = fig.add_subplot(gs[3])
    ax_spacer.axis('off')

    # Bar plot
    ax1 = fig.add_subplot(gs[4])
    bars = ax1.bar(bar_labels, bar_values, color=bar_colors)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1, int(height),
                 ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel("Count")
    ax1.set_title("Detailed trial counts")
    ax1.tick_params(axis='x', labelrotation=30)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig.suptitle(fig_title, fontsize=14, y=0.90)

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = Path(file_path).stem
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{base_filename}_summary.{ext}", dpi=400)

    print("DONE!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--animal', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--box', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    analyze(args.file, args.animal, args.date, args.box, args.output)

if __name__ == "__main__":
    main()

