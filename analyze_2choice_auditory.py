# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:26:34 2025

@author: JoanaCatarino
"""
# analyze_2choice_auditory.py

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import norm

def analyze(file_path, animal, date, box, output_dir):
   
    # Get info from file name 
    df = pd.read_csv(file_path).fillna(0)
    protocol = "Two-choice Auditory task"
    fig_title = f"{protocol} | Animal: {animal} | Date: {date} | Box {box}"

    # Extract tone-spout mapping for a specific animal
    mapping_file_path = Path(r"L:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv")
    spout_mapping_df = pd.read_csv(mapping_file_path)
    animal_id = int(animal)
    mapping_row = spout_mapping_df[spout_mapping_df["Animal"] == animal_id].iloc[0]
    pair_5khz = f"5KHz → {mapping_row['5KHz']} spout"
    pair_10khz = f"10KHz → {mapping_row['10KHz']} spout"
    mapping_subtitle = f"Tone-spout mapping: {pair_5khz}, {pair_10khz}"

    # Calculate lick latency
    df["tone_time"] = df["trial_start"] + 1.2
    df["lick_latency"] = df["lick_time"] - df["tone_time"]

    # Define the different variables to plot    
    valid_licks = df[df["lick_latency"].notna()]
    left_licks = valid_licks[valid_licks["left_spout"] == 1]
    right_licks = valid_licks[valid_licks["right_spout"] == 1]
    left_colors = left_licks["reward"].map({1: "green", 0: "red"}) 
    right_colors = right_licks["reward"].map({1: "green", 0: "red"})

    categories = [
        "early lick", "omission", "correct left", "correct right",
        "incorrect left", "incorrect right"
    ]
    category_to_y = {cat: i for i, cat in enumerate(categories)}
    category_colors = {
        "early lick": "black", "omission": "gray",
        "correct left": "green", "correct right": "green",
        "incorrect left": "red", "incorrect right": "red",
    }

    df["category"] = None
    df.loc[df["early_lick"] == 1, "category"] = "early lick"
    df.loc[(df["omission"] == 1) & (df["category"].isna()), "category"] = "omission"
    df.loc[(df["left_spout"] == 1) & (df["reward"] == 1), "category"] = "correct left"
    df.loc[(df["right_spout"] == 1) & (df["reward"] == 1), "category"] = "correct right"
    df.loc[(df["punishment"] == 1) & (df["left_spout"] == 1), "category"] = "incorrect left"
    df.loc[(df["punishment"] == 1) & (df["right_spout"] == 1), "category"] = "incorrect right"

    df_plot = df[df["category"].notna()]
    df_sorted = df_plot.sort_values("trial_number")
    df_sorted["autom_reward"] = df_sorted["autom_reward"].astype(int) 
    min_trial = df_sorted["trial_number"].min()
    max_trial = df_sorted["trial_number"].max()

    # Count data for bar plot
    count_5khz = df[df["5KHz"] == 1].shape[0]
    count_10khz = df[df["10KHz"] == 1].shape[0]
    omission_5khz = df[(df["omission"] == 1) & (df["5KHz"] == 1)].shape[0]
    omission_10khz = df[(df["omission"] == 1) & (df["10KHz"] == 1)].shape[0]
    total_early_licks = df[df["early_lick"] == 1].shape[0]
    count_left_correct = df[(df["left_spout"] == 1) & (df["reward"] == 1)].shape[0]
    count_left_incorrect = df[(df["left_spout"] == 1) & (df["punishment"] == 1)].shape[0]
    count_right_correct = df[(df["right_spout"] == 1) & (df["reward"] == 1)].shape[0]
    count_right_incorrect = df[(df["right_spout"] == 1) & (df["punishment"] == 1)].shape[0]
    
    bar_labels = [
        "5KHz Trials", "10KHz Trials", 
        "5KHz Omissions", "10KHz Omissions",
        "Early Licks",
        "Left Correct", "Left Incorrect",
        "Right Correct", "Right Incorrect"
    ]
    
    bar_values = [
        count_5khz, count_10khz, 
        omission_5khz, omission_10khz,
        total_early_licks,
        count_left_correct, count_left_incorrect,
        count_right_correct, count_right_incorrect
    ]
    

    # Make figure with different plots
    
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[1, 1.5, 1, 1], hspace=0.8, wspace=0.3)

    # Row 1 - Trial Outcomes
    ax0 = fig.add_subplot(gs[0, :])
    for _, row in df.iterrows():
        trial = row["trial_number"]
        if row["autom_reward"] == 1:
            ax0.axvspan(trial - 0.5, trial + 0.5, color="purple", alpha=0.1)
        elif row["5KHz"] == 1:
            ax0.axvspan(trial - 0.5, trial + 0.5, color="#BFF9FF", alpha=0.2)
        elif row["10KHz"] == 1:
            ax0.axvspan(trial - 0.5, trial + 0.5, color="#F5A783", alpha=0.2)
    
    for category in categories:
        subset = df_sorted[(df_sorted["category"] == category) & (df_sorted["autom_reward"] != 1)]
        
        ax0.scatter(subset["trial_number"], [category_to_y[category]] * len(subset),
                    color=category_colors[category], s=10, zorder=3)
    
    ax0.invert_yaxis()
    ax0.set_yticks(list(category_to_y.values()))
    ax0.set_yticklabels(list(category_to_y.keys()))
    ax0.set_xlabel("Trial Number")
    ax0.set_title("Trial Outcomes across session")
    ax0.set_xlim(min_trial - 1, max_trial + 1)
    ax0.legend(handles=[
        Patch(facecolor="#BFF9FF", edgecolor="none", alpha=0.2, label="5KHz stim"),
        Patch(facecolor="#F5A783", edgecolor="none", alpha=0.2, label="10KHz stim"),
        Patch(facecolor="purple", edgecolor="none", alpha=0.1, label="Auto reward")
    ], bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

    # Row 2 - Lick Latencies
    # Left Spout
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(left_licks["trial_number"], left_licks["lick_latency"], color='gray')
    ax1.scatter(left_licks["trial_number"], left_licks["lick_latency"], c=left_colors)
    ax1.set_title("Lick Latency - Left Spout")
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Lick Latency (s)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right Spout
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(right_licks["trial_number"], right_licks["lick_latency"], color='gray')
    ax2.scatter(right_licks["trial_number"], right_licks["lick_latency"], c=right_colors)
    ax2.set_title("Lick Latency - Right Spout")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Lick Latency (s)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Row 3 - Counts of trial statistics    
    ax3 = fig.add_subplot(gs[2, :])
    bars = ax3.bar(bar_labels, bar_values,
                   color=["skyblue", "lightsalmon", "gray", "gray", "black", "green", "red", "green", "red"])
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval),
                 ha='center', va='bottom', fontsize=10)
    ax3.set_ylabel("Count")
    ax3.set_title("Detailed Trial Counts")
    ax3.tick_params(axis='x', labelrotation=45)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Row 4 - HR / FA / d' and Performance Breakdown
    trial_numbers = df_plot["trial_number"].astype(int).values
    total_trials = np.arange(1, len(trial_numbers)+1)
    correct_trials = (df_plot["reward"] == 1).cumsum().values
    incorrect_trials = (df_plot["punishment"] == 1).cumsum().values

    HR = (correct_trials + 0.5) / (total_trials + 1)
    FA = (incorrect_trials + 0.5) / (total_trials + 1)
    HR = np.clip(HR, 0.01, 0.99)
    FA = np.clip(FA, 0.01, 0.99)
    d_prime = norm.ppf(HR) - norm.ppf(FA)

    ax4 = fig.add_subplot(gs[3, 0])
    ax4b = ax4.twinx()
    ax4.step(trial_numbers, HR, where='post', color='black', label='Hit Rate', linewidth=2)
    ax4.step(trial_numbers, FA, where='post', color='red', label='False Alarm', linewidth=2)
    ax4b.step(trial_numbers, d_prime, color='#9DB4C0', label="d'", linewidth=2)
    ax4.set_ylim(0, 1)
    ax4b.set_ylim(min(d_prime) - 0.5, max(d_prime) + 0.5)
    ax4.set_ylabel("HR / FA")
    ax4b.set_ylabel("d'")
    ax4.set_title("Performance over Trials")
    ax4.grid(True)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

    ax5 = fig.add_subplot(gs[3, 1])
    
    num_total_trials = len(df_plot)
    num_correct = (df_plot["reward"] == 1).sum()
    num_incorrect = (df_plot["punishment"] == 1).sum()
    correct_left = ((df_plot["reward"] == 1) & (df_plot["left_spout"] == 1)).sum()
    correct_right = ((df_plot["reward"] == 1) & (df_plot["right_spout"] == 1)).sum()
    
    bar_labels_pct = ['Correct', 'Incorrect', 'Correct Left', 'Correct Right']
    bar_values_pct = [
    num_correct / num_total_trials * 100,
    num_incorrect / num_total_trials * 100,
    correct_left / num_total_trials * 100,
    correct_right / num_total_trials * 100
    ]
    
    bars = ax5.bar(bar_labels_pct, bar_values_pct, color=['green', 'red', 'green', 'green'])
    for bar in bars:
        yval = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center')
    ax5.set_ylim(0, 100)
    ax5.set_ylabel("Percentage (%)")
    ax5.set_title("Performance Summary")
    ax5.tick_params(axis='x', labelrotation=30)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    

    fig.suptitle(fig_title, fontsize=14, y=0.98)
    plt.figtext(0.5, 0.95, mapping_subtitle, ha='center', fontsize=12)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base_filename = Path(file_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in ['png', 'pdf', 'svg']:
        save_path = output_dir / f"{base_filename}_summary.{ext}"
        fig.savefig(save_path, dpi=400)
        
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
