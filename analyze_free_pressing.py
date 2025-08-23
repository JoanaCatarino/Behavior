# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:27:44 2025

@author: JoanaCatarino
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path

def analyze(file_path, animal, date, box, output_dir):
    print(f"Starting Free Pressing analysis for: {file_path}")
    df = pd.read_csv(file_path)

    required_columns = {'lick', 'left_spout', 'right_spout', 'QW', 'trial_start', 'trial_end', 'lick_time', 'session_start'}
    if not required_columns.issubset(df.columns):
        print(f"Skipping file (missing columns): {file_path}")
        return

    # Preprocessing
    session_start = df['session_start'].iloc[0]
    licks_df = df[['trial_number', 'lick', 'left_spout', 'right_spout']].copy()
    licks_df['time'] = licks_df.index * 0.1
    licks_only = licks_df[licks_df['lick'] == 1].copy()
    licks_only['true_relative_time_min'] = (df.loc[licks_only.index, 'lick_time'] - session_start) / 60

    start_time = df['trial_start'].min()
    end_time = df['trial_end'].max()
    session_duration_minutes = (end_time - start_time) / 60

    # Cumulative licks
    licks_only['is_left'] = licks_only['left_spout'] == 1
    licks_only['is_right'] = licks_only['right_spout'] == 1
    cumulative_total = licks_only.groupby('trial_number').size().cumsum()
    cumulative_left = licks_only[licks_only['is_left']].groupby('trial_number').size().cumsum()
    cumulative_right = licks_only[licks_only['is_right']].groupby('trial_number').size().cumsum()

    total_presses = len(licks_only)
    left_presses = licks_only['left_spout'].sum()
    right_presses = licks_only['right_spout'].sum()

    # Plot setup
    labels = ['Total', 'Left', 'Right']
    values = [total_presses, left_presses, right_presses]
    colors = ['#F5A885', '#BB5C7A', '#5EA5A3']
    x_pos = [0.5, 1.0, 1.5]

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"Free Pressing | Animal: {animal} | Date: {date} | Box {box}", fontsize=14)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.6)

    # QW background color map
    qw_colors = {
        1: "#FAD4D4",  # light pink
        2: "#FCE5CD",  # light orange
        3: "#FFF2CC"   # light yellow
    }

    # Plot 1: Licks Over Time
    ax0 = fig.add_subplot(gs[0, :])
    for _, row in df.iterrows():
        start_min = (row["trial_start"] - session_start) / 60
        end_min = (row["trial_end"] - session_start) / 60
        color = qw_colors.get(row["QW"], None)
        if color:
            ax0.axvspan(start_min, end_min, color=color, alpha=0.5)
    ax0.scatter(licks_only['true_relative_time_min'], [1]*len(licks_only), alpha=0.6, color='#AC90BF', marker='x')
    ax0.set_xlabel("Time (min)")
    ax0.set_title("Presses Over Time")
    ax0.set_yticks([])
    ax0.set_xlim(0, session_duration_minutes)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    # Plot 2: Cumulative Licks Over Trials
    ax1 = fig.add_subplot(gs[1, :])
    for _, row in df.iterrows():
        trial = row["trial_number"]
        color = qw_colors.get(row["QW"], None)
        if color:
            ax1.axvspan(trial - 0.5, trial + 0.5, color=color, alpha=0.5)
    ax1.plot(cumulative_total, drawstyle='steps-post', label="Total Presses", color="#F5A885")
    ax1.plot(cumulative_left, drawstyle='steps-post', label="Left Press", color="#BB5C7A")
    ax1.plot(cumulative_right, drawstyle='steps-post', label="Right Press", color="#5EA5A3")
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Total Presses")
    ax1.set_title("Presses Over Trials")
    ax1.grid(True, color="#DAD4D4")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 3: Summary Bar Plot
    ax2 = fig.add_subplot(gs[2, 0])
    bars = ax2.bar(x_pos, values, width=0.2, color=colors)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Presses")
    ax2.set_title("Press Summary")
    ax2.set_xlim(0.3, 1.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Plot 4: Combined Legend
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.axis('off')
    qw_legend_patches = [
        Patch(facecolor="#FAD4D4", edgecolor='none', alpha=0.5, label="QW = 1"),
        Patch(facecolor="#FCE5CD", edgecolor='none', alpha=0.5, label="QW = 2"),
        Patch(facecolor="#FFF2CC", edgecolor='none', alpha=0.5, label="QW = 3")
    ]
    lick_type_legend = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
    all_legend = qw_legend_patches + lick_type_legend
    ax3.legend(handles=all_legend, loc='center', fontsize=11, frameon=False, ncol=1)

    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_filename = Path(file_path).stem
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ['png', 'pdf', 'svg']:
        save_path = Path(output_dir) / f"{base_filename}_summary.{ext}"
        fig.savefig(save_path, dpi=400)
        print(f"Saved: {save_path}")

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
