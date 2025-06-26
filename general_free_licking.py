# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:11:33 2025

@author: JoanaCatarino
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
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

def load_lick_counts(file_path):
    df = pd.read_csv(file_path)
    left_licks = df['left_spout'].sum()
    right_licks = df['right_spout'].sum()
    total_licks = df['lick'].sum()
    return left_licks, right_licks, total_licks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    summary = []

    for file_path in args.files:
        try:
            date, box = extract_metadata(file_path)
            left, right, total = load_lick_counts(file_path)
            summary.append({
                "date": date,
                "box": box,
                "left_licks": left,
                "right_licks": right,
                "total_licks": total
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
    lefts = [s["left_licks"] for s in summary]
    rights = [s["right_licks"] for s in summary]
    totals = [s["total_licks"] for s in summary]
    boxes = [s["box"] for s in summary]
    day_labels = [f"Day {i+1}" for i in range(len(dates))]
    x = range(len(dates))

    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))

    # Plot 1: Left and Right licks
    axs[0].plot(x, lefts, '-o', label="Left Licks", color='#BB5C7A')
    axs[0].plot(x, rights, '-o', label="Right Licks", color='#5EA5A3')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(day_labels, rotation=45)
    axs[0].set_ylabel("Licks")
    axs[0].set_title(f"Left and Right licks across days", pad=20)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    legend1 = axs[0].legend()
    legend1.set_frame_on(False)

    for i, box in enumerate(boxes):
        max_lick = max(lefts[i], rights[i])
        axs[0].text(i, max_lick + 4, f"Box {box}", ha='center', va='bottom', fontsize=9)

    # Plot 2: Total licks
    axs[1].plot(x, totals, '-o', label="Total Licks", color='#F5A885')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(day_labels, rotation=45)
    axs[1].set_ylabel("Licks")
    axs[1].set_title("Total licks across days", pad=20)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    legend1 = axs[1].legend()
    legend1.set_frame_on(False)

    for i, box in enumerate(boxes):
        axs[1].text(i, totals[i] + 4, f"Box {box}", ha='center', va='bottom', fontsize=9)

    
    plt.subplots_adjust(hspace=0.6) 
    fig.suptitle(f"Animal {args.animal} — Free Licking data across days", fontsize=12, fontweight='bold', y=0.95)

    # Save figure
    base_dir = Path(r"Z:/dmclab/Joana/Behavior/Data") / args.animal / "Analysis" / "Across-days"
    base_dir.mkdir(parents=True, exist_ok=True)

    fig_filename = base_dir / f"{args.animal}_FreeLick_across_days"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_filename.with_suffix(f".{ext}"), dpi=500)

    print(f"✅ Plot saved to: {fig_filename.with_suffix('.png')}, .pdf, .svg")
    plt.close()

if __name__ == "__main__":
    main()
