# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:11:33 2025

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

def load_lick_counts(file_path):
    df = pd.read_csv(file_path)
    left_licks = df['left_spout'].sum()
    right_licks = df['right_spout'].sum()
    total_licks = df['lick'].sum()
    
    # Get most frequent QW value
    if 'QW' in df.columns and not df['QW'].isna().all():
        qw_value = df['QW'].mode()[0]
    else:
        qw_value = 'NA'
    
    
    return left_licks, right_licks, total_licks, qw_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    summary = []

    for file_path in args.files:
        try:
            date, box = extract_metadata(file_path)
            left, right, total, qw = load_lick_counts(file_path)
            summary.append({
                "date": date,
                "box": box,
                "left_licks": left,
                "right_licks": right,
                "total_licks": total,
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
    lefts = [s["left_licks"] for s in summary]
    rights = [s["right_licks"] for s in summary]
    totals = [s["total_licks"] for s in summary]
    boxes = [s["box"] for s in summary]
    day_labels = [f"Day {i+1}" for i in range(len(dates))]
    qws = [s["QW"] for s in summary]
    x = range(len(dates))

    # Define color map for QWs
    qw_colors = {
        0: "#FFFFFF",
        1: "#FAD4D4",  # light pink
        2: "#FCE5CD",  # light orange
        3: "#FFF2CC",   # light yellow
        'NA': "#F5F5F5"
    }
        
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Add QW background shading
    for i, qw in enumerate(qws):
        color = qw_colors.get(qw, "#F5F5F5")
        axs[0].axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3, zorder=0)
        axs[1].axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3, zorder=0)

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
        
    
    # Get left/right line handles
    line_handles, line_labels = axs[0].get_legend_handles_labels()

    # QW patch handles
    qw_handles = [
        Patch(facecolor=color, edgecolor='k', label=f"QW {qw}")
        for qw, color in qw_colors.items() if qw != 'NA'
    ]

    # Combine both
    combined_handles = line_handles + qw_handles
    axs[0].legend(
        handles=combined_handles,
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=False
    )
    axs[1].legend(frameon=False)

    
    plt.subplots_adjust(hspace=0.5) 
    fig.suptitle(f"Animal {args.animal} — Free Licking data across days", fontsize=12, fontweight='bold', y=0.95)

    # Save figure
    base_dir = Path(r"L:/dmclab/Joana/Behavior/Data") / args.animal / "Analysis" / "Across-days"
    base_dir.mkdir(parents=True, exist_ok=True)

    fig_filename = base_dir / f"{args.animal}_FreeLick_across_days"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(fig_filename.with_suffix(f".{ext}"), dpi=500)
        
    print(f"✅ Plot saved to: {fig_filename.with_suffix('.png')}, .pdf, .svg")
        
    # Export summary data to CSV
    summary_df = pd.DataFrame(summary)
    csv_filename = base_dir / f"{args.animal}_FreeLick_across_days.csv"
    summary_df.to_csv(csv_filename, index=False)
    print(f"✅ Data exported to: {csv_filename}")


if __name__ == "__main__":
    main()
