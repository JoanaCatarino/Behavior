# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:05:15 2025

@author: JoanaCatarino
"""


import os
import re
import glob
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_ROOT     = r"L:\dmclab\Joana\Behavior\Data"          # raw data root
ANALYSIS_ROOT = r"L:\dmclab\Joana\Behavior\Data"          # where figures go
CSV_GLOBS      = ["2ChoiceAuditory*.csv", "2ChoiceBlocks*.csv"]                    # pattern for CSVs
FIG_FORMATS   = ("png", "pdf", "svg")                     # files to write
CLEAN_OLD     = True                                      # remove old 'performance*.*' files first
DATE_REGEX    = re.compile(r"^(\d{4})[-_]?(\d{2})[-_]?(\d{2})$")  # 20250723 / 2025-07-23 / 2025_07_23
BOX_REGEX     = re.compile(r"[Bb]ox[_\-]?([A-Za-z0-9]+)")  # ← extract box number
TONE_MAP_FILE = Path(r"L:\dmclab\Joana\Behavior\Spout-tone map\spout_tone_generator.csv")


def parse_date(folder_name: str) -> datetime | None:
    """Return datetime if folder_name looks like YYYYMMDD / YYYY-MM-DD / YYYY_MM_DD."""
    m = DATE_REGEX.fullmatch(folder_name)
    if not m:
        return None
    y, mth, d = map(int, m.groups())
    return datetime(y, mth, d)


def load_day_csvs(day_dir: str) -> tuple[pd.DataFrame, str | None]:
    files = []
    for pattern in CSV_GLOBS:
        files.extend(glob.glob(os.path.join(day_dir, pattern)))

    if not files:
        return pd.DataFrame(), None

    # Detect box from first matching filename
    box_label = None
    m = BOX_REGEX.search(Path(files[0]).stem)
    if m:
        box_label = f"Box {m.group(1)}"

    dfs = [pd.read_csv(fp) for fp in files]
    return pd.concat(dfs, ignore_index=True), box_label

'''
def day_metrics(df: pd.DataFrame) -> dict | None:
    """Compute percentages for one day. Return None if DF is empty."""
    if df.empty:
        return None
    return {
        "perc_correct":       (df["reward"] == 1).mean() * 100,
        "perc_incorrect":     (df["punishment"] == 1).mean() * 100,
        "perc_correct_left":  ((df["reward"] == 1) & (df["left_spout"]  == 1)).mean() * 100,
        "perc_correct_right": ((df["reward"] == 1) & (df["right_spout"] == 1)).mean() * 100,
    }
'''


def day_metrics(df: pd.DataFrame) -> dict | None:
    """Compute percentages for one day. Return None if DF is empty."""
    if df.empty:
        return None
    return {
        "perc_correct":       (df["reward"] == 1).mean() * 100,
        "perc_incorrect":     (df["punishment"] == 1).mean() * 100,
        "perc_correct_left":  ((df["reward"] == 1) & (df["left_spout"]  == 1)).mean() * 100,
        "perc_correct_right": ((df["reward"] == 1) & (df["right_spout"] == 1)).mean() * 100,
    }

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
        
def load_tone_mapping(animal_id: str | int, csv_path: Path) -> str | None:
    """
    Read tone→spout mapping and return a formatted string.
    Expected columns: 'Animal', '8KHz', '16KHz' (case sensitive per your CSV).
    If not found or file missing, return None.
    """
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    # Normalize to string for safe comparison
    df["Animal"] = df["Animal"].astype(str)
    row = df[df["Animal"] == str(animal_id)]
    if row.empty:
        return None

    row = row.iloc[0]
    pair_8khz  = f"8KHz → {row['8KHz']} spout"
    pair_16khz = f"16KHz → {row['16KHz']} spout"
    return f"Tone–spout mapping: {pair_8khz}, {pair_16khz}"


def process_animal(animal_dir: str) -> None:
    animal_id    = os.path.basename(animal_dir)
    behavior_dir = os.path.join(animal_dir, "Behavior")
    if not os.path.isdir(behavior_dir):
        return

    rows = []
    boxes = []
    for day_name in os.listdir(behavior_dir):
        date_obj = parse_date(day_name)
        if date_obj is None:
            continue
        df_day, box_label = load_day_csvs(os.path.join(behavior_dir, day_name))
        metrics = day_metrics(df_day)
        if metrics is None:
            continue
        metrics["date"] = date_obj
        rows.append(metrics)
        boxes.append(box_label or "Box ?")

    if not rows:
        print(f"{animal_id}: no usable CSVs.")
        return

    perf = pd.DataFrame(rows).set_index("date").sort_index()
    boxes = np.array(boxes)[perf.index.argsort()] 

    # Create 'Day N' labels
    day_labels = [f"Day {i+1}" for i in range(len(perf))]
    x = np.arange(len(perf))
    bw = 0.35
    
    # Load tone mapping
    tone_text = load_tone_mapping(animal_id, TONE_MAP_FILE)
    # Compose suptitle text
    sup_lines = [f"Animal {animal_id} — 2-Choice Auditory across days (N={len(perf)})"]
    if tone_text:
        sup_lines.append(tone_text)
    sup_title = "\n".join(sup_lines)

    # ---------------------- FIGURE --------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), constrained_layout=False, sharex=False, gridspec_kw={"hspace": 0.5}  
    )

    # TOP: Correct vs Incorrect
    bars1a = ax1.bar(x - bw/2, perf["perc_correct"],   width=bw, color="green", label="Correct")
    bars1b = ax1.bar(x + bw/2, perf["perc_incorrect"], width=bw, color="red",   label="Incorrect")

    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title(f"Accuracy across days – animal {animal_id}", pad=25)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels(day_labels, rotation=0)  # show 'Day N' labels on top plot too
    annotate_bars(ax1, bars1a)
    annotate_bars(ax1, bars1b)

    # BOTTOM: Correct‑Left vs Correct‑Right
    bars2a = ax2.bar(x - bw/2, perf["perc_correct_left"],  width=bw, color="#BB5C7A",  label="Correct Left")
    bars2b = ax2.bar(x + bw/2, perf["perc_correct_right"], width=bw, color="#5EA5A3", label="Correct Right")

    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Side-specific accuracy across days", pad=25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels, rotation=0)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", linestyle=":", alpha=0.35)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    annotate_bars(ax2, bars2a)
    annotate_bars(ax2, bars2b)
    
    # ── Box labels above both plots ──────────────────────────
    for idx, box in zip(x, boxes):
        # Position text between the two bars, slightly below 103 %
        for ax in (ax1, ax2):
            ax.text(idx, 103, box, ha="center", va="bottom", fontsize=9)
    
    # Add suptitle
    fig.suptitle(sup_title, fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.85, hspace=0.80)  # more gap from suptitle + between plots

    # ---------------------- SAVE ----------------------------
    analysis_dir = Path(ANALYSIS_ROOT) / animal_id / "Analysis" / "Across-days"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if CLEAN_OLD:
        for old in analysis_dir.glob("performance*.*"):
            try:
                old.unlink()
            except Exception as e:
                print(f"  ! Could not remove {old}: {e}")

    base = analysis_dir / "performance"
    for ext in FIG_FORMATS:
        fig.savefig(f"{base}.{ext}", dpi=300)

    print(f"{animal_id}: saved/overwritten performance.{', '.join(FIG_FORMATS)}")



# ---------------------- MAIN -------------------------------
if __name__ == "__main__":
    for entry in os.listdir(DATA_ROOT):
        path = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(path):
            process_animal(path)

