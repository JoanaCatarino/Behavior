# -*- coding: utf-8 -*-
"""
Created on Sun May 25 12:14:50 2025

@author: JoanaCatarino
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import os

# --- Load data ---
file_path = "Z:/dmclab/Joana/Behavior/Data/925145/Behavior/20250525/2ChoiceAuditory_925145_20250525_112707_box1.csv"
df = pd.read_csv(file_path)
df = df.fillna(0)

# --- Extract info from filename ---
filename = os.path.basename(file_path)
parts = filename.replace(".csv", "").split("_")
protocol_raw = parts[0]
animal = parts[1]
date = parts[2]
box = parts[-1]
protocol_translated = "Two-choice Auditory task"
fig_title = f"{protocol_translated} | Animal: {animal} | Date: {date} | {box}"

# --- Load spout-tone mapping and create subtitle ---
mapping_file_path = "Z:/dmclab/Joana/Behavior/Spout-tone map/spout_tone_generator.csv"
spout_mapping_df = pd.read_csv(mapping_file_path)
animal_id = int(animal)
mapping_row = spout_mapping_df[spout_mapping_df["Animal"] == animal_id].iloc[0]
pair_5khz = f"5KHz → {mapping_row['5KHz']} spout"
pair_10khz = f"10KHz → {mapping_row['10KHz']} spout"
mapping_subtitle = f"Tone-spout mapping: {pair_5khz}, {pair_10khz}"

# --- Add timing and latency calculations ---
df["tone_time"] = df["trial_start"] + 1.2
df["lick_latency"] = df["lick_time"] - df["tone_time"]

# --- Lick separation ---
valid_licks = df[df["lick_latency"].notna()]
left_licks = valid_licks[valid_licks["left_spout"] == 1]
right_licks = valid_licks[valid_licks["right_spout"] == 1]
left_colors = left_licks["reward"].map({1: "green", 0: "red"}) 
right_colors = right_licks["reward"].map({1: "green", 0: "red"})

# --- Trial outcome classification ---
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
min_trial = df_sorted["trial_number"].min()
max_trial = df_sorted["trial_number"].max()

# --- Bar plot values ---
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

# --- Create figure ---
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1.5, 1], hspace=0.5)

# Trial history
ax0 = fig.add_subplot(gs[0, :])
for _, row in df_sorted.iterrows():
    trial = row["trial_number"]
    if row["autom_reward"] == 1:
        ax0.axvspan(trial - 0.5, trial + 0.5, color="purple", alpha=0.1)
    elif row["5KHz"] == 1:
        ax0.axvspan(trial - 0.5, trial + 0.5, color="#BFF9FF", alpha=0.2)
    elif row["10KHz"] == 1:
        ax0.axvspan(trial - 0.5, trial + 0.5, color="#F5A783", alpha=0.2)
for category in categories:
    subset = df_sorted[df_sorted["category"] == category]
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

# Lick latency - Left
ax1 = fig.add_subplot(gs[1, 0])
for _, row in left_licks.iterrows():
    if row["autom_reward"] == 1:
        trial = row["trial_number"]
        ax1.axvspan(trial - 0.5, trial + 0.5, color="purple", alpha=0.1)
ax1.plot(left_licks["trial_number"], left_licks["lick_latency"], color='gray')
ax1.scatter(left_licks["trial_number"], left_licks["lick_latency"], c=left_colors)
ax1.set_title("Lick Latency - Left Spout")
ax1.set_xlabel("Trial Number")
ax1.set_ylabel("Lick Latency (s)")
ax1.set_ylim(bottom=0)
ax1.grid(True)

# Lick latency - Right
ax2 = fig.add_subplot(gs[1, 1])
for _, row in right_licks.iterrows():
    if row["autom_reward"] == 1:
        trial = row["trial_number"]
        ax2.axvspan(trial - 0.5, trial + 0.5, color="purple", alpha=0.1)
ax2.plot(right_licks["trial_number"], right_licks["lick_latency"], color='gray')
ax2.scatter(right_licks["trial_number"], right_licks["lick_latency"], c=right_colors)
ax2.set_title("Lick Latency - Right Spout")
ax2.set_xlabel("Trial Number")
ax2.set_ylabel("Lick Latency (s)")
ax2.set_ylim(bottom=0)
ax2.grid(True)

# Bar plot
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

# Full figure title and subtitle
fig.suptitle(fig_title, fontsize=14, y=1)
plt.figtext(0.5, 0.955, mapping_subtitle, ha='center', fontsize=12)
