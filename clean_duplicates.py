# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 18:07:54 2025

@author: JoanaCatarino
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import shutil

# --- Step 1: File and Save Directory Selection ---
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select CSV file to clean",
    filetypes=[("CSV files", "*.csv")]
)
if not file_path:
    raise Exception("No file selected.")

save_dir = filedialog.askdirectory(
    title="Select folder where modified file should be saved"
)
if not save_dir:
    raise Exception("No save folder selected.")

# --- Step 2: Prepare file paths ---
file_name = os.path.basename(file_path)
base_name, ext = os.path.splitext(file_name)
old_dir = os.path.join(save_dir, "old")
os.makedirs(old_dir, exist_ok=True)

old_backup_path = os.path.join(old_dir, f"{base_name}_old{ext}")
final_path = os.path.join(save_dir, file_name)

# --- Step 3: Load original file ---
df_original = pd.read_csv(file_path)

# --- Step 4: Deduplicate trial_number with custom logic ---
def resolve_duplicates(group):
    # Prefer rewarded trials
    rewarded = group[group["reward"] == 1]
    if not rewarded.empty:
        group = rewarded

    # Drop exact duplicates
    group = group.drop_duplicates()

    # Keep only one (first remaining)
    return group.iloc[[0]]

cleaned_list = [resolve_duplicates(group) for _, group in df_original.groupby("trial_number")]
df_cleaned = pd.concat(cleaned_list, ignore_index=True)

# --- Step 5: Save backup and cleaned file ---
shutil.copy(file_path, old_backup_path)     # Backup original
df_cleaned.to_csv(final_path, index=False)  # Save cleaned version ✅

# --- Step 6: Optional summary ---
print("✅ Cleaning completed.")
print(f"📁 Original file saved as backup in: {old_backup_path}")
print(f"💾 Cleaned file saved to: {final_path}")