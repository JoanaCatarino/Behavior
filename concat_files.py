# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 19:02:36 2025

@author: JoanaCatarino
"""

import os
import re
import shutil
import pandas as pd
from datetime import datetime

# Base directory path
base_dir = r"L:\dmclab\Joana\Behavior\Data"

# Check if folder name is only digits
def is_digit_folder(name):
    return name.isdigit()

# Extract timestamp from filename
def extract_timestamp(filename):
    match = re.search(r'2ChoiceAuditory_\d+_(\d{8}_\d{6})_box', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return None

# Loop through digit-named folders
for subject_folder in os.listdir(base_dir):
    if not is_digit_folder(subject_folder):
        continue

    behavior_path = os.path.join(base_dir, subject_folder, "Behavior")
    if not os.path.isdir(behavior_path):
        continue

    for date_folder in os.listdir(behavior_path):
        date_path = os.path.join(behavior_path, date_folder)
        if not os.path.isdir(date_path):
            continue

        # Get all 2ChoiceAuditory CSV files
        files = [f for f in os.listdir(date_path) if f.startswith("2ChoiceAuditory") and f.endswith(".csv")]

        # Extract timestamps and sort
        files_with_time = [(f, extract_timestamp(f)) for f in files if extract_timestamp(f)]
        if len(files_with_time) != 2:
            print(f"⚠️ Skipping {date_path} — found {len(files_with_time)} valid 2ChoiceAuditory files.")
            continue

        files_with_time.sort(key=lambda x: x[1])  # oldest first
        (older_file, _), (newer_file, _) = files_with_time

        path_old = os.path.join(date_path, older_file)
        path_new = os.path.join(date_path, newer_file)

        try:
            df_old = pd.read_csv(path_old)
            df_new = pd.read_csv(path_new)

            # Update trial numbers in the newer file
            max_trial_old = df_old["trial_number"].max()
            df_new["trial_number"] += max_trial_old

            # Concatenate
            df_concat = pd.concat([df_old, df_new], ignore_index=True)

            # Move original files to old/ subfolder
            old_folder = os.path.join(date_path, "old")
            os.makedirs(old_folder, exist_ok=True)

            shutil.move(path_old, os.path.join(old_folder, older_file.replace(".csv", "_old.csv")))
            shutil.move(path_new, os.path.join(old_folder, newer_file.replace(".csv", "_old.csv")))

            # Save combined file using the older file's name
            final_path = os.path.join(date_path, older_file)
            df_concat.to_csv(final_path, index=False)

            print(f"✅ Concatenated and saved: {final_path}")

        except Exception as e:
            print(f"❌ Error processing {date_path}: {e}")
