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
    match = re.search(r'2ChoiceBlocks_\d+_(\d{8}_\d{6})_box', filename)
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
        files = [f for f in os.listdir(date_path) if f.startswith("2ChoiceBlocks") and f.endswith(".csv")]

        # Extract timestamps and sort
        files_with_time = [(f, extract_timestamp(f)) for f in files if extract_timestamp(f)]
        if len(files_with_time) < 2:
            print(f"⚠️ Skipping {date_path} — found {len(files_with_time)} valid 2ChoiceAuditory files.")
            continue

        # Sort oldest → newest
        files_with_time.sort(key=lambda x: x[1])
        sorted_files = [f[0] for f in files_with_time]

        # Track concatenated dataframe
        concatenated = None
        trial_offset = 0

        try:
            for i, filename in enumerate(sorted_files):
                path = os.path.join(date_path, filename)
                df = pd.read_csv(path)

                # Apply trial offset except for first file
                df["trial_number"] += trial_offset

                # Update offset for next file
                trial_offset += df["trial_number"].max()

                # Concatenate
                if concatenated is None:
                    concatenated = df
                else:
                    concatenated = pd.concat([concatenated, df], ignore_index=True)

            # Move original files into old/
            old_folder = os.path.join(date_path, "old")
            os.makedirs(old_folder, exist_ok=True)

            for filename in sorted_files:
                shutil.move(
                    os.path.join(date_path, filename),
                    os.path.join(old_folder, filename.replace(".csv", "_old.csv"))
                )

            # Save using the name of the oldest file
            final_filename = sorted_files[0]
            final_path = os.path.join(date_path, final_filename)

            concatenated.to_csv(final_path, index=False)

            print(f"✅ Concatenated {len(sorted_files)} files → {final_path}")

        except Exception as e:
            print(f"❌ Error processing {date_path}: {e}")