# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:35:48 2025

@author: JoanaCatarino
"""

import os
import shutil
import re
from collections import defaultdict

# Paths
transfering_folder = r"L:/dmclab/Joana/Behavior/Data/transfering"
data_base_folder = r"L:/dmclab/Joana/Behavior/Data"

# Pattern to extract animal ID and date from the filename
pattern = r"_([0-9]{6})_([0-9]{8})_"

# Gather files grouped by base name (excluding extension)
file_groups = defaultdict(list)
for filename in os.listdir(transfering_folder):
    if filename.endswith(('.csv', '.json')):
        base_name = os.path.splitext(filename)[0]  # without extension
        file_groups[base_name].append(filename)

# Process each group
for base_name, files in file_groups.items():
    match = re.search(pattern, base_name)
    if match:
        animal_id = match.group(1)
        date = match.group(2)

        # Create target folder
        target_folder = os.path.join(data_base_folder, animal_id, "Behavior", date)
        os.makedirs(target_folder, exist_ok=True)

        # Move each file in the group
        for fname in files:
            src = os.path.join(transfering_folder, fname)
            dst = os.path.join(target_folder, fname)
            shutil.move(src, dst)
            print(f"Moved {fname} to {dst}")
    else:
        print(f"Skipping file (no animal ID or date match): {files}")

print("Done!")