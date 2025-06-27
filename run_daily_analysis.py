# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:25:17 2025

@author: JoanaCatarino
"""

import os
import re
import subprocess
from pathlib import Path

# Base data directory
DATA_DIR = Path(r"L:/dmclab/Joana/Behavior/Data")
ANALYSIS_FOLDER_NAME = "Analysis"

# Map protocol prefix to script
protocol_to_script = {
    'FreeLick': 'analyze_free_licking.py',
    'SpoutSamp': 'analyze_spout_sampling.py',
    '2ChoiceAuditory': 'analyze_2choice_auditory.py',
    'AdaptSensorimotor': 'analyze_adapt_sensorimotor.py',
    #'AdaptSensorimotor_distractor': 'analyze_adapt_sensorimotor_distractor.py'
}

# Regex to parse filenames
filename_regex = re.compile(
    r'(?P<protocol>[^_]+)_(?P<animal>\d+)_(?P<date>\d{8})_\d+_box(?P<box>\w+)',
    re.IGNORECASE
)

def analyze_new_data():
    for animal_dir in DATA_DIR.iterdir():
        if not animal_dir.is_dir():
            continue
        animal_id = animal_dir.name
        behavior_dir = animal_dir / "Behavior"
        analysis_dir = animal_dir / ANALYSIS_FOLDER_NAME
        if not behavior_dir.exists():
            continue
        analysis_dir.mkdir(exist_ok=True)

        for date_dir in behavior_dir.iterdir():
            if not date_dir.is_dir():
                continue
            date_str = date_dir.name
            analysis_subdir = analysis_dir / date_str
            if analysis_subdir.exists():
                continue  # Skip if already analyzed

            data_files = list(date_dir.glob("*_*_*_box*.csv"))
            print(f"🔍 Checking: {date_dir}")
            print(f"📂 Files found: {[f.name for f in data_files]}")

            for file in data_files:
                match = filename_regex.match(file.stem)
                if not match:
                    continue

                protocol_prefix = match.group('protocol')
                animal = match.group('animal')
                date = match.group('date')
                box = match.group('box')

                script_name = protocol_to_script.get(protocol_prefix)
                if not script_name:
                    print(f"⚠️  No analysis script for protocol '{protocol_prefix}' — skipping file: {file.name}")
                    continue

                print(f"✅ Running {script_name} for {file.name}")
                analysis_subdir.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "python", script_name,
                    "--file", str(file),
                    "--animal", animal,
                    "--date", date,
                    "--box", box,
                    "--output", str(analysis_subdir)
                ])
                break  # Only analyze the first matching file per date

if __name__ == "__main__":
    analyze_new_data()