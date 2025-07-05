# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:42:33 2025

@author: JoanaCatarino
"""

import re
import subprocess
from pathlib import Path
from collections import defaultdict

# Base data directory
DATA_DIR = Path(r"L:/dmclab/Joana/Behavior/Data")

# Map protocol prefix to script
protocol_to_script = {
    'FreeLick': 'general_free_licking.py',
    'SpoutSamp': 'general_spout_sampling.py',
    '2ChoiceAuditory': 'general_2choice_auditory.py',
    #'AdaptSensorimotor': 'general_adapt_sensorimotor.py',
    # 'AdaptSensorimotor_distractor': 'analyze_adapt_sensorimotor_distractor.py'
}

# Regex to parse filenames
filename_regex = re.compile(
    r'(?P<protocol>[^_]+)_(?P<animal>\d+)_(?P<date>\d{8})_\d+_box(?P<box>\w+)',
    re.IGNORECASE
)

def analyze_all_animals():
    for animal_dir in DATA_DIR.iterdir():
        if not animal_dir.is_dir() or not animal_dir.name.isdigit():
            continue  # Skip non-animal directories

        animal_id = animal_dir.name
        behavior_dir = animal_dir / "Behavior"
        if not behavior_dir.exists():
            print(f"🚫 No Behavior folder for animal {animal_id}")
            continue

        print(f"\n🐭 Processing animal {animal_id}")

        # Collect CSV files grouped by protocol
        protocol_files = defaultdict(list)

        for date_dir in behavior_dir.iterdir():
            if not date_dir.is_dir():
                continue

            for file in date_dir.glob("*_*_*_box*.csv"):
                match = filename_regex.match(file.stem)
                if not match:
                    continue

                protocol = match.group("protocol")
                date = match.group("date")
                box = match.group("box")

                protocol_files[protocol].append({
                    "path": str(file),
                    "date": date,
                    "box": box
                })

        # Run analysis per protocol
        for protocol, files in protocol_files.items():
            script = protocol_to_script.get(protocol)
            if not script:
                print(f"⚠️  No analysis script for protocol '{protocol}' — skipping.")
                continue
            
            
            uncomment when we finish scripts --> to not overwrite data and analyze all animals everytime
            # Check if already analyzed
            output_folder = DATA_DIR / animal_id / "Analysis" / "Across-days"
            output_plot = output_folder / f"{animal_id}_{protocol}_across_days.png"
            output_csv = output_folder / f"{animal_id}_{protocol}_across_days.csv"

            if output_plot.exists() and output_csv.exists():
                print(f"⏭️  Skipping {protocol} for animal {animal_id} — already analyzed.")
                continue
           

            print(f"📊 Running {script} on {len(files)} files for protocol '{protocol}' for animal {animal_id}")

            subprocess.run([
                "python", script,
                "--animal", animal_id,
                "--files", *[f["path"] for f in files]
            ])

if __name__ == "__main__":
    analyze_all_animals()
