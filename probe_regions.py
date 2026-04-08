# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:06:31 2026

@author: JoanaCatarino
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms
from brainglobe_atlasapi import BrainGlobeAtlas


# =============================================================================
# USER SETTINGS
# =============================================================================

PROBE_FOLDER = Path(r"L:\dmclab\Joana\PFC-Str_behavior_project\Histology\999770\all_probes")
FILE_PATTERN = "*neuropixels_probe*.csv"

OUTPUT_DIR = Path(r"L:\dmclab\Joana\PFC-Str_behavior_project\Behavior\plots\adaptive_task\probe_region_profiles_barstyle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_NAME = "allen_mouse_10um"
VOXEL_SIZE_UM = 10.0

# Figure appearance
FIGSIZE = (2.2, 8.0)
DPI = 400
BACKGROUND_COLOR = "white"

# Labels
Y_LABEL = "Distance along probe (um)"

# Save formats
SAVE_PNG = True
SAVE_SVG = True

# Optional filtering
USE_ONLY_INSIDE_BRAIN = True

# Merge adjacent identical labels
MERGE_ADJACENT_SAME_REGION = True

# Minimum segment height to show text label
MIN_LABEL_HEIGHT_UM = 40

# Font sizes
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 8
REGION_TEXT_SIZE = 10

# Bar appearance
BAR_LEFT = 0.0
BAR_RIGHT = 1.0

# Label side: "left" or "right"
LABEL_SIDE = "left"

# If True, set the top of each probe to 0 um
NORMALIZE_DEPTH_TO_ZERO = True

# Optional shared y-limit across all probes
USE_GLOBAL_Y_LIM = False
GLOBAL_YMAX = None   # e.g. 5000

# Show region labels
SHOW_REGION_LABELS = True

LABEL_SIDE = "right"
# =============================================================================
# COLUMN CANDIDATES
# =============================================================================

DEPTH_CANDIDATES = [
    "depth(um)",
    "depth_um",
    "depth",
    "distance_to_tip(um)",
    "distance_to_tip_um",
]

ACRONYM_CANDIDATES = ["acronym", "structure_acronym"]
NAME_CANDIDATES = ["name", "structure_name"]
STRUCTURE_ID_CANDIDATES = ["structure_id", "id"]

COORD_AP_CANDIDATES = ["ap_coords"]
COORD_DV_CANDIDATES = ["dv_coords"]
COORD_ML_CANDIDATES = ["ml_coords"]


# =============================================================================
# HELPERS
# =============================================================================

def rgb_triplet_to_hex(rgb_triplet):
    r, g, b = [int(v) for v in rgb_triplet]
    return f"#{r:02X}{g:02X}{b:02X}"


def first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_structure_color_map(atlas: BrainGlobeAtlas):
    color_map = {}
    for _, rec in atlas.structures.items():
        sid = rec.get("id", None)
        rgb = rec.get("rgb_triplet", None)
        if sid is not None and rgb is not None:
            color_map[int(sid)] = rgb_triplet_to_hex(rgb)
    return color_map


def compute_depth_from_coords(df: pd.DataFrame) -> np.ndarray:
    """
    Compute distance along the probe from 3D atlas coordinates.
    Uses projection onto the best-fit line through the probe points.
    """
    ap_col = first_existing_column(df, COORD_AP_CANDIDATES)
    dv_col = first_existing_column(df, COORD_DV_CANDIDATES)
    ml_col = first_existing_column(df, COORD_ML_CANDIDATES)

    if ap_col is None or dv_col is None or ml_col is None:
        raise ValueError(
            "No depth column found and 3D coordinate columns are missing. "
            f"Available columns: {list(df.columns)}"
        )

    coords = df[[ap_col, dv_col, ml_col]].to_numpy(dtype=float) * VOXEL_SIZE_UM

    # remove rows with nan coords
    valid = ~np.isnan(coords).any(axis=1)
    coords_valid = coords[valid]

    if len(coords_valid) < 2:
        raise ValueError("Not enough valid coordinate rows to compute distance along probe.")

    # fit best line
    center = coords_valid.mean(axis=0)
    centered = coords_valid - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0] / np.linalg.norm(vh[0])

    # project all valid points onto line
    t = (coords_valid - center) @ direction

    # sort projections so depth runs along the probe
    t = t - np.nanmin(t)

    # place back into full-length array
    depth_um = np.full(len(df), np.nan, dtype=float)
    depth_um[valid] = t

    return depth_um


def load_probe_csv(csv_path: Path, use_only_inside_brain: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()

    if use_only_inside_brain and "inside_brain" in df.columns:
        inside = df["inside_brain"].astype(str).str.lower().isin(["true", "1", "yes"])
        if inside.any():
            df = df.loc[inside].copy()

    acronym_col = first_existing_column(df, ACRONYM_CANDIDATES)
    name_col = first_existing_column(df, NAME_CANDIDATES)
    sid_col = first_existing_column(df, STRUCTURE_ID_CANDIDATES)
    depth_col = first_existing_column(df, DEPTH_CANDIDATES)

    if acronym_col is None:
        raise ValueError(f"{csv_path.name} is missing an acronym column. Available columns:\n{list(df.columns)}")
    if name_col is None:
        raise ValueError(f"{csv_path.name} is missing a name column. Available columns:\n{list(df.columns)}")
    if sid_col is None:
        raise ValueError(f"{csv_path.name} is missing a structure_id column. Available columns:\n{list(df.columns)}")

    out = pd.DataFrame()
    out["acronym"] = df[acronym_col].astype(str)
    out["name"] = df[name_col].astype(str)
    out["structure_id"] = pd.to_numeric(df[sid_col], errors="coerce")

    if depth_col is not None:
        out["depth_um"] = pd.to_numeric(df[depth_col], errors="coerce")
    else:
        print(f"[info] {csv_path.name}: no depth column found, computing distance along probe from coordinates.")
        out["depth_um"] = compute_depth_from_coords(df)

    out = out.dropna(subset=["depth_um", "structure_id"]).copy()
    out["structure_id"] = out["structure_id"].astype(int)

    out = out.sort_values("depth_um").reset_index(drop=True)

    if NORMALIZE_DEPTH_TO_ZERO:
        out["depth_um"] = out["depth_um"] - out["depth_um"].min()

    if out.empty:
        raise ValueError(f"{csv_path.name} has no usable rows after cleaning.")

    return out


def build_segments(df: pd.DataFrame):
    depths = df["depth_um"].to_numpy(dtype=float)
    acr = df["acronym"].astype(str).to_numpy()
    names = df["name"].astype(str).to_numpy()
    sids = df["structure_id"].to_numpy(dtype=int)

    segments = []
    start_idx = 0

    for i in range(1, len(df)):
        if acr[i] != acr[i - 1]:
            segments.append({
                "start_idx": start_idx,
                "end_idx": i - 1,
                "depths": depths[start_idx:i],
                "acronym": acr[start_idx],
                "name": names[start_idx],
                "structure_id": int(sids[start_idx]),
            })
            start_idx = i

    segments.append({
        "start_idx": start_idx,
        "end_idx": len(df) - 1,
        "depths": depths[start_idx:len(df)],
        "acronym": acr[start_idx],
        "name": names[start_idx],
        "structure_id": int(sids[start_idx]),
    })

    if MERGE_ADJACENT_SAME_REGION and len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["acronym"] == prev["acronym"]:
                prev["depths"] = np.concatenate([prev["depths"], seg["depths"]])
                prev["end_idx"] = seg["end_idx"]
            else:
                merged.append(seg)
        segments = merged

    return segments


def segment_bounds_from_depths(depths: np.ndarray):
    depths = np.asarray(depths, dtype=float)

    if len(depths) == 1:
        d = depths[0]
        return np.array([d - 10, d + 10], dtype=float)

    diffs = np.diff(depths)
    nonzero = diffs[diffs != 0]
    fallback = np.nanmedian(nonzero) if len(nonzero) else 20.0
    diffs = np.where(diffs == 0, fallback, diffs)

    bounds = np.empty(len(depths) + 1, dtype=float)
    bounds[1:-1] = (depths[:-1] + depths[1:]) / 2
    bounds[0] = depths[0] - diffs[0] / 2
    bounds[-1] = depths[-1] + diffs[-1] / 2
    return bounds


def compute_global_ymax(all_probe_dfs):
    y_max = 0
    for df in all_probe_dfs:
        if len(df):
            y_max = max(y_max, float(np.nanmax(df["depth_um"].to_numpy(dtype=float))))
    return y_max


def plot_probe_profile(
    df: pd.DataFrame,
    atlas_color_map: dict,
    animal_id: str,
    probe_name: str,
    output_dir: Path,
    global_ymax=None,
):
    segments = build_segments(df)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for seg in segments:
        depths = seg["depths"]
        acronym = seg["acronym"]
        sid = seg["structure_id"]

        color = atlas_color_map.get(int(sid), "#B0B0B0")
        y_bounds = segment_bounds_from_depths(depths)

        ax.fill_betweenx(
            y_bounds,
            BAR_LEFT,
            BAR_RIGHT,
            color=color,
            linewidth=0,
        )

        height = y_bounds[-1] - y_bounds[0]
        y_center = 0.5 * (y_bounds[0] + y_bounds[-1])

        if SHOW_REGION_LABELS and height >= MIN_LABEL_HEIGHT_UM:
            if LABEL_SIDE.lower() == "left":
                ax.text(
                    -0.06,
                    y_center,
                    acronym,
                    transform=text_transform,
                    va="center",
                    ha="right",
                    fontsize=REGION_TEXT_SIZE,
                    color="black",
                )
            else:
                ax.text(
                    1.06,
                    y_center,
                    acronym,
                    transform=text_transform,
                    va="center",
                    ha="left",
                    fontsize=REGION_TEXT_SIZE,
                    color="black",
                )

    probe_id = probe_name.split("_")[-1].replace("probe", "Probe ")

    ax.set_title(f"{animal_id} – {probe_id}", fontsize=TITLE_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE)
    ax.set_xlabel("")

    # remove x-axis completely
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # keep only left y-axis
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.set_xlim(-0.15, 1.15)

    ax.invert_yaxis()

    if global_ymax is not None:
        ax.set_ylim(global_ymax, 0)
    else:
        ymax = float(np.nanmax(df["depth_um"].to_numpy(dtype=float))) * 1.02
        ax.set_ylim(ymax, 0)

    if LABEL_SIDE.lower() == "left":
        plt.subplots_adjust(left=0.42, right=0.88, top=0.95, bottom=0.06)
    else:
        plt.subplots_adjust(left=0.18, right=0.60, top=0.95, bottom=0.06)

    png_path = output_dir / f"{probe_name}_barstyle.png"
    svg_path = output_dir / f"{probe_name}_barstyle.svg"

    if SAVE_PNG:
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    if SAVE_SVG:
        fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())

    

    return png_path, svg_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    probe_files = sorted(PROBE_FOLDER.glob(FILE_PATTERN))
    if not probe_files:
        raise FileNotFoundError(
            f"No files matching '{FILE_PATTERN}' were found in:\n{PROBE_FOLDER}"
        )

    atlas = BrainGlobeAtlas(ATLAS_NAME)
    atlas_color_map = get_structure_color_map(atlas)

    print(f"Found {len(probe_files)} probe file(s).")
    print(f"Saving to: {OUTPUT_DIR}")

    loaded = []
    for csv_path in probe_files:
        print(f"\nProcessing: {csv_path.name}")
        df = load_probe_csv(csv_path, use_only_inside_brain=USE_ONLY_INSIDE_BRAIN)
        loaded.append((csv_path, df))

    global_ymax = None
    if USE_GLOBAL_Y_LIM:
        global_ymax = compute_global_ymax([df for _, df in loaded])
        if GLOBAL_YMAX is not None:
            global_ymax = GLOBAL_YMAX
        print(f"Using global y max: {global_ymax:.2f}")

    for csv_path, df in loaded:
        animal_id = csv_path.stem.split("_")[0]
        probe_name = csv_path.stem

        png_path, svg_path = plot_probe_profile(
            df=df,
            atlas_color_map=atlas_color_map,
            animal_id=animal_id,
            probe_name=probe_name,
            output_dir=OUTPUT_DIR,
            global_ymax=global_ymax,
        )

        if SAVE_PNG:
            print(f"  saved PNG: {png_path}")
        if SAVE_SVG:
            print(f"  saved SVG: {svg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()