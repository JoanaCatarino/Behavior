# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:41:01 2026

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

BASE_HISTO_DIR = Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Histology")

ANIMAL_IDS = [
    "986168",
    # "997770",
    # "999770",
]

FILE_PATTERN = "*neuropixels_probe*.csv"
OUTPUT_FOLDER_NAME = "probe_region_profiles_confidence_width"

ATLAS_NAME = "allen_mouse_10um"
VOXEL_SIZE_UM = 10.0

FIGSIZE = (2.2, 8.0)
DPI = 400
BACKGROUND_COLOR = "white"

Y_LABEL = "Depth from dura (um)"

SAVE_PNG = True
SAVE_SVG = True

USE_ONLY_INSIDE_BRAIN = True
MERGE_ADJACENT_SAME_REGION = True

MIN_LABEL_HEIGHT_UM = 40

TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 8
REGION_TEXT_SIZE = 10

BAR_LEFT = 0.0
BAR_RIGHT = 1.0

LABEL_SIDE = "right"

USE_GLOBAL_Y_LIM = False
GLOBAL_YMAX = None

SHOW_REGION_LABELS = True


# =============================================================================
# CONFIDENCE SETTINGS
# =============================================================================

CONFIDENCE_COLUMN = "distance_to_nearest_structure(um)"

CONFIDENCE_RADIUS_VOXELS = 10
CONFIDENCE_MAX_UM = CONFIDENCE_RADIUS_VOXELS * VOXEL_SIZE_UM

SHOW_CONFIDENCE_AXIS = True
CONFIDENCE_AXIS_LABEL = "Confidence"

BACKGROUND_BAR_COLOR = "#E6E6E6"


# =============================================================================
# COLUMN CANDIDATES
# =============================================================================

DEPTH_CANDIDATES = [
    "depth(um)",
    "depth_um",
    "depth",
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


def get_probe_folder(animal_id: str) -> Path:
    return BASE_HISTO_DIR / animal_id / "all_probes"


def get_output_dir(animal_id: str) -> Path:
    output_dir = BASE_HISTO_DIR / animal_id / "all_probes" / OUTPUT_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_sphere_offsets(radius_voxels: int) -> np.ndarray:
    grid = np.array(
        np.meshgrid(
            np.arange(-radius_voxels, radius_voxels + 1),
            np.arange(-radius_voxels, radius_voxels + 1),
            np.arange(-radius_voxels, radius_voxels + 1),
            indexing="ij",
        )
    ).reshape(3, -1).T

    distances = np.sqrt((grid ** 2).sum(axis=1))
    return grid[distances <= radius_voxels]


def estimate_confidence_from_coords(
    df: pd.DataFrame,
    atlas: BrainGlobeAtlas,
    radius_voxels: int = CONFIDENCE_RADIUS_VOXELS,
    voxel_size_um: float = VOXEL_SIZE_UM,
) -> np.ndarray:

    ap_col = first_existing_column(df, COORD_AP_CANDIDATES)
    dv_col = first_existing_column(df, COORD_DV_CANDIDATES)
    ml_col = first_existing_column(df, COORD_ML_CANDIDATES)

    if ap_col is None or dv_col is None or ml_col is None:
        print("[warning] Coordinate columns not found. Confidence will be skipped.")
        return np.full(len(df), np.nan, dtype=float)

    coords = df[[ap_col, dv_col, ml_col]].to_numpy(dtype=float)
    coords = np.round(coords).astype(int)

    annot = atlas.annotation
    shape = np.array(annot.shape, dtype=int)
    sphere_offsets = make_sphere_offsets(radius_voxels)

    confidence_um = np.full(len(coords), np.nan, dtype=float)

    for i, center in enumerate(coords):
        if np.any(center < 0) or np.any(center >= shape):
            continue

        current_id = annot[center[0], center[1], center[2]]

        sphere_coords = center[None, :] + sphere_offsets

        valid = np.all((sphere_coords >= 0) & (sphere_coords < shape[None, :]), axis=1)
        sphere_coords = sphere_coords[valid]

        sphere_ids = annot[
            sphere_coords[:, 0],
            sphere_coords[:, 1],
            sphere_coords[:, 2],
        ]

        different = sphere_ids != current_id

        if not np.any(different):
            confidence_um[i] = radius_voxels * voxel_size_um
        else:
            different_coords = sphere_coords[different]
            voxel_distances = np.sqrt(((different_coords - center[None, :]) ** 2).sum(axis=1))
            confidence_um[i] = np.min(voxel_distances) * voxel_size_um

    return confidence_um


def confidence_um_to_x(confidence_values: np.ndarray) -> np.ndarray:
    confidence_values = np.asarray(confidence_values, dtype=float)
    confidence_x = confidence_values / CONFIDENCE_MAX_UM
    confidence_x = np.clip(confidence_x, 0, 1)
    return confidence_x


def load_probe_csv(
    csv_path: Path,
    atlas: BrainGlobeAtlas,
    use_only_inside_brain: bool = True,
) -> pd.DataFrame:

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
        raise ValueError(
            f"{csv_path.name} is missing an acronym column. "
            f"Available columns:\n{list(df.columns)}"
        )

    if name_col is None:
        raise ValueError(
            f"{csv_path.name} is missing a name column. "
            f"Available columns:\n{list(df.columns)}"
        )

    if sid_col is None:
        raise ValueError(
            f"{csv_path.name} is missing a structure_id column. "
            f"Available columns:\n{list(df.columns)}"
        )

    if depth_col is None:
        raise ValueError(
            f"{csv_path.name} is missing depth(um). "
            f"Available columns:\n{list(df.columns)}"
        )

    out = pd.DataFrame()
    out["acronym"] = df[acronym_col].astype(str)
    out["name"] = df[name_col].astype(str)
    out["structure_id"] = pd.to_numeric(df[sid_col], errors="coerce")
    out["depth_um"] = pd.to_numeric(df[depth_col], errors="coerce")

    if CONFIDENCE_COLUMN in df.columns:
        out[CONFIDENCE_COLUMN] = pd.to_numeric(df[CONFIDENCE_COLUMN], errors="coerce")
    else:
        print(f"[info] {csv_path.name}: calculating confidence from atlas annotation.")
        out[CONFIDENCE_COLUMN] = estimate_confidence_from_coords(
            df=df,
            atlas=atlas,
            radius_voxels=CONFIDENCE_RADIUS_VOXELS,
            voxel_size_um=VOXEL_SIZE_UM,
        )

    out = out.dropna(subset=["depth_um", "structure_id"]).copy()
    out["structure_id"] = out["structure_id"].astype(int)

    out = out.sort_values("depth_um").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"{csv_path.name} has no usable rows after cleaning.")

    return out


def build_segments(df: pd.DataFrame):
    depths = df["depth_um"].to_numpy(dtype=float)
    acr = df["acronym"].astype(str).to_numpy()
    names = df["name"].astype(str).to_numpy()
    sids = df["structure_id"].to_numpy(dtype=int)

    if CONFIDENCE_COLUMN in df.columns:
        confidence = df[CONFIDENCE_COLUMN].to_numpy(dtype=float)
    else:
        confidence = np.full(len(df), np.nan, dtype=float)

    segments = []
    start_idx = 0

    for i in range(1, len(df)):
        if acr[i] != acr[i - 1]:
            segments.append({
                "depths": depths[start_idx:i],
                "acronym": acr[start_idx],
                "name": names[start_idx],
                "structure_id": int(sids[start_idx]),
                "confidence": confidence[start_idx:i],
            })
            start_idx = i

    segments.append({
        "depths": depths[start_idx:len(df)],
        "acronym": acr[start_idx],
        "name": names[start_idx],
        "structure_id": int(sids[start_idx]),
        "confidence": confidence[start_idx:len(df)],
    })

    if MERGE_ADJACENT_SAME_REGION and len(segments) > 1:
        merged = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]

            if seg["acronym"] == prev["acronym"]:
                prev["depths"] = np.concatenate([prev["depths"], seg["depths"]])
                prev["confidence"] = np.concatenate([prev["confidence"], seg["confidence"]])
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


def draw_confidence_region(ax, y_bounds: np.ndarray, confidence_values: np.ndarray, color: str):
    ax.fill_betweenx(
        y_bounds,
        BAR_LEFT,
        BAR_RIGHT,
        color=BACKGROUND_BAR_COLOR,
        linewidth=0,
        zorder=1,
    )

    if confidence_values is None or len(confidence_values) == 0:
        return

    confidence_values = np.asarray(confidence_values, dtype=float)

    if np.all(np.isnan(confidence_values)):
        return

    if len(confidence_values) == 1:
        confidence_x = np.array([confidence_values[0], confidence_values[0]], dtype=float)
    else:
        confidence_x = np.concatenate([
            confidence_values,
            [confidence_values[-1]],
        ])

    confidence_x = confidence_um_to_x(confidence_x)
    confidence_x = np.nan_to_num(confidence_x, nan=0.0)

    ax.fill_betweenx(
        y_bounds,
        BAR_LEFT,
        confidence_x,
        color=color,
        linewidth=0,
        zorder=2,
    )


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
        confidence_values = seg.get("confidence", None)

        color = atlas_color_map.get(int(sid), "#B0B0B0")
        y_bounds = segment_bounds_from_depths(depths)

        draw_confidence_region(
            ax=ax,
            y_bounds=y_bounds,
            confidence_values=confidence_values,
            color=color,
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

    if SHOW_CONFIDENCE_AXIS:
        ax.set_xlabel(CONFIDENCE_AXIS_LABEL, fontsize=6)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50", "100"], fontsize=6)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(axis="x", labelsize=6, length=2)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=TICK_SIZE)

    ax.set_xlim(-0.15, 1.15)

    ymin = max(0, float(np.nanmin(df["depth_um"])))
    ymax = global_ymax if global_ymax is not None else float(np.nanmax(df["depth_um"])) * 1.02

    ax.set_ylim(ymax, ymin)

    if LABEL_SIDE.lower() == "left":
        plt.subplots_adjust(left=0.42, right=0.88, top=0.95, bottom=0.08)
    else:
        plt.subplots_adjust(left=0.18, right=0.60, top=0.95, bottom=0.08)

    png_path = output_dir / f"{probe_name}_barstyle_confidence_width.png"
    svg_path = output_dir / f"{probe_name}_barstyle_confidence_width.svg"

    if SAVE_PNG:
        fig.savefig(
            png_path,
            dpi=DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    if SAVE_SVG:
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )


    return png_path, svg_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    atlas = BrainGlobeAtlas(ATLAS_NAME)
    atlas_color_map = get_structure_color_map(atlas)

    for animal_id in ANIMAL_IDS:
        probe_folder = get_probe_folder(animal_id)
        output_dir = get_output_dir(animal_id)

        probe_files = sorted(probe_folder.glob(FILE_PATTERN))

        if not probe_files:
            print(f"[warning] No files matching '{FILE_PATTERN}' found for animal {animal_id}:")
            print(probe_folder)
            continue

        print(f"\nAnimal {animal_id}")
        print(f"Found {len(probe_files)} probe file(s).")
        print(f"Saving to: {output_dir}")

        loaded = []

        for csv_path in probe_files:
            print(f"Processing: {csv_path.name}")

            df = load_probe_csv(
                csv_path=csv_path,
                atlas=atlas,
                use_only_inside_brain=USE_ONLY_INSIDE_BRAIN,
            )

            loaded.append((csv_path, df))

        global_ymax = None

        if USE_GLOBAL_Y_LIM:
            global_ymax = compute_global_ymax([df for _, df in loaded])

            if GLOBAL_YMAX is not None:
                global_ymax = GLOBAL_YMAX

            print(f"Using global y max: {global_ymax:.2f}")

        for csv_path, df in loaded:
            probe_name = csv_path.stem

            png_path, svg_path = plot_probe_profile(
                df=df,
                atlas_color_map=atlas_color_map,
                animal_id=animal_id,
                probe_name=probe_name,
                output_dir=output_dir,
                global_ymax=global_ymax,
            )

            if SAVE_PNG:
                print(f"  saved PNG: {png_path}")

            if SAVE_SVG:
                print(f"  saved SVG: {svg_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()