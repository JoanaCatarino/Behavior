# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:39:06 2026

@author: JoanaCatarino
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from brainglobe_atlasapi import BrainGlobeAtlas


# =============================================================================
# USER SETTINGS
# =============================================================================

PROBE_FOLDER = Path(r"L:\dmclab\Joana\PFC-Str_behavior_project\Histology\999770\all_probes")
FILE_PATTERN = "*neuropixels_probe*.csv"

ATLAS_NAME = "allen_mouse_10um"
VOXEL_SIZE_UM = 10.0

# Whole brain appearance
BRAIN_COLOR = "lightgray"
BRAIN_ALPHA = 0.10

# Region appearance
REGION_ALPHA = 0.15

# Probe appearance
PROBE_COLOR = "black"
PROBE_LINE_WIDTH = 8
RENDER_PROBES_AS_TUBES = True
PROBE_TUBE_RADIUS_UM = 18

# Probe fitting
USE_ONLY_INSIDE_BRAIN = True
EXTEND_TRACT_UM = 0.0

# Allen acronyms to color
REGIONS_TO_COLOR = [
    "CP",
    "ACB",
    "PL",
    "ILA",
    "MOs",
    "ACA",
    "ORB",
    "AI",
    "FRP",
]

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = Path(r"L:\dmclab\Joana\PFC-Str_behavior_project\Behavior\plots\adaptive_task\lab_meeting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEGEND_PNG = OUTPUT_DIR / "manual_region_legend.png"
LEGEND_CSV = OUTPUT_DIR / "manual_region_legend.csv"

# Simple screenshot
SAVE_SCREENSHOT = False
SCREENSHOT_PATH = OUTPUT_DIR / "brain_regions_probes.png"

# Publication export
SAVE_PNG = True
PNG_PATH = OUTPUT_DIR / "brain_regions_probes_pub.png"
PNG_SCALE = 4
WINDOW_SIZE = (1800, 1400)

SAVE_VECTOR = True
VECTOR_PATH = OUTPUT_DIR / "brain_regions_probes_pub.pdf"  # can also be .pdf

# Rotation video
SAVE_MOVIE = True
MOVIE_PATH = OUTPUT_DIR / "brain_rotation.mp4"
MOVIE_FRAMES = 180
MOVIE_ORBIT_FACTOR = 1.6
MOVIE_VIEWUP = (0, 0, 1)


# =============================================================================
# HELPERS
# =============================================================================

REQUIRED_COLUMNS = ["ap_coords", "dv_coords", "ml_coords"]


def load_probe_csv(csv_path: Path, use_only_inside_brain: bool = True) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    if use_only_inside_brain and "inside_brain" in df.columns:
        inside = df["inside_brain"].astype(str).str.lower().isin(["true", "1", "yes"])
        if inside.any():
            df = df.loc[inside].copy()

    if "distance_to_tip(um)" in df.columns:
        df = df.sort_values("distance_to_tip(um)")
    elif "depth(um)" in df.columns:
        df = df.sort_values("depth(um)")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    # Coordinate convention that matched your rendering
    x_um = df["ap_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM
    y_um = df["dv_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM
    z_um = df["ml_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM

    coords_um = np.column_stack([x_um, y_um, z_um])

    _, unique_idx = np.unique(coords_um, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    coords_um = coords_um[unique_idx]

    if len(coords_um) < 2:
        raise ValueError(f"{csv_path.name} does not contain enough valid points.")

    return coords_um


def fit_best_line_3d(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0] / np.linalg.norm(vh[0])
    t = centered @ direction
    return center, direction, t


def make_linear_tract(points: np.ndarray, extend_um: float = 0.0) -> np.ndarray:
    center, direction, t = fit_best_line_3d(points)
    p0 = center + (t.min() - extend_um) * direction
    p1 = center + (t.max() + extend_um) * direction
    return np.vstack([p0, p1])


def rgb_triplet_to_hex(rgb_triplet):
    r, g, b = [int(v) for v in rgb_triplet]
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_triplet_to_float(rgb_triplet):
    rgb = np.asarray(rgb_triplet, dtype=float) / 255.0
    return tuple(rgb.tolist())


def find_structure_record(atlas: BrainGlobeAtlas, query: str):
    query_low = str(query).strip().lower()
    for _, rec in atlas.structures.items():
        acr = str(rec.get("acronym", "")).strip().lower()
        name = str(rec.get("name", "")).strip().lower()
        if query_low == acr or query_low == name:
            return rec
    return None


def get_region_info(atlas: BrainGlobeAtlas, query: str):
    rec = find_structure_record(atlas, query)
    if rec is None:
        return None

    rgb = rec["rgb_triplet"]
    return {
        "query": query,
        "acronym": rec["acronym"],
        "name": rec["name"],
        "rgb_triplet": rgb,
        "color_float": rgb_triplet_to_float(rgb),
        "color_hex": rgb_triplet_to_hex(rgb),
    }


def load_mesh_clean(mesh_path: Path) -> pv.PolyData:
    mesh = pv.read(str(mesh_path))

    try:
        mesh = mesh.extract_surface()
    except Exception:
        pass

    try:
        mesh = mesh.triangulate()
    except Exception:
        pass

    try:
        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.01)
    except Exception:
        pass

    return mesh


def add_surface_mesh(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    color,
    opacity: float,
):
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=False,
        smooth_shading=True,
        specular=0.05,
        diffuse=0.85,
        ambient=0.15,
    )


def add_probe_line(
    plotter: pv.Plotter,
    tract: np.ndarray,
    color="black",
    line_width: float = 4,
    as_tube: bool = True,
):
    line = pv.Line(tract[0], tract[1])

    if as_tube:
        tube = line.tube(radius=PROBE_TUBE_RADIUS_UM)
        plotter.add_mesh(
            tube,
            color=color,
            opacity=1.0,
            show_edges=False,
            smooth_shading=True,
            specular=0.0,
            diffuse=1.0,
            ambient=0.1,
        )
    else:
        plotter.add_mesh(
            line,
            color=color,
            line_width=line_width,
            render_lines_as_tubes=True,
        )


def save_legend(region_table, png_path: Path, csv_path: Path):
    if not region_table:
        print("No regions were added, so legend is empty.")
        return

    df = (
        pd.DataFrame(region_table)
        .drop_duplicates(subset=["acronym"])
        .sort_values("acronym")
        .reset_index(drop=True)
    )
    df.to_csv(csv_path, index=False)

    fig_h = max(2.0, 0.35 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    handles = [
        Patch(
            facecolor=row["color_hex"],
            edgecolor="black",
            label=f'{row["acronym"]} | {row["name"]}'
        )
        for _, row in df.iterrows()
    ]

    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_publication_outputs(
    plotter: pv.Plotter,
    png_path: Path,
    vector_path: Path,
    movie_path: Path,
):
    plotter.show(auto_close=False)

    if SAVE_PNG:
        plotter.screenshot(str(png_path), scale=PNG_SCALE)
        print(f"Saved PNG: {png_path}")

    if SAVE_VECTOR:
        try:
            plotter.save_graphic(str(vector_path))
            print(f"Saved vector graphic: {vector_path}")
        except Exception as e:
            print(f"Vector export failed: {e}")

    if SAVE_MOVIE:
        try:
            orbit = plotter.generate_orbital_path(
                factor=MOVIE_ORBIT_FACTOR,
                n_points=MOVIE_FRAMES,
                viewup=MOVIE_VIEWUP,
                shift=0.0,
            )
            plotter.open_movie(str(movie_path), quality=10)
            plotter.orbit_on_path(
                orbit,
                write_frames=True,
                viewup=MOVIE_VIEWUP,
                threaded=False,
                progress_bar=True,
            )
            print(f"Saved movie: {movie_path}")
        except Exception as e:
            print(f"Movie export failed: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    probe_files = sorted(PROBE_FOLDER.glob(FILE_PATTERN))
    if not probe_files:
        raise FileNotFoundError(
            f"No files matching '{FILE_PATTERN}' were found in:\n{PROBE_FOLDER}"
        )

    print(f"Found {len(probe_files)} probe file(s).")

    atlas = BrainGlobeAtlas(ATLAS_NAME)

    plotter = pv.Plotter(window_size=WINDOW_SIZE, off_screen=False)
    plotter.set_background("white")
    plotter.image_scale = PNG_SCALE

    # Whole brain mesh
    root_mesh_path = atlas.root_meshfile()
    print(f"Loading root mesh: {root_mesh_path}")
    root_mesh = load_mesh_clean(root_mesh_path)
    add_surface_mesh(
        plotter=plotter,
        mesh=root_mesh,
        color=BRAIN_COLOR,
        opacity=BRAIN_ALPHA,
    )

    # Selected regions
    region_table = []
    added_acronyms = set()

    print("\nTrying to add requested regions:")
    for query in REGIONS_TO_COLOR:
        info = get_region_info(atlas, query)

        if info is None:
            print(f"Could not resolve region: {query}")
            continue

        acr = info["acronym"]
        if acr in added_acronyms:
            continue

        try:
            mesh_path = atlas.meshfile_from_structure(acr)
            print(f"Adding region: {query} -> {acr} ({info['name']})")
            print(f"  mesh: {mesh_path}")

            region_mesh = load_mesh_clean(mesh_path)
            add_surface_mesh(
                plotter=plotter,
                mesh=region_mesh,
                color=info["color_float"],
                opacity=REGION_ALPHA,
            )

            region_table.append(info)
            added_acronyms.add(acr)

        except Exception as e:
            print(f"Could not render region {acr}: {e}")

    # Probes
    for csv_path in probe_files:
        print(f"Loading: {csv_path.name}")
        raw_points = load_probe_csv(
            csv_path,
            use_only_inside_brain=USE_ONLY_INSIDE_BRAIN
        )
        tract = make_linear_tract(raw_points, extend_um=EXTEND_TRACT_UM)
        add_probe_line(
            plotter=plotter,
            tract=tract,
            color=PROBE_COLOR,
            line_width=PROBE_LINE_WIDTH,
            as_tube=RENDER_PROBES_AS_TUBES,
        )

    save_legend(region_table, LEGEND_PNG, LEGEND_CSV)

    print("\nSaved legend:")
    print(LEGEND_CSV)
    print(LEGEND_PNG)

    if SAVE_SCREENSHOT:
        plotter.show(auto_close=False)
        plotter.screenshot(str(SCREENSHOT_PATH))
        print(f"Saved screenshot: {SCREENSHOT_PATH}")
        plotter.close()
    else:
        export_publication_outputs(
            plotter=plotter,
            png_path=PNG_PATH,
            vector_path=VECTOR_PATH,
            movie_path=MOVIE_PATH,
        )
        plotter.close()

    print("\nAll outputs folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()