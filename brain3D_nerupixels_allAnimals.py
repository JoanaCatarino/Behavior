# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:16:32 2026

@author: JoanaCatarino
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

pv.set_jupyter_backend("none")
pv.global_theme.notebook = False

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from brainglobe_atlasapi import BrainGlobeAtlas


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT_PROBE_FOLDER = Path(
    r"L:\dmclab\Joana\PFC-Str_behavior_project\Histology"
)

FILE_PATTERN = "*neuropixels_probe*.csv"

# Probes to exclude from the 3D plot
# Use either full file names or parts of file names
EXCLUDE_PROBES = [
    #"986170_20251201_imec1_str_neuropixels_probe3.csv",
    #"986170_20251211_imec1_str_neuropixels_probe7.csv",
    #"986170_20251125_imec0_str_neuropixels_probe1.csv",
    #"986170_20251125_imec1_pfc_neuropixels_probe0.csv",
    #"986170_20251201_imec0_pfc_neuropixels_probe2.csv",
    #"999770_20251206_imec0_str_neuropixels_probe5.csv",
    #"999770_20251206_imec1_pfc_neuropixels_probe4.csv",

]


ATLAS_NAME = "allen_mouse_10um"
VOXEL_SIZE_UM = 10.0

BRAIN_COLOR = "lightgray"
BRAIN_ALPHA = 0.10

REGION_ALPHA = 0.10

PROBE_LINE_WIDTH = 8
RENDER_PROBES_AS_TUBES = True
PROBE_TUBE_RADIUS_UM = 18

USE_ONLY_INSIDE_BRAIN = True
EXTEND_TRACT_UM = 0.0

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

# Animal ID : experimental group
ANIMAL_GROUPS = {
    "986170": "fezf2",
    "999770": "tlx3",
    "986235": "tlx3",
    "986171": "fezf2",
    "986168": "fezf2",
    
}

GROUP_COLORS = {
    "tlx3": "#DA821D",
    "fezf2": "#B55CB5",
    "fmr1-tlx3": "#2ca02c",
    "fmr1-fezf2": "#d62728",
}


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = ROOT_PROBE_FOLDER / "all_animals_3d_probe_plot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEGEND_PNG = OUTPUT_DIR / "group_legend.png"
LEGEND_CSV = OUTPUT_DIR / "group_legend.csv"

SAVE_SCREENSHOT = False
SCREENSHOT_PATH = OUTPUT_DIR / "all_animals_brain_probes.png"

SAVE_PNG = False
PNG_PATH = OUTPUT_DIR / "all_animals_brain_probes_pub.png"
PNG_SCALE = 4
WINDOW_SIZE = (1800, 1400)

SAVE_VECTOR = False
VECTOR_PATH = OUTPUT_DIR / "all_animals_brain_probes_pub.pdf"

SAVE_MOVIE = True
MOVIE_PATH = OUTPUT_DIR / "all_animals_brain_rotation.mp4"
MOVIE_FRAMES = 180
MOVIE_ORBIT_FACTOR = 1.6
MOVIE_VIEWUP = (0, 0, 1)


# =============================================================================
# HELPERS
# =============================================================================

REQUIRED_COLUMNS = ["ap_coords", "dv_coords", "ml_coords"]


def find_probe_files_for_animals(root_folder: Path, animal_groups: dict):
    records = []

    for animal_id, group in animal_groups.items():
        animal_folder = root_folder / animal_id / "all_probes"

        if not animal_folder.exists():
            print(f"WARNING: folder does not exist for animal {animal_id}: {animal_folder}")
            continue

        probe_files = sorted(animal_folder.glob(FILE_PATTERN))

        if not probe_files:
            print(f"WARNING: no probe files found for animal {animal_id}")
            continue

        for probe_file in probe_files:
            records.append(
                {
                    "animal_id": animal_id,
                    "group": group,
                    "csv_path": probe_file,
                }
            )

    return records


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

    x_um = df["ap_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM
    y_um = df["dv_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM
    z_um = df["ml_coords"].to_numpy(dtype=float) * VOXEL_SIZE_UM

    coords_um = np.column_stack([x_um, y_um, z_um])

    _, unique_idx = np.unique(coords_um, axis=0, return_index=True)
    coords_um = coords_um[np.sort(unique_idx)]

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


def add_surface_mesh(plotter, mesh, color, opacity):
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
    plotter,
    tract,
    color,
    line_width=4,
    as_tube=True,
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


def save_group_legend(group_colors, animal_groups, png_path: Path, csv_path: Path):
    rows = []

    for animal_id, group in animal_groups.items():
        rows.append(
            {
                "animal_id": animal_id,
                "group": group,
                "color_hex": group_colors[group],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    groups = sorted(df["group"].unique())

    handles = [
        Patch(
            facecolor=group_colors[group],
            edgecolor="black",
            label=group,
        )
        for group in groups
    ]

    fig_h = max(2.0, 0.4 * len(groups) + 0.8)
    fig, ax = plt.subplots(figsize=(5, fig_h))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_publication_outputs(plotter, png_path, vector_path, movie_path):
    plotter.reset_camera()
    plotter.show(interactive=True, auto_close=False)

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

            plotter.open_movie(str(movie_path), framerate=30, quality=10)

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
    probe_records = find_probe_files_for_animals(ROOT_PROBE_FOLDER, ANIMAL_GROUPS)

    if not probe_records:
        raise FileNotFoundError("No probe files were found for the selected animals.")

    print(f"Found {len(probe_records)} probe file(s) from several animals.")

    missing_colors = sorted(
        set(ANIMAL_GROUPS.values()) - set(GROUP_COLORS.keys())
    )

    if missing_colors:
        raise ValueError(f"Missing colors for groups: {missing_colors}")

    atlas = BrainGlobeAtlas(ATLAS_NAME)

    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
        off_screen=False,
        notebook=False,
    )

    plotter.set_background("white")
    plotter.image_scale = PNG_SCALE

    root_mesh_path = atlas.root_meshfile()
    print(f"Loading root mesh: {root_mesh_path}")

    root_mesh = load_mesh_clean(root_mesh_path)

    add_surface_mesh(
        plotter=plotter,
        mesh=root_mesh,
        color=BRAIN_COLOR,
        opacity=BRAIN_ALPHA,
    )

    print("\nTrying to add requested regions:")

    added_acronyms = set()

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

            region_mesh = load_mesh_clean(mesh_path)

            add_surface_mesh(
                plotter=plotter,
                mesh=region_mesh,
                color=info["color_float"],
                opacity=REGION_ALPHA,
            )

            added_acronyms.add(acr)

        except Exception as e:
            print(f"Could not render region {acr}: {e}")

    print("\nAdding probes:")

    print("\nAdding probes:")

    for record in probe_records:
        animal_id = record["animal_id"]
        group = record["group"]
        csv_path = record["csv_path"]
    
        # Skip excluded probes
        if any(excluded in csv_path.name for excluded in EXCLUDE_PROBES):
            print(f"SKIPPING excluded probe: {animal_id} | {csv_path.name}")
            continue
    
        color = GROUP_COLORS[group]
    
        print(f"{animal_id} | {group} | {csv_path.name}")
    
        raw_points = load_probe_csv(
            csv_path,
            use_only_inside_brain=USE_ONLY_INSIDE_BRAIN,
        )
    
        tract = make_linear_tract(raw_points, extend_um=EXTEND_TRACT_UM)
    
        add_probe_line(
            plotter=plotter,
            tract=tract,
            color=color,
            line_width=PROBE_LINE_WIDTH,
            as_tube=RENDER_PROBES_AS_TUBES,
        )

    save_group_legend(
        group_colors=GROUP_COLORS,
        animal_groups=ANIMAL_GROUPS,
        png_path=LEGEND_PNG,
        csv_path=LEGEND_CSV,
    )

    print("\nSaved group legend:")
    print(LEGEND_CSV)
    print(LEGEND_PNG)

    plotter.add_axes()
    plotter.reset_camera()
    plotter.show(interactive=True, auto_close=False)

    if SAVE_SCREENSHOT:
        plotter.screenshot(str(SCREENSHOT_PATH))
        print(f"Saved screenshot: {SCREENSHOT_PATH}")
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