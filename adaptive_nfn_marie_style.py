# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:12:06 2026

@author: JoanaCatarino
"""

from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

#%% Paths

# ============================================================
# PATHS
# ============================================================

COHORT_DIRS = {
    1: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_1"),
    2: Path(r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_2"),
}

SESSIONS_XLSX = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/all_animals_recorded.xlsx"
)

BEHAVIOR_QC_FILE = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/qc_behavior/behavior_qc_metrics.csv"
)

OUTDIR = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/test"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


#%% Plot settings

# ============================================================
# PLOT SETTINGS
# ============================================================

STRAIN_COLORS = {
    "Tlx3": "#393838",
    "Fezf2": "#828282",
    "Fmr1-Fezf2": "#AB395B",
    "Fmr1-Tlx3": "#87B663",
}

STRAIN_LINE_COLORS = {
    "Tlx3": "#494848",       
    "Fezf2": "#ACA9A9",      
    "Fmr1-Fezf2": "#D17D97", 
    "Fmr1-Tlx3": "#BFD6A8",  
}

STRAIN_LABELS = {
    "Tlx3": r"$\bf{L5\ IT}$ (Tlx3-Cre) n=5",
    "Fezf2": r"$\bf{L5\ PT}$ (Fezf2-CreER) n=4",
    "Fmr1-Fezf2": r"$\bf{Fmr1}$ (x Fezf2-CreER) n=3",
    "Fmr1-Tlx3": r"$\bf{Fmr1}$ (x Tlx3-Cre)",
}

STRAIN_ORDER = ["Tlx3", "Fezf2", "Fmr1-Fezf2", "Fmr1-Tlx3"]

# choose strains to include here
SELECTED_STRAINS = ["Tlx3", "Fezf2", "Fmr1-Fezf2"]

STAGE_ORDER = ["naive", "trained"]
STAGE_LABELS = {
    "naive": "Pre-Adaptive",
    "trained": "Post-Adaptive",
}

plot_config = {
    "dpi": 500,
    "tick_fontsize": 18,
    "label_fontsize": 20,
    "title_fontsize": 20,
    "subtitle_fontsize": 14,
    "top_margin": 0.85,
    "ylabel_pad": 14,
    "legend_fontsize": 14,
    "marker_size": 9,
    "summary_marker_size": 11,
    "capsize": 4,
    "linewidth": 2.0,
    "connection_linewidth": 1.2,
    "box_linewidth": 1.6,

    "stage_gap_single": 1.0,
    "strain_gap_single": 0.18,
    "box_width_single": 0.14,

    "stage_gap_category": 1.3,
    "strain_gap_category": 0.28,
    "box_width_category": 0.22,

    "animal_jitter": 0.012, 
    "category_jitter": 0.012,

    "dot_cluster_sep_single": 0.0,
    "dot_cluster_sep_category": 0.0
}

plt.rcParams.update({
    "axes.facecolor": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "sans-serif",
})

#%% Spout-tone map


# ============================================================
# LOAD SPOUT–TONE MAP
# ============================================================

mapping_file = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Spout-tone map/spout_tone_generator.csv"
)
spout_map_df = pd.read_csv(mapping_file)


def get_tone_map(animal_id):
    row = spout_map_df[spout_map_df["Animal"] == int(animal_id)].iloc[0]
    return {
        "8KHz": row["8KHz"],
        "16KHz": row["16KHz"]
    }

#%% Helpers 

# ============================================================
# HELPERS
# ============================================================

_csv_cache = {}


def read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _csv_cache:
        _csv_cache[path] = pd.read_csv(path)
    return _csv_cache[path]


def sem_safe(x):
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan
    return x.sem()


def save_figure(fig, name, outdir=OUTDIR):
    fig.savefig(
        outdir / f"{name}.pdf",
        dpi=plot_config["dpi"],
        bbox_inches="tight",
        transparent=True
    )
    fig.savefig(
        outdir / f"{name}.svg",
        bbox_inches="tight",
        transparent=True
    )


def extract_datetime_from_filename(filename: str):
    match = re.search(r"AdaptSensorimotor_(\d+)_(\d{8})_(\d{6})", filename)
    if not match:
        return None
    return datetime.strptime(match.group(2) + match.group(3), "%Y%m%d%H%M%S")


def extract_session_date_from_filename(filename: str):
    match = re.search(r"AdaptSensorimotor_(\d+)_(\d{8})_(\d{6})", filename)
    if not match:
        return None
    return match.group(2)


def find_adaptive_files(meta_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    unique_animals = meta_df[["animal", "cohort"]].drop_duplicates()

    for _, row in unique_animals.iterrows():
        animal = str(row["animal"]).strip()
        cohort = int(row["cohort"])

        if cohort not in COHORT_DIRS:
            print(f"Warning: cohort {cohort} not found for animal {animal}")
            continue

        behavior_dir = COHORT_DIRS[cohort] / animal / "Behavior"
        if not behavior_dir.exists():
            print(f"Warning: missing Behavior folder for animal {animal} in cohort {cohort}")
            continue

        for day_folder in sorted([d for d in behavior_dir.iterdir() if d.is_dir()]):
            for csv_path in sorted(day_folder.glob("AdaptSensorimotor*.csv")):
                filename = csv_path.name
                dt = extract_datetime_from_filename(filename)
                session_date = extract_session_date_from_filename(filename)

                if dt is None or session_date is None:
                    print(f"Could not parse filename: {filename}")
                    continue

                records.append({
                    "animal": animal,
                    "cohort": cohort,
                    "session_date": session_date,
                    "datetime": dt,
                    "path": str(csv_path),
                    "filename": filename,
                })

    if not records:
        raise FileNotFoundError("No AdaptSensorimotor csv files were found.")

    return (
        pd.DataFrame(records)
        .sort_values(["animal", "session_date", "datetime"])
        .reset_index(drop=True)
    )


def load_included_sessions(qc_file: Path) -> pd.DataFrame:
    qc_df = pd.read_csv(qc_file, dtype={"animal": str, "session_date": str})
    qc_df.columns = [c.strip() for c in qc_df.columns]

    required_cols = ["animal", "session_date", "include"]
    for col in required_cols:
        if col not in qc_df.columns:
            raise ValueError(f"Missing required column in QC file: {col}")

    qc_df["animal"] = (
        qc_df["animal"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    qc_df["session_date"] = (
        qc_df["session_date"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    included_df = qc_df[qc_df["include"] == True].copy()
    included_df = included_df.drop_duplicates(subset=["animal", "session_date"])
    return included_df[["animal", "session_date"]]


def filter_selected_strains(df, selected_strains):
    return df[df["strain"].isin(selected_strains)].copy()


def get_present_strains(df, selected_strains):
    present = df["strain"].astype(str).unique().tolist()
    return [s for s in STRAIN_ORDER if s in selected_strains and s in present]


def add_strain_legend(fig, df, color_dict):
    present_strains = [s for s in STRAIN_ORDER if s in df["strain"].astype(str).unique()]

    handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=color_dict[s],
            markeredgecolor="none",
            markeredgewidth=0,
            color=color_dict[s],
            label=STRAIN_LABELS.get(s, s),
        )
        for s in present_strains
    ]

    if handles:
        fig.legend(
            handles=handles,
            frameon=False,
            fancybox=False,
            edgecolor="black",
            facecolor=None,
            fontsize=plot_config["legend_fontsize"],
            loc="center left",
            bbox_to_anchor=(1.02, 0.75),
            borderpad=0.8,
            labelspacing=0.4,
            handletextpad=0.5,
        )


def get_stage_centers_single():
    return {
        "naive": 1.0,
        "trained": 1.0 + plot_config["stage_gap_single"],
    }


def get_stage_centers_category(category_order):
    centers = np.arange(len(category_order)) * 3.6
    half_gap = plot_config["stage_gap_category"] / 2
    stage_pos = {
        "naive": centers - half_gap,
        "trained": centers + half_gap,
    }
    return centers, stage_pos


def get_strain_offsets(present_strains, mode="single"):
    if len(present_strains) == 1:
        return {present_strains[0]: 0.0}

    if mode == "single":
        step = plot_config["strain_gap_single"]
    else:
        step = plot_config["strain_gap_category"]

    spread = step * (len(present_strains) - 1)
    offsets = np.linspace(-spread / 2, spread / 2, len(present_strains))
    return {strain: offsets[i] for i, strain in enumerate(present_strains)}


def get_box_and_dot_positions(base_x, strain_offset, mode="single"):
    """
    Put dots directly on top of the box center.
    """
    x = base_x + strain_offset
    return x, x


def add_stage_labels_below_categories(ax, category_order, stage_pos):
    y0, y1 = ax.get_ylim()
    text_y = y0 - 0.065 * (y1 - y0)

    for i in range(len(category_order)):
        ax.text(
            stage_pos["naive"][i],
            text_y,
            "Pre",
            ha="center",
            va="top",
            fontsize=plot_config["tick_fontsize"] - 2,
        )
        ax.text(
            stage_pos["trained"][i],
            text_y,
            "Post",
            ha="center",
            va="top",
            fontsize=plot_config["tick_fontsize"] - 2,
        )
        
def pivot_block_metric_wide(df, value_col, category_col="block_type", prefix=None):
    """
    Converts long block-type metrics into wide format:
    one row per animal/session/stage, one column per block type.
    """
    if df.empty:
        return pd.DataFrame()

    base_cols = ["animal", "session_date", "strain", "stage"]

    wide = (
        df.pivot_table(
            index=base_cols,
            columns=category_col,
            values=value_col,
            aggfunc="first"
        )
        .reset_index()
    )

    wide.columns.name = None

    rename_map = {}
    for col in wide.columns:
        if col in base_cols:
            continue
        if prefix is None:
            rename_map[col] = str(col)
        else:
            rename_map[col] = f"{prefix}_{col}"

    wide = wide.rename(columns=rename_map)
    return wide      

def pivot_two_category_metric_wide(df, value_col, cat1, cat2, prefix):
    """
    Example output columns:
    pe_first_sound, pe_last_sound, etc.
    """
    if df.empty:
        return pd.DataFrame()

    base_cols = ["animal", "session_date", "strain", "stage"]

    df = df.copy()
    df["combined_key"] = df[cat2].astype(str) + "_" + df[cat1].astype(str)

    wide = (
        df.pivot_table(
            index=base_cols,
            columns="combined_key",
            values=value_col,
            aggfunc="first"
        )
        .reset_index()
    )

    wide.columns.name = None
    rename_map = {
        col: f"{prefix}_{col}"
        for col in wide.columns if col not in base_cols
    }
    wide = wide.rename(columns=rename_map)
    return wide


#%% Agregation

# ============================================================
# AGGREGATION
# ============================================================

def aggregate_animal_stage(df, value_col, extra_group_cols=None):
    if extra_group_cols is None:
        extra_group_cols = []

    group_cols = ["animal", "strain", "stage"] + extra_group_cols

    agg = (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(
            mean_value="mean",
            sem_value=sem_safe,
            n_sessions="count",
        )
        .reset_index()
    )
    return agg


def aggregate_strain_stage_from_animals(animal_df, extra_group_cols=None):
    if extra_group_cols is None:
        extra_group_cols = []

    group_cols = ["strain", "stage"] + extra_group_cols

    agg = (
        animal_df.groupby(group_cols, dropna=False)["mean_value"]
        .agg(
            mean_value="mean",
            sem_value=sem_safe,
            n_animals="count",
        )
        .reset_index()
    )
    return agg

#%% Axis styling

# ============================================================
# AXIS STYLING
# ============================================================

def style_axis(ax, ylabel, title, subtitle=None, ylim=None, chance_line=None, ref_line=None):
    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="black", alpha=0.5, linewidth=1.2)

    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7, linewidth=1.2)

    if ylim is not None:
        ax.set_ylim(*ylim)

    full_title = title if subtitle is None else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=plot_config["title_fontsize"], pad=16)
    ax.set_ylabel(
        ylabel,
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    # 🔥 IMPORTANT: remove top/right ticks
    ax.tick_params(
        axis="both",
        labelsize=plot_config["tick_fontsize"],
        direction="in",
        top=False,
        right=False,
        length=6,
        width=1.2,
        pad=8,
    )

    # keep only left/bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(False)


def draw_colored_boxplot(ax, values, position, color, width):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return

    # slightly stronger fill color
    light_color = lighten_color(color, amount=0.2)

    q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    iqr = q3 - q1

    w_low = max(np.min(values), q1 - 1.5 * iqr)
    w_high = min(np.max(values), q3 + 1.5 * iqr)

    cap = width * 0.55
    lw = plot_config["box_linewidth"]

    # Box (no outline)
    ax.fill(
        [position - width / 2, position + width / 2, position + width / 2, position - width / 2],
        [q1, q1, q3, q3],
        facecolor=light_color,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    # Median (white, thicker)
    ax.plot(
        [position - width / 2, position + width / 2],
        [med, med],
        color="white",
        linewidth=lw * 1.4,
        zorder=3,
    )

    # 🔥 Whiskers in SAME COLOR as group
    ax.plot(
        [position, position],
        [w_low, q1],
        color=color,
        linewidth=lw,
        zorder=2,
    )
    ax.plot(
        [position, position],
        [q3, w_high],
        color=color,
        linewidth=lw,
        zorder=2,
    )

    # 🔥 Caps in SAME COLOR as group
    ax.plot(
        [position - cap / 2, position + cap / 2],
        [w_low, w_low],
        color=color,
        linewidth=lw,
        zorder=2,
    )
    ax.plot(
        [position - cap / 2, position + cap / 2],
        [w_high, w_high],
        color=color,
        linewidth=lw,
        zorder=2,
    )

def lighten_color(color, amount=0.35):
    """
    Lightens the given color.
    amount=0 -> original color
    amount=1 -> white
    """
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)

def darken_color(color, amount=0.2):
    """
    Darkens the given color.
    amount=0 → original color
    amount=1 → black
    """
    c = mcolors.to_rgb(color)
    return tuple(x * (1 - amount) for x in c)

#%% Generic plotting

# ============================================================
# GENERIC PLOTTING
# ============================================================

def plot_animal_stage_metric(
    df,
    value_col,
    ylabel,
    title,
    strain_colors,
    selected_strains,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(8.6, 7.2),
    dpi=300,
    subtitle="(animal means, sessions pooled)",
):
    plot_df = df[df["stage"].isin(STAGE_ORDER)].copy()
    plot_df = filter_selected_strains(plot_df, selected_strains)

    if plot_df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    animal_df = aggregate_animal_stage(plot_df, value_col=value_col)
    present_strains = get_present_strains(animal_df, selected_strains)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    stage_centers = get_stage_centers_single()
    strain_offsets = get_strain_offsets(present_strains, mode="single")

    rng = np.random.default_rng(42)
    animal_ids = sorted(animal_df["animal"].astype(str).unique())
    jitter_map = {
        animal: (rng.random() - 0.5) * plot_config["animal_jitter"]
        for animal in animal_ids
    }

    all_x = []

    # boxplots per strain per stage
    for stage in STAGE_ORDER:
        for strain in present_strains:
            vals = animal_df.loc[
                (animal_df["stage"] == stage) & (animal_df["strain"] == strain),
                "mean_value"
            ].dropna().values

            base_x = stage_centers[stage]
            strain_offset = strain_offsets[strain]
            box_x, dot_x = get_box_and_dot_positions(base_x, strain_offset, mode="single")
            all_x.append(box_x)

            draw_colored_boxplot(
                ax=ax,
                values=vals,
                position=box_x,
                color=strain_colors[strain],
                width=plot_config["box_width_single"],
            )

    # connect same animal across stages
    pivot_df = animal_df.pivot_table(
        index=["animal", "strain"],
        columns="stage",
        values="mean_value",
        aggfunc="first"
    ).reset_index()
    
    for _, row in pivot_df.iterrows():
        strain = row["strain"]
        if strain not in strain_offsets:
            continue
    
        if pd.notna(row.get("naive")) and pd.notna(row.get("trained")):
            animal = str(row["animal"])
    
            base_x1 = stage_centers["naive"]
            base_x2 = stage_centers["trained"]
            strain_offset = strain_offsets[strain]
    
            _, dot_x1 = get_box_and_dot_positions(base_x1, strain_offset, mode="single")
            _, dot_x2 = get_box_and_dot_positions(base_x2, strain_offset, mode="single")
    
            x1 = dot_x1 + jitter_map[animal]
            x2 = dot_x2 + jitter_map[animal]
    
            # 🔥 pale line in same color as group
            line_color = STRAIN_LINE_COLORS[strain]

            ax.plot(
                [x1, x2],
                [row["naive"], row["trained"]],
                color=line_color,
                alpha=0.3,
                linewidth=1.8,
                zorder=1,
            )

    # animal dots on top of boxes
    for _, row in animal_df.iterrows():
        animal = str(row["animal"])
        strain = row["strain"]
        if strain not in strain_offsets:
            continue

        base_x = stage_centers[row["stage"]]
        strain_offset = strain_offsets[strain]
        _, dot_x = get_box_and_dot_positions(base_x, strain_offset, mode="single")
        x = dot_x + jitter_map[animal]

        color = darken_color(strain_colors[strain], amount=0.5)  # full strength color

        ax.scatter(
            x,
            row["mean_value"],
            s=60,
            color=color,
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(min(all_x) - 0.30, max(all_x) + 0.30)
    ax.set_xticks([stage_centers["naive"], stage_centers["trained"]])
    ax.set_xticklabels(
        [STAGE_LABELS["naive"], STAGE_LABELS["trained"]],
        fontsize=plot_config["tick_fontsize"],
    )

    style_axis(
        ax,
        ylabel=ylabel,
        title=title,
        subtitle=subtitle,
        ylim=ylim,
        chance_line=chance_line,
        ref_line=ref_line,
    )
    add_strain_legend(fig, animal_df, strain_colors)
    plt.tight_layout()
    fig.subplots_adjust(top=plot_config["top_margin"])
    return fig, ax

def plot_animal_stage_metric_by_category(
    df,
    category_col,
    value_col,
    category_order,
    category_labels,
    ylabel,
    title,
    strain_colors,
    selected_strains,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(11.0, 7.0),
    dpi=300,
    subtitle="(animal means, sessions pooled)",
):
    plot_df = df[df["stage"].isin(STAGE_ORDER)].copy()
    plot_df = filter_selected_strains(plot_df, selected_strains)

    if plot_df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    animal_df = aggregate_animal_stage(
        plot_df,
        value_col=value_col,
        extra_group_cols=[category_col],
    )

    present_strains = get_present_strains(animal_df, selected_strains)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    centers, stage_pos = get_stage_centers_category(category_order)
    cat_to_idx = {cat: i for i, cat in enumerate(category_order)}
    strain_offsets = get_strain_offsets(present_strains, mode="category")

    rng = np.random.default_rng(42)
    animal_ids = sorted(animal_df["animal"].astype(str).unique())
    jitter_map = {
        animal: (rng.random() - 0.5) * plot_config["category_jitter"]
        for animal in animal_ids
    }

    all_x = []

    # boxplots per strain per stage per category
    for cat in category_order:
        idx = cat_to_idx[cat]
        for stage in STAGE_ORDER:
            for strain in present_strains:
                vals = animal_df.loc[
                    (animal_df[category_col] == cat) &
                    (animal_df["stage"] == stage) &
                    (animal_df["strain"] == strain),
                    "mean_value"
                ].dropna().values

                base_x = stage_pos[stage][idx]
                strain_offset = strain_offsets[strain]
                box_x, dot_x = get_box_and_dot_positions(base_x, strain_offset, mode="category")
                all_x.append(box_x)

                draw_colored_boxplot(
                    ax=ax,
                    values=vals,
                    position=box_x,
                    color=strain_colors[strain],
                    width=plot_config["box_width_category"],
                )

    # connect same animal within each category
    for cat in category_order:
        idx = cat_to_idx[cat]
        cat_df = animal_df[animal_df[category_col] == cat].copy()
    
        pivot_df = cat_df.pivot_table(
            index=["animal", "strain"],
            columns="stage",
            values="mean_value",
            aggfunc="first"
        ).reset_index()
    
        for _, row in pivot_df.iterrows():
            strain = row["strain"]
            if strain not in strain_offsets:
                continue
    
            if pd.notna(row.get("naive")) and pd.notna(row.get("trained")):
                animal = str(row["animal"])
    
                base_x1 = stage_pos["naive"][idx]
                base_x2 = stage_pos["trained"][idx]
                strain_offset = strain_offsets[strain]
    
                _, dot_x1 = get_box_and_dot_positions(base_x1, strain_offset, mode="category")
                _, dot_x2 = get_box_and_dot_positions(base_x2, strain_offset, mode="category")
    
                x1 = dot_x1 + jitter_map[animal]
                x2 = dot_x2 + jitter_map[animal]
    
                # 🔥 pale line in same color as group
                line_color = STRAIN_LINE_COLORS[strain]
    
                ax.plot(
                    [x1, x2],
                    [row["naive"], row["trained"]],
                    color=line_color,
                    alpha=0.3,
                    linewidth=1.8,
                    zorder=1,
                )

    # animal dots on top of boxes
    for _, row in animal_df.iterrows():
        cat = row[category_col]
        strain = row["strain"]
        if cat not in cat_to_idx or strain not in strain_offsets:
            continue

        animal = str(row["animal"])
        base_x = stage_pos[row["stage"]][cat_to_idx[cat]]
        strain_offset = strain_offsets[strain]
        _, dot_x = get_box_and_dot_positions(base_x, strain_offset, mode="category")
        x = dot_x + jitter_map[animal]

        color = darken_color(strain_colors[strain], amount=0.5)  # full strength color

        ax.scatter(
            x,
            row["mean_value"],
            s=60,
            color=color,
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(min(all_x) - 0.35, max(all_x) + 0.35)
    ax.set_xticks(centers)
    ax.set_xticklabels(category_labels, fontsize=plot_config["tick_fontsize"])

    style_axis(
        ax,
        ylabel=ylabel,
        title=title,
        subtitle=subtitle,
        ylim=ylim,
        chance_line=chance_line,
        ref_line=ref_line,
    )
    add_stage_labels_below_categories(ax, category_order, stage_pos)
    add_strain_legend(fig, animal_df, strain_colors)
    plt.tight_layout()
    fig.subplots_adjust(top=plot_config["top_margin"])
    return fig, ax


#%% Metrics

# ============================================================
# METRICS
# ============================================================

def compute_session_metrics(row):
    df = read_csv_cached(row["path"]).copy()

    out = {
        "animal": row["animal"],
        "strain": row["strain"],
        "cohort": row["cohort"],
        "session_date": row["session_date"],
        "stage": row["stage"],
        "filename": row["filename"],
    }

    out["total_trials"] = len(df)

    if "block" in df.columns:
        block_counts = {"sound": 0, "action-right": 0, "action-left": 0}
        prev = None

        for b in df["block"]:
            if b != prev:
                if b in block_counts:
                    block_counts[b] += 1
                prev = b

        out["n_blocks"] = sum(block_counts.values())
        out["n_sound_blocks"] = block_counts["sound"]
        out["n_action_right_blocks"] = block_counts["action-right"]
        out["n_action_left_blocks"] = block_counts["action-left"]
    else:
        out["n_blocks"] = np.nan
        out["n_sound_blocks"] = np.nan
        out["n_action_right_blocks"] = np.nan
        out["n_action_left_blocks"] = np.nan

    return out

def build_combined_session_metrics_table(
    metrics_df,
    perf_df,
    percentcorrect_df,
    omissions_df,
    pe_df,
):
    """
    Merge all session-level metrics into one table.
    One row = one animal/session/stage.
    """

    base_cols = ["animal", "session_date", "strain", "stage"]

    combined_df = metrics_df.copy()

    # total performance
    if not perf_df.empty:
        combined_df = combined_df.merge(
            perf_df[base_cols + ["performance"]],
            on=base_cols,
            how="left"
        )

    # percent correct
    if not percentcorrect_df.empty:
        combined_df = combined_df.merge(
            percentcorrect_df[base_cols + ["percentcorrect"]],
            on=base_cols,
            how="left"
        )

    # percent omissions
    if not omissions_df.empty:
        combined_df = combined_df.merge(
            omissions_df[base_cols + ["percent_omissions"]],
            on=base_cols,
            how="left"
        )

    # PE counts
    if not pe_df.empty:
        combined_df = combined_df.merge(
            pe_df[base_cols + ["Action→Sound", "Sound→Action"]],
            on=base_cols,
            how="left"
        )

    return combined_df


def compute_performance_tables(sessions_df):
    performance_records = []
    perf_block_records = []
    block_types = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_session = ["reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()
            n_omissions = df[
                (df["omission"] == 1) & (df["catch_trial"] == 0)
            ].shape[0]

            denom = n_correct + n_incorrect + n_omissions
            performance = (n_correct / denom * 100) if denom > 0 else np.nan

            performance_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "strain": row["strain"],
                "stage": row["stage"],
                "performance": performance,
            })

        needed_block = ["block", "reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]
                if df_blk.empty:
                    continue

                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()
                n_omissions = df_blk[
                    (df_blk["omission"] == 1) & (df_blk["catch_trial"] == 0)
                ].shape[0]

                denom = n_correct + n_incorrect + n_omissions
                perf = (n_correct / denom * 100) if denom > 0 else np.nan

                perf_block_records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk,
                    "performance": perf,
                })

    return pd.DataFrame(performance_records), pd.DataFrame(perf_block_records)


def make_blocktypes_df(metrics_df):
    df_blocktypes = metrics_df[
        [
            "animal",
            "strain",
            "stage",
            "session_date",
            "n_sound_blocks",
            "n_action_right_blocks",
            "n_action_left_blocks",
        ]
    ].copy()

    df_blocktypes = df_blocktypes.rename(columns={
        "n_sound_blocks": "sound",
        "n_action_right_blocks": "action-right",
        "n_action_left_blocks": "action-left",
    })

    return df_blocktypes.melt(
        id_vars=["animal", "strain", "stage", "session_date"],
        value_vars=["sound", "action-right", "action-left"],
        var_name="block",
        value_name="count",
    )

# ============================================================
# TRIALS UNTIL SWITCH PER BLOCK TYPE
# ============================================================

def compute_trials_until_switch_tables(sessions_df):
    print("\n=== Computing trials until switch per block type ===")

    block_order = ["sound", "action-right", "action-left"]
    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        if "block" not in df.columns:
            print(f"⚠ Missing 'block' column in {row['filename']}")
            continue

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        for blk_idx in df["block_index"].unique():
            df_blk = df[df["block_index"] == blk_idx]

            if df_blk.empty:
                continue

            blk_type = df_blk["block"].iloc[0]
            if blk_type not in block_order:
                continue

            n_trials = len(df_blk)

            records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block_type": blk_type,
                "n_trials": n_trials,
            })

    tri_df = pd.DataFrame(records)

    if tri_df.empty:
        print("⚠ No trials-until-switch records computed")
        return pd.DataFrame(), pd.DataFrame()

    session_means_df = (
        tri_df.groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "block_type"]
        )["n_trials"]
        .mean()
        .reset_index()
    )

    print("\n=== TRIALS UNTIL SWITCH TABLE ===")
    print(tri_df.head())

    print("\n=== SESSION MEANS TABLE ===")
    print(session_means_df.head())

    return tri_df, session_means_df


# ============================================================
# PERCENT CORRECT
# ============================================================

def compute_percentcorrect_tables(sessions_df):
    print("\n=== Computing percent correct tables ===")

    percentcorrect_records = []
    percentcorrect_block_records = []

    block_types = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_session = ["reward", "punishment"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()

            denom = n_correct + n_incorrect
            percentcorrect = (n_correct / denom * 100) if denom > 0 else np.nan

            percentcorrect_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "correct": n_correct,
                "incorrect": n_incorrect,
                "percentcorrect": percentcorrect,
            })
        else:
            print(f"⚠ Missing columns for session percent correct in {row['filename']}")

        needed_block = ["block", "reward", "punishment"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]

                if df_blk.empty:
                    continue

                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()

                denom = n_correct + n_incorrect
                percentcorrect_blocks = (n_correct / denom * 100) if denom > 0 else np.nan

                percentcorrect_block_records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk,
                    "percentcorrect": percentcorrect_blocks,
                })
        else:
            print(f"⚠ Missing columns for block percent correct in {row['filename']}")

    percentcorrect_df = pd.DataFrame(percentcorrect_records)
    percentcorrect_block_df = pd.DataFrame(percentcorrect_block_records)

    print("\n=== PERCENT CORRECT TABLE ===")
    print(percentcorrect_df.head())

    print("\n=== PERCENT CORRECT PER BLOCK TYPE ===")
    print(percentcorrect_block_df.head())

    return percentcorrect_df, percentcorrect_block_df


# ============================================================
# PERCENT OMISSIONS
# ============================================================

def compute_percentomissions_tables(sessions_df):
    print("\n=== Computing percent omissions tables ===")

    omission_records = []
    omission_block_records = []

    block_types = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_session = ["reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_session):
            n_correct = (df["reward"] == 1).sum()
            n_incorrect = (df["punishment"] == 1).sum()
            n_omissions = ((df["omission"] == 1) & (df["catch_trial"] == 0)).sum()

            denom = n_correct + n_incorrect + n_omissions
            percent_omissions = (n_omissions / denom * 100) if denom > 0 else np.nan

            omission_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "percent_omissions": percent_omissions,
            })
        else:
            print(f"⚠ Missing columns for session omissions in {row['filename']}")

        needed_block = ["block", "reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]
                if df_blk.empty:
                    continue

                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()
                n_omissions = ((df_blk["omission"] == 1) & (df_blk["catch_trial"] == 0)).sum()

                denom = n_correct + n_incorrect + n_omissions
                percent_omissions = (n_omissions / denom * 100) if denom > 0 else np.nan

                omission_block_records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk,
                    "percent_omissions": percent_omissions,
                })
        else:
            print(f"⚠ Missing columns for block omissions in {row['filename']}")

    omissions_df = pd.DataFrame(omission_records)
    omissions_block_df = pd.DataFrame(omission_block_records)

    print("\n=== PERCENT OMISSIONS TABLE ===")
    print(omissions_df.head())

    print("\n=== PERCENT OMISSIONS PER BLOCK TYPE ===")
    print(omissions_block_df.head())

    return omissions_df, omissions_block_df


# ============================================================
# PERSEVERATIVE ERRORS
# ============================================================

def correct_side(block_type, stim, tone_map):
    if block_type == "sound":
        return tone_map[stim]
    elif block_type == "action-left":
        return "left"
    elif block_type == "action-right":
        return "right"
    return np.nan


def classify_pe_after_switch(row, new_block, prev_block, tone_map):
    if row["omission"] == 1:
        return np.nan

    if "8KHz" in row.index and row["8KHz"] == 1:
        stim = "8KHz"
    elif "16KHz" in row.index and row["16KHz"] == 1:
        stim = "16KHz"
    else:
        return np.nan

    habitual = tone_map[stim]
    new_correct = correct_side(new_block, stim, tone_map)
    prev_correct = correct_side(prev_block, stim, tone_map)

    if row["reward"] == 1:
        return 0

    if row["punishment"] == 1:
        if new_block == "sound":
            return 1 if habitual == prev_correct else 0
        if prev_block == "sound":
            return 1 if habitual != new_correct else 0

    return np.nan


def process_session_pe_counts_fixed(row, window=20):
    try:
        df = read_csv_cached(row["path"]).copy()
    except Exception as e:
        print(f"⚠ Could not read {row['filename']}: {e}")
        return None

    needed_cols = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
    if not needed_cols.issubset(df.columns):
        print(f"⚠ Missing columns for PE analysis in {row['filename']}")
        return None

    tone_map = get_tone_map(row["animal"])

    A2S_count = 0
    S2A_count = 0

    for i in range(1, len(df)):
        prev_block = df.loc[i - 1, "block"]
        new_block = df.loc[i, "block"]

        if prev_block == new_block:
            continue

        if prev_block in ["action-left", "action-right"] and new_block == "sound":
            direction = "Action→Sound"
        elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
            direction = "Sound→Action"
        else:
            continue

        next_trials = df.iloc[i:i + window].copy()

        for _, tr in next_trials.iterrows():
            pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)
            if not np.isnan(pe):
                if direction == "Action→Sound":
                    A2S_count += pe
                else:
                    S2A_count += pe

    return {
        "Action→Sound": A2S_count,
        "Sound→Action": S2A_count,
    }


def compute_pe_tables(sessions_df, window=20):
    print("\n=== Computing perseverative errors around block switches ===")

    pe_records = []

    for _, row in sessions_df.iterrows():
        res = process_session_pe_counts_fixed(row, window=window)
        if res is None:
            continue

        pe_records.append({
            "animal": row["animal"],
            "session_date": row["session_date"],
            "session_id": f"{row['animal']}_{row['session_date']}",
            "strain": row["strain"],
            "stage": row["stage"],
            "Action→Sound": res["Action→Sound"],
            "Sound→Action": res["Sound→Action"],
        })

    pe_df = pd.DataFrame(pe_records)

    if pe_df.empty:
        print("⚠ No PE data available.")
    else:
        print("\n=== PE TABLE ===")
        print(pe_df.head())

    return pe_df


# ============================================================
# PE FIRST VS LAST BLOCKS
# ============================================================

def compute_pe_first_last_blocks(sessions_df, window=20):
    print("\n=== Computing PE first vs last blocks ===")

    records = []
    block_order = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Error loading {row['filename']}: {e}")
            continue

        needed = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
        if not needed.issubset(df.columns):
            continue

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()

        block_seq = (
            df.groupby("block_index")["block"]
            .first()
            .reset_index()
            .rename(columns={"block": "block_type"})
        )

        for blk_type in block_order:
            blk_indices = block_seq.loc[block_seq["block_type"] == blk_type, "block_index"].tolist()
            if len(blk_indices) == 0:
                continue

            first_blocks = blk_indices[:2]
            last_blocks = blk_indices[-2:]

            for period_name, block_list in [("first", first_blocks), ("last", last_blocks)]:
                pe_count = 0

                for blk_idx in block_list:
                    blk_df = df[df["block_index"] == blk_idx]
                    if blk_df.empty:
                        continue

                    start_i = blk_df.index[0]
                    prev_i = start_i - 1
                    if prev_i < 0:
                        continue

                    prev_block = df.loc[prev_i, "block"]
                    new_block = df.loc[start_i, "block"]
                    next_trials = df.loc[start_i:start_i + window - 1]

                    for _, tr in next_trials.iterrows():
                        pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)
                        if not pd.isna(pe):
                            pe_count += pe

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "block_type": blk_type,
                    "period": period_name,
                    "pe_count": pe_count
                })

    out = pd.DataFrame(records)

    print("\n=== PE FIRST/LAST BLOCKS ===")
    print(out.head())

    return out

#%% Plot wrappers

# ============================================================
# PLOT WRAPPERS
# ============================================================

def plot_total_trials_animals(metrics_df):
    return plot_animal_stage_metric(
        df=metrics_df,
        value_col="total_trials",
        ylabel="Trials performed",
        title="Total trials",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(300, 800),
    )



def plot_total_blocks_animals(metrics_df):
    return plot_animal_stage_metric(
        df=metrics_df,
        value_col="n_blocks",
        ylabel="Number of blocks",
        title="Total block count",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 30),
    )


def plot_block_types_animals(metrics_df):
    df_blocktypes = make_blocktypes_df(metrics_df)

    return plot_animal_stage_metric_by_category(
        df=df_blocktypes,
        category_col="block",
        value_col="count",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Blocks per type",
        title="Number of blocks per block type",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 14),
    )


def plot_performance_per_session_animals(perf_df):
    return plot_animal_stage_metric(
        df=perf_df,
        value_col="performance",
        ylabel="Performance (%)",
        title="Performance",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
        chance_line=50,
    )


def plot_performance_per_blocktype_animals(perf_block_df):
    return plot_animal_stage_metric_by_category(
        df=perf_block_df,
        category_col="block_type",
        value_col="performance",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Performance (%)",
        title="Performance per block type",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
        chance_line=50,
    )


def plot_trials_until_switch_blocktype_animals(session_means_df):
    return plot_animal_stage_metric_by_category(
        df=session_means_df,
        category_col="block_type",
        value_col="n_trials",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Trials until switch",
        title="Trials until block switch",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 350),
        ref_line=20,
    )


def plot_percentcorrect_per_session_animals(percentcorrect_df):
    return plot_animal_stage_metric(
        df=percentcorrect_df,
        value_col="percentcorrect",
        ylabel="Correct (%)",
        title="Percentage correct",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
        chance_line=50,
    )


def plot_percentcorrect_per_blocktype_animals(percentcorrect_block_df):
    return plot_animal_stage_metric_by_category(
        df=percentcorrect_block_df,
        category_col="block_type",
        value_col="percentcorrect",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Correct (%)",
        title="Percentage correct per block type",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
        chance_line=50,
    )


def plot_percentomissions_per_session_animals(omissions_df):
    return plot_animal_stage_metric(
        df=omissions_df,
        value_col="percent_omissions",
        ylabel="Omissions (%)",
        title="Percentage of omissions",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
    )


def plot_percentomissions_per_blocktype_animals(omissions_block_df):
    return plot_animal_stage_metric_by_category(
        df=omissions_block_df,
        category_col="block_type",
        value_col="percent_omissions",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Omissions (%)",
        title="Percentage of omissions per block type",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 100),
    )


def plot_pe_triangle_animals(pe_df):
    df_long = pe_df.melt(
        id_vars=["animal", "session_date", "session_id", "strain", "stage"],
        value_vars=["Action→Sound", "Sound→Action"],
        var_name="switch_type",
        value_name="pe_count"
    )

    return plot_animal_stage_metric_by_category(
        df=df_long,
        category_col="switch_type",
        value_col="pe_count",
        category_order=["Action→Sound", "Sound→Action"],
        category_labels=["Action→Sound", "Sound→Action"],
        ylabel="Number of perseverative errors",
        title="Perseverative errors after block switches",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 70),
    )


def plot_pe_first_last_blocks_animals(df_pe_fl):
    period_labels = {"first": "First", "last": "Last"}

    df_plot = df_pe_fl.copy()
    df_plot["period_label"] = df_plot["period"].map(period_labels)

    return plot_animal_stage_metric_by_category(
        df=df_plot,
        category_col="period_label",
        value_col="pe_count",
        category_order=["First", "Last"],
        category_labels=["First", "Last"],
        ylabel="Number of perseverative errors",
        title="Perseverative errors in first vs last blocks",
        strain_colors=STRAIN_COLORS,
        selected_strains=SELECTED_STRAINS,
        ylim=(0, 15),
    )

#%% PE time-course plot

# ============================================================
# PE TIME COURSE AFTER SWITCH
# ============================================================

def compute_pe_timecourse_tables(sessions_df, window=20):
    print("\n=== Computing PE time course after switch ===")

    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        needed_cols = {"block", "reward", "punishment", "omission", "8KHz", "16KHz"}
        if not needed_cols.issubset(df.columns):
            print(f"⚠ Missing columns for PE time course in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        for i in range(1, len(df)):
            prev_block = df.loc[i - 1, "block"]
            new_block = df.loc[i, "block"]

            if prev_block == new_block:
                continue

            if prev_block in ["action-left", "action-right"] and new_block == "sound":
                switch_type = "Action→Sound"
            elif prev_block == "sound" and new_block in ["action-left", "action-right"]:
                switch_type = "Sound→Action"
            else:
                continue

            next_trials = df.iloc[i:i + window].copy()

            for k, (_, tr) in enumerate(next_trials.iterrows(), start=1):
                pe = classify_pe_after_switch(tr, new_block, prev_block, tone_map)

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "switch_type": switch_type,
                    "trial_from_switch": k,
                    "pe": pe,
                })

    pe_timecourse_df = pd.DataFrame(records)

    if pe_timecourse_df.empty:
        print("⚠ No PE time course data computed.")
        return pd.DataFrame(), pd.DataFrame()

    session_timecourse_df = (
        pe_timecourse_df
        .groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "switch_type", "trial_from_switch"],
            dropna=False
        )["pe"]
        .mean()
        .reset_index()
    )

    return pe_timecourse_df, session_timecourse_df


def plot_pe_timecourse(session_timecourse_df, stage, strain_colors):
    df = session_timecourse_df.copy()
    df = filter_selected_strains(df, SELECTED_STRAINS)
    df = df[df["stage"] == stage].copy()

    if df.empty:
        print(f"⚠ No PE time course data for stage: {stage}")
        return None, None

    switch_order = ["Action→Sound", "Sound→Action"]
    present_strains = get_present_strains(df, SELECTED_STRAINS)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        dpi=plot_config["dpi"],
        sharey=True
    )

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df[df["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

        for strain in present_strains:
            d_strain = df_sw[df_sw["strain"] == strain].copy()
            if d_strain.empty:
                continue

            stats_df = (
                d_strain.groupby("trial_from_switch")["pe"]
                .agg(["mean", "sem"])
                .reset_index()
                .sort_values("trial_from_switch")
            )

            x = stats_df["trial_from_switch"].values
            y = stats_df["mean"].values * 100
            sem = stats_df["sem"].fillna(0).values * 100

            ax.plot(
                x,
                y,
                linewidth=2.8,
                color=strain_colors[strain],
                label=strain
            )

            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=strain_colors[strain],
                alpha=0.18
            )

        max_trial = int(df_sw["trial_from_switch"].max())
        ax.set_xlim(1, max_trial)
        ax.set_xticks(range(1, max_trial + 1, 2))
        ax.set_ylim(0, 100)
        ax.set_xlabel(
            "Trial after switch",
            fontsize=plot_config["label_fontsize"],
            labelpad=13
        )
        ax.set_title(
            switch_type,
            fontsize=plot_config["title_fontsize"],
            pad=15
        )
        ax.tick_params(
            labelsize=plot_config["tick_fontsize"],
            top=False,
            right=False
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    axes[0].set_ylabel(
        "Perseverative errors (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        Line2D(
            [0], [0],
            color=strain_colors[strain],
            linewidth=2.8,
            label=strain
        )
        for strain in present_strains
    ]

    fig.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=plot_config["legend_fontsize"],
    )

    fig.suptitle(
        f"PE Time Course After Switch — {STAGE_LABELS[stage]}",
        fontsize=plot_config["title_fontsize"],
        y=1.02
    )

    plt.tight_layout()
    
    return fig, axes

#%% Main 

# ============================================================
# MAIN
# ============================================================

def main():
    meta = pd.read_excel(SESSIONS_XLSX, dtype={"animal": str, "session_date": str})
    meta.columns = [c.strip() for c in meta.columns]

    required_cols = ["animal", "sex", "strain", "cohort", "session_date", "stage"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing required column in Excel file: {col}")

    meta["animal"] = (
        meta["animal"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    meta["session_date"] = (
        meta["session_date"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    meta["stage"] = meta["stage"].astype(str).str.strip().str.lower()
    meta["strain"] = meta["strain"].astype(str).str.strip()
    meta["cohort"] = pd.to_numeric(meta["cohort"], errors="coerce")
    meta = meta[meta["cohort"].notna()].copy()
    meta["cohort"] = meta["cohort"].astype(int)

    meta = meta[meta["stage"].isin(STAGE_ORDER)].copy()
    meta = meta[meta["strain"].isin(SELECTED_STRAINS)].copy()

    included_sessions_df = load_included_sessions(BEHAVIOR_QC_FILE)
    meta = meta.merge(included_sessions_df, on=["animal", "session_date"], how="inner")

    if meta.empty:
        raise ValueError("No sessions left after filtering QC + stage + strain.")

    files_df = find_adaptive_files(meta)

    sessions_df = files_df.merge(
        meta,
        on=["animal", "cohort", "session_date"],
        how="inner",
    )

    if sessions_df.empty:
        raise ValueError("No matching included sessions found between metadata Excel and files.")

    metrics = []
    for _, row in sessions_df.iterrows():
        try:
            metrics.append(compute_session_metrics(row))
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        raise ValueError("No metrics could be computed.")

    perf_df, perf_block_df = compute_performance_tables(sessions_df)
    tri_df, session_means_df = compute_trials_until_switch_tables(sessions_df)
    percentcorrect_df, percentcorrect_block_df = compute_percentcorrect_tables(sessions_df)
    omissions_df, omissions_block_df = compute_percentomissions_tables(sessions_df)
    pe_df = compute_pe_tables(sessions_df, window=20)
    pe_first_last_df = compute_pe_first_last_blocks(sessions_df, window=20)
    pe_timecourse_df, session_pe_timecourse_df = compute_pe_timecourse_tables(sessions_df, window=20)

    combined_metrics_df = build_combined_session_metrics_table(
        metrics_df=metrics_df,
        perf_df=perf_df,
        percentcorrect_df=percentcorrect_df,
        omissions_df=omissions_df,
        pe_df=pe_df,
    )

    # add block-type performance columns
    perf_block_wide = pivot_block_metric_wide(
        perf_block_df,
        value_col="performance",
        category_col="block_type",
        prefix="performance"
    )
    if not perf_block_wide.empty:
        combined_metrics_df = combined_metrics_df.merge(
            perf_block_wide,
            on=["animal", "session_date", "strain", "stage"],
            how="left"
        )

    # add block-type percent correct columns
    percentcorrect_block_wide = pivot_block_metric_wide(
        percentcorrect_block_df,
        value_col="percentcorrect",
        category_col="block_type",
        prefix="percentcorrect"
    )
    if not percentcorrect_block_wide.empty:
        combined_metrics_df = combined_metrics_df.merge(
            percentcorrect_block_wide,
            on=["animal", "session_date", "strain", "stage"],
            how="left"
        )

    # add block-type percent omissions columns
    omissions_block_wide = pivot_block_metric_wide(
        omissions_block_df,
        value_col="percent_omissions",
        category_col="block_type",
        prefix="percentomissions"
    )
    if not omissions_block_wide.empty:
        combined_metrics_df = combined_metrics_df.merge(
            omissions_block_wide,
            on=["animal", "session_date", "strain", "stage"],
            how="left"
        )

    # add trials-until-switch block-type columns
    switch_block_wide = pivot_block_metric_wide(
        session_means_df,
        value_col="n_trials",
        category_col="block_type",
        prefix="trials_until_switch"
    )
    if not switch_block_wide.empty:
        combined_metrics_df = combined_metrics_df.merge(
            switch_block_wide,
            on=["animal", "session_date", "strain", "stage"],
            how="left"
        )

    
    pe_first_last_wide = pivot_two_category_metric_wide(
        pe_first_last_df,
        value_col="pe_count",
        cat1="block_type",
        cat2="period",
        prefix="pe"
    )
    if not pe_first_last_wide.empty:
        combined_metrics_df = combined_metrics_df.merge(
            pe_first_last_wide,
            on=["animal", "session_date", "strain", "stage"],
            how="left"
        )    


    combined_metrics_df.to_csv(
            OUTDIR / "adaptive_sessions_metrics_included.csv",
            index=False
        )


    # Animal-level plots
    fig, _ = plot_total_trials_animals(metrics_df)
    if fig is not None:
        save_figure(fig, "total_trials_per_animal_stage")

    fig, _ = plot_total_blocks_animals(metrics_df)
    if fig is not None:
        save_figure(fig, "total_blocks_per_animal_stage")

    fig, _ = plot_block_types_animals(metrics_df)
    if fig is not None:
        save_figure(fig, "blocktypes_per_animal_stage")

    fig, _ = plot_performance_per_session_animals(perf_df)
    if fig is not None:
        save_figure(fig, "performance_per_animal_stage")

    fig, _ = plot_performance_per_blocktype_animals(perf_block_df)
    if fig is not None:
        save_figure(fig, "performance_per_blocktype_per_animal_stage")
        
    fig, _ = plot_trials_until_switch_blocktype_animals(session_means_df)
    if fig is not None:
        save_figure(fig, "trials_until_switch_per_blocktype_per_animal_stage")

    fig, _ = plot_percentcorrect_per_session_animals(percentcorrect_df)
    if fig is not None:
        save_figure(fig, "percentcorrect_per_animal_stage")

    fig, _ = plot_percentcorrect_per_blocktype_animals(percentcorrect_block_df)
    if fig is not None:
        save_figure(fig, "percentcorrect_per_blocktype_per_animal_stage")

    fig, _ = plot_percentomissions_per_session_animals(omissions_df)
    if fig is not None:
        save_figure(fig, "percentomissions_per_animal_stage")

    fig, _ = plot_percentomissions_per_blocktype_animals(omissions_block_df)
    if fig is not None:
        save_figure(fig, "percentomissions_per_blocktype_per_animal_stage")

    fig, _ = plot_pe_triangle_animals(pe_df)
    if fig is not None:
        save_figure(fig, "pe_switchtype_per_animal_stage")

    fig, _ = plot_pe_first_last_blocks_animals(pe_first_last_df)
    if fig is not None:
        save_figure(fig, "pe_first_last_blocks_per_animal_stage")

    fig, _ = plot_pe_timecourse(session_pe_timecourse_df, stage="naive", strain_colors=STRAIN_COLORS)
    if fig is not None:
        save_figure(fig, "pe_timecourse_after_switch_pre_adaptive")

    fig, _ = plot_pe_timecourse(session_pe_timecourse_df, stage="trained", strain_colors=STRAIN_COLORS)
    if fig is not None:
        save_figure(fig, "pe_timecourse_after_switch_post_adaptive")


    print("Done.")
    return metrics_df, sessions_df, perf_df, perf_block_df


if __name__ == "__main__":
    metrics_df, sessions_df, perf_df, perf_block_df = main()