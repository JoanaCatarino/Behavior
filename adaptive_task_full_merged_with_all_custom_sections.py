
from pathlib import Path
import re
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

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
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/plots/adaptive_task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)
STATSDIR = OUTDIR / "stats"
STATSDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# COLORS
# ============================================================

STRAIN_COLORS_transp = {
    "Tlx3": "#91D1C6",
    "Fezf2": "#C4889A",
    "Fmr1-Fezf2": "#EEB583",
    "Fmr1-Tlx3": "#A9C893",
}

STRAIN_COLORS = {
    "Tlx3": "#5FC4B3",
    "Fezf2": "#B1536F",
    "Fmr1-Fezf2": "#F19647",
    "Fmr1-Tlx3": "#87B663",
}

STRAIN_ORDER = ["Tlx3", "Fezf2", "Fmr1-Fezf2", "Fmr1-Tlx3"]
STAGE_ORDER = ["naive", "trained"]
STAGE_LABELS = {"naive": "Naive", "trained": "Trained"}

# ============================================================
# GLOBAL PLOT CONFIG
# ============================================================

plot_config = {
    "dpi": 500,
    "tick_fontsize": 12,
    "label_fontsize": 14,
    "title_fontsize": 15,
    "ylabel_pad": 14,
    "subplot_wspace": 0.6,
    "legend_fontsize": 12,
    "legend_title_fontsize": 13,
    "legend_loc": "lower left",
    "legend_bbox": (1.02, 0.0),
}

# ============================================================
# LOAD SPOUT–TONE MAP
# ============================================================

mapping_file = Path(
    r"L:/dmclab/Joana/PFC-Str_behavior_project/Spout-tone map/spout_tone_generator.csv"
)
spout_map_df = pd.read_csv(mapping_file)


def get_tone_map(animal_id):
    row = spout_map_df[spout_map_df["Animal"] == int(animal_id)].iloc[0]
    return {"8KHz": row["8KHz"], "16KHz": row["16KHz"]}


# ============================================================
# HELPERS
# ============================================================

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

    files_df = pd.DataFrame(records)
    return files_df.sort_values(["animal", "session_date", "datetime"]).reset_index(drop=True)


def load_included_sessions(qc_file: Path) -> pd.DataFrame:
    qc_df = pd.read_csv(qc_file, dtype={"animal": str, "session_date": str})
    qc_df.columns = [c.strip() for c in qc_df.columns]

    required_cols = ["animal", "session_date", "include"]
    for col in required_cols:
        if col not in qc_df.columns:
            raise ValueError(f"Missing required column in QC file: {col}")

    qc_df["animal"] = qc_df["animal"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    qc_df["session_date"] = qc_df["session_date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    included_df = qc_df[qc_df["include"] == True].copy()
    included_df = included_df.drop_duplicates(subset=["animal", "session_date"])
    return included_df[["animal", "session_date"]]


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


def get_two_stage_positions(n_groups, group_spacing=2.4, stage_offset=0.32):
    centers = np.arange(n_groups) * group_spacing
    stage_pos = {"naive": centers - stage_offset, "trained": centers + stage_offset}
    return centers, stage_pos


def add_stage_letters(ax, category_order, stage_pos):
    y0, y1 = ax.get_ylim()
    text_y = y0 - 0.06 * (y1 - y0)
    for i in range(len(category_order)):
        ax.text(stage_pos["naive"][i], text_y, "N", ha="center", va="top",
                fontsize=plot_config["tick_fontsize"])
        ax.text(stage_pos["trained"][i], text_y, "T", ha="center", va="top",
                fontsize=plot_config["tick_fontsize"])


def add_strain_legend(fig, df, color_dict, marker="^", linestyle="", linewidth=2.5):
    handles = []
    for strain in STRAIN_ORDER:
        color = color_dict.get(strain)
        if color is None:
            continue
        if strain in df["strain"].astype(str).unique():
            if linestyle == "":
                h = plt.Line2D([0], [0], marker=marker, linestyle="", markersize=9,
                               markerfacecolor=color, markeredgewidth=1.8,
                               markeredgecolor=color, color=color, label=strain)
            else:
                h = plt.Line2D([0], [0], color=color, linewidth=linewidth, label=strain)
            handles.append(h)
    if handles:
        fig.legend(handles=handles, title="Strain", frameon=False)


# ============================================================
# STATS HELPERS
# ============================================================

def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text)).strip("_")


def p_to_stars(p):
    if pd.isna(p):
        return "NA"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def p_to_text(p, prefix="p="):
    if pd.isna(p):
        return "NA"
    return f"{prefix}{p:.3g}"


def add_sig_bracket(ax, x1, x2, y, h, text, fontsize=10):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.3, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def safe_mannwhitney(x, y):
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    if len(x) < 2 or len(y) < 2:
        return {"n1": len(x), "n2": len(y), "U": np.nan, "p": np.nan, "note": "not enough observations"}
    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return {"n1": len(x), "n2": len(y), "U": stat, "p": p, "note": ""}


def safe_kruskal_from_groups(group_dict, group_order):
    used = []
    arrays = []
    for g in group_order:
        vals = pd.Series(group_dict.get(g, [])).dropna().values
        if len(vals) > 0:
            used.append(g)
            arrays.append(vals)
    if len(arrays) < 2:
        return {"H": np.nan, "p": np.nan, "groups_used": used, "note": "fewer than 2 groups with data"}
    stat, p = kruskal(*arrays)
    return {"H": stat, "p": p, "groups_used": used, "note": ""}


def dunn_posthoc_long(df, group_col, value_col, group_order=None, p_adjust="bonferroni"):
    sub = df[[group_col, value_col]].dropna().copy()
    if sub.empty or sub[group_col].nunique() < 2:
        return pd.DataFrame(columns=["group1", "group2", "p_adjusted"])
    mat = sp.posthoc_dunn(sub, val_col=value_col, group_col=group_col, p_adjust=p_adjust)
    if group_order is None:
        groups = list(mat.index)
    else:
        groups = [g for g in group_order if g in mat.index]
    rows = []
    for g1, g2 in itertools.combinations(groups, 2):
        rows.append({"group1": g1, "group2": g2, "p_adjusted": mat.loc[g1, g2]})
    return pd.DataFrame(rows)


def compute_two_stage_stats(df, value_col, stage_col="stage", strain_col="strain", strain_order=None):
    out = {}
    naive = df.loc[df[stage_col] == "naive", value_col]
    trained = df.loc[df[stage_col] == "trained", value_col]
    mw_all = safe_mannwhitney(naive, trained)
    out["overall_naive_vs_trained"] = pd.DataFrame([{
        "comparison": "naive_vs_trained_all",
        "n_naive": mw_all["n1"],
        "n_trained": mw_all["n2"],
        "U": mw_all["U"],
        "p": mw_all["p"],
        "note": mw_all["note"],
    }])

    if strain_order is None:
        strain_order = sorted(df[strain_col].dropna().unique())

    rows = []
    for strain in strain_order:
        sub = df[df[strain_col] == strain]
        res = safe_mannwhitney(
            sub.loc[sub[stage_col] == "naive", value_col],
            sub.loc[sub[stage_col] == "trained", value_col]
        )
        rows.append({
            "strain": strain,
            "n_naive": res["n1"],
            "n_trained": res["n2"],
            "U": res["U"],
            "p_uncorrected": res["p"],
            "note": res["note"],
        })

    per_strain = pd.DataFrame(rows)
    valid = per_strain["p_uncorrected"].notna()
    if valid.any():
        reject, p_corr, _, _ = multipletests(per_strain.loc[valid, "p_uncorrected"], method="fdr_bh")
        per_strain.loc[valid, "p_fdr"] = p_corr
        per_strain.loc[valid, "reject_fdr_0_05"] = reject
    out["per_strain_naive_vs_trained"] = per_strain

    naive_df = df[df[stage_col] == "naive"].copy()
    naive_groups = {s: naive_df.loc[naive_df[strain_col] == s, value_col].dropna().values for s in strain_order}
    kw_naive = safe_kruskal_from_groups(naive_groups, strain_order)
    out["between_strains_naive_kw"] = pd.DataFrame([{
        "stage": "naive", "H": kw_naive["H"], "p": kw_naive["p"],
        "groups_used": ", ".join(kw_naive["groups_used"]), "note": kw_naive["note"]
    }])
    out["between_strains_naive_dunn"] = dunn_posthoc_long(
        naive_df, group_col=strain_col, value_col=value_col, group_order=strain_order
    )

    trained_df = df[df[stage_col] == "trained"].copy()
    trained_groups = {s: trained_df.loc[trained_df[strain_col] == s, value_col].dropna().values for s in strain_order}
    kw_trained = safe_kruskal_from_groups(trained_groups, strain_order)
    out["between_strains_trained_kw"] = pd.DataFrame([{
        "stage": "trained", "H": kw_trained["H"], "p": kw_trained["p"],
        "groups_used": ", ".join(kw_trained["groups_used"]), "note": kw_trained["note"]
    }])
    out["between_strains_trained_dunn"] = dunn_posthoc_long(
        trained_df, group_col=strain_col, value_col=value_col, group_order=strain_order
    )
    return out


def compute_two_stage_stats_by_category(df, category_col, value_col, category_order,
                                        stage_col="stage", strain_col="strain", strain_order=None):
    all_overall_rows = []
    all_per_strain_rows = []
    all_kw_rows = []
    all_dunn_rows = []

    for cat in category_order:
        sub = df[df[category_col] == cat].copy()
        if sub.empty:
            continue
        stats = compute_two_stage_stats(
            sub, value_col=value_col, stage_col=stage_col, strain_col=strain_col, strain_order=strain_order
        )
        tmp = stats["overall_naive_vs_trained"].copy()
        tmp.insert(0, category_col, cat)
        all_overall_rows.append(tmp)

        tmp = stats["per_strain_naive_vs_trained"].copy()
        tmp.insert(0, category_col, cat)
        all_per_strain_rows.append(tmp)

        tmp1 = stats["between_strains_naive_kw"].copy()
        tmp1.insert(0, category_col, cat)
        all_kw_rows.append(tmp1)

        tmp2 = stats["between_strains_trained_kw"].copy()
        tmp2.insert(0, category_col, cat)
        all_kw_rows.append(tmp2)

        d1 = stats["between_strains_naive_dunn"].copy()
        if not d1.empty:
            d1.insert(0, "stage", "naive")
            d1.insert(0, category_col, cat)
            all_dunn_rows.append(d1)

        d2 = stats["between_strains_trained_dunn"].copy()
        if not d2.empty:
            d2.insert(0, "stage", "trained")
            d2.insert(0, category_col, cat)
            all_dunn_rows.append(d2)

    out = {
        "overall_naive_vs_trained_by_category": pd.concat(all_overall_rows, ignore_index=True) if all_overall_rows else pd.DataFrame(),
        "per_strain_naive_vs_trained_by_category": pd.concat(all_per_strain_rows, ignore_index=True) if all_per_strain_rows else pd.DataFrame(),
        "between_strains_kw_by_category": pd.concat(all_kw_rows, ignore_index=True) if all_kw_rows else pd.DataFrame(),
        "between_strains_dunn_by_category": pd.concat(all_dunn_rows, ignore_index=True) if all_dunn_rows else pd.DataFrame(),
    }
    if not out["overall_naive_vs_trained_by_category"].empty:
        valid = out["overall_naive_vs_trained_by_category"]["p"].notna()
        if valid.any():
            reject, p_corr, _, _ = multipletests(
                out["overall_naive_vs_trained_by_category"].loc[valid, "p"], method="fdr_bh"
            )
            out["overall_naive_vs_trained_by_category"].loc[valid, "p_fdr"] = p_corr
            out["overall_naive_vs_trained_by_category"].loc[valid, "reject_fdr_0_05"] = reject
    return out


def init_master_stats_store():
    return []


def add_to_master_stats(master_stats_list, source_title, stats_name, df_stats):
    if master_stats_list is None:
        return
    if isinstance(df_stats, pd.DataFrame) and not df_stats.empty:
        tmp = df_stats.copy()
        tmp.insert(0, "stats_name", stats_name)
        tmp.insert(0, "source_title", source_title)
        master_stats_list.append(tmp)


def save_stats_dict(stats_dict, prefix, master_stats_list=None):
    for name, df_stats in stats_dict.items():
        if isinstance(df_stats, pd.DataFrame) and not df_stats.empty:
            df_stats.to_csv(STATSDIR / f"{sanitize_filename(prefix)}_{name}.csv", index=False)
            add_to_master_stats(master_stats_list, prefix, name, df_stats)

        
        
def append_stats_to_master(master_stats_list, stats_dict, prefix):
    """
    Append every stats dataframe in stats_dict to one master list,
    adding columns that identify where the stats came from.
    """
    if master_stats_list is None:
        return

    for stats_name, df_stats in stats_dict.items():
        if not isinstance(df_stats, pd.DataFrame):
            continue
        if df_stats.empty:
            continue

        tmp = df_stats.copy()
        tmp.insert(0, "stats_name", stats_name)
        tmp.insert(0, "source", prefix)
        master_stats_list.append(tmp)
        


def save_master_stats(master_stats_list, outpath):
    if not master_stats_list:
        print("⚠ No master statistics to save.")
        return

    master_df = pd.concat(master_stats_list, ignore_index=True, sort=False)
    master_df.to_csv(outpath, index=False)
    print(f"Saved master statistics summary to: {outpath}")


# ============================================================
# GENERIC PLOTTING
# ============================================================

def plot_two_stage_single_metric(
    df, value_col, ylabel, title, strain_colors, ylim=None, chance_line=None, ref_line=None,
    figsize=(6, 5), dpi=300, master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats(df=df, value_col=value_col, stage_col="stage", strain_col="strain",
                                    strain_order=STRAIN_ORDER)
    save_stats_dict(stats, prefix=title, master_stats_list=master_stats_list)
    overall_p = stats["overall_naive_vs_trained"]["p"].iloc[0]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x_map = {"naive": 0.85, "trained": 1.15}
    mean_x_map = {"naive": 0.93, "trained": 1.23}

    for _, row in df.iterrows():
        stage = row["stage"]
        jitter = (rng.random() - 0.5) * 0.08
        color = strain_colors.get(row["strain"], "gray")
        ax.scatter(x_map[stage] + jitter, row[value_col], marker="^", s=130,
                   facecolors=color, edgecolors=color, linewidths=1.8, alpha=0.95)

    for stage in STAGE_ORDER:
        d = df.loc[df["stage"] == stage, value_col].dropna()
        if len(d) == 0:
            continue
        ax.errorbar(mean_x_map[stage], d.mean(), yerr=sem_safe(d), fmt="^", markersize=12,
                    color="black", capsize=4, linewidth=2, zorder=5)

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="black", alpha=0.6)
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)

    ax.set_xlim(0.7, 1.32)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1
    bracket_y = y1 - 0.08 * yr
    add_sig_bracket(ax, x_map["naive"], x_map["trained"], bracket_y, 0.03 * yr,
                    f"{p_to_stars(overall_p)}\n{p_to_text(overall_p)}", fontsize=10)

    ax.set_xticks([0.89, 1.19])
    ax.set_xticklabels(["Naive", "Trained"], fontsize=plot_config["label_fontsize"])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(ylabel, fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"])
    ax.set_title(title, fontsize=plot_config["title_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, ax


def plot_two_stage_by_category(
    df, category_col, value_col, category_order, category_labels, ylabel, title, strain_colors,
    ylim=None, chance_line=None, ref_line=None, figsize=(9, 6), dpi=300, master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats_by_category(
        df=df, category_col=category_col, value_col=value_col, category_order=category_order,
        stage_col="stage", strain_col="strain", strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats, prefix=title, master_stats_list=master_stats_list)

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    centers, stage_pos = get_two_stage_positions(len(category_order), group_spacing=2.3, stage_offset=0.28)
    cat_to_idx = {cat: i for i, cat in enumerate(category_order)}

    for _, row in df.iterrows():
        cat = row[category_col]
        stage = row["stage"]
        if cat not in cat_to_idx or stage not in STAGE_ORDER:
            continue
        x0 = stage_pos[stage][cat_to_idx[cat]]
        jitter = (rng.random() - 0.5) * 0.10
        color = strain_colors.get(row["strain"], "gray")
        ax.scatter(x0 + jitter, row[value_col], marker="^", s=130,
                   facecolors=color, edgecolors=color, linewidths=1.8, alpha=0.95)

    for stage in STAGE_ORDER:
        for cat in category_order:
            d = df.loc[(df["stage"] == stage) & (df[category_col] == cat), value_col].dropna()
            if len(d) == 0:
                continue
            ax.errorbar(stage_pos[stage][cat_to_idx[cat]] + 0.12, d.mean(), yerr=sem_safe(d),
                        fmt="^", markersize=12, color="black", capsize=4, linewidth=2, zorder=5)

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="gray", alpha=0.6)
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1
    overall_cat_stats = stats["overall_naive_vs_trained_by_category"]
    for i, cat in enumerate(category_order):
        sub = overall_cat_stats[overall_cat_stats[category_col] == cat]
        if sub.empty:
            continue
        p = sub["p_fdr"].iloc[0] if "p_fdr" in sub.columns and pd.notna(sub["p_fdr"].iloc[0]) else sub["p"].iloc[0]
        y = y1 - (0.10 + i * 0.03) * yr
        add_sig_bracket(ax, stage_pos["naive"][i], stage_pos["trained"][i], y, 0.02 * yr,
                        f"{p_to_stars(p)}\n{p_to_text(p)}", fontsize=9)

    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(category_labels, fontsize=plot_config["label_fontsize"])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(ylabel, fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"])
    ax.set_title(title, fontsize=plot_config["title_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_stage_letters(ax, category_order, stage_pos)
    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, ax


def plot_two_stage_single_metric_by_strain(
    df, value_col, ylabel, title, strain_colors, ylim=None, chance_line=None, ref_line=None,
    figsize=(8, 5), dpi=300, master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats(df=df, value_col=value_col, stage_col="stage", strain_col="strain",
                                    strain_order=STRAIN_ORDER)
    save_stats_dict(stats, prefix=f"{title}_by_strain", master_stats_list=master_stats_list)
    naive_kw_p = stats["between_strains_naive_kw"]["p"].iloc[0]
    trained_kw_p = stats["between_strains_trained_kw"]["p"].iloc[0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_map = {"naive": 0.85, "trained": 1.15}
    offsets = np.linspace(-0.15, 0.15, len(STRAIN_ORDER))
    present_strains = [s for s in STRAIN_ORDER if s in df["strain"].astype(str).unique()]

    for i, strain in enumerate(STRAIN_ORDER):
        if strain not in present_strains:
            continue
        color = strain_colors.get(strain, "gray")
        offset = offsets[i]
        for stage in STAGE_ORDER:
            d = df.loc[(df["stage"] == stage) & (df["strain"] == strain), value_col].dropna()
            if len(d) == 0:
                continue
            ax.errorbar(x_map[stage] + offset, d.mean(), yerr=sem_safe(d), fmt="^", markersize=12,
                        color=color, markerfacecolor=color, markeredgecolor=color, capsize=4, linewidth=2)

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="black", alpha=0.6)
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)
    ax.set_xlim(0.55, 1.45)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1
    add_sig_bracket(ax, x_map["naive"] - 0.18, x_map["naive"] + 0.18, y1 - 0.10 * yr, 0.02 * yr,
                    f"KW {p_to_stars(naive_kw_p)}\n{p_to_text(naive_kw_p)}", fontsize=9)
    add_sig_bracket(ax, x_map["trained"] - 0.18, x_map["trained"] + 0.18, y1 - 0.22 * yr, 0.02 * yr,
                    f"KW {p_to_stars(trained_kw_p)}\n{p_to_text(trained_kw_p)}", fontsize=9)

    ax.set_xticks([0.85, 1.15])
    ax.set_xticklabels(["Naive", "Trained"], fontsize=plot_config["label_fontsize"])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(ylabel, fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"])
    ax.set_title(f"{title} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, ax


def plot_two_stage_by_category_by_strain(
    df, category_col, value_col, category_order, category_labels, ylabel, title, strain_colors,
    ylim=None, chance_line=None, ref_line=None, figsize=(10, 6), dpi=300, master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats_by_category(
        df=df, category_col=category_col, value_col=value_col, category_order=category_order,
        stage_col="stage", strain_col="strain", strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats, prefix=f"{title}_by_strain", master_stats_list=master_stats_list)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    centers, stage_pos = get_two_stage_positions(len(category_order), group_spacing=2.4, stage_offset=0.30)
    cat_to_idx = {cat: i for i, cat in enumerate(category_order)}
    strain_offsets = np.linspace(-0.15, 0.15, len(STRAIN_ORDER))

    for s_idx, strain in enumerate(STRAIN_ORDER):
        if strain not in df["strain"].astype(str).unique():
            continue
        color = strain_colors.get(strain, "gray")
        strain_offset = strain_offsets[s_idx]
        for stage in STAGE_ORDER:
            for cat in category_order:
                d = df.loc[(df["stage"] == stage) & (df["strain"] == strain) & (df[category_col] == cat),
                           value_col].dropna()
                if len(d) == 0:
                    continue
                x = stage_pos[stage][cat_to_idx[cat]] + strain_offset
                ax.errorbar(x, d.mean(), yerr=sem_safe(d), fmt="^", markersize=11,
                            color=color, markerfacecolor=color, markeredgecolor=color, capsize=4, linewidth=2)

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="gray", alpha=0.6)
    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)
    if ylim is not None:
        ax.set_ylim(*ylim)

    kw_df = stats["between_strains_kw_by_category"]
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1
    for i, cat in enumerate(category_order):
        sub_naive = kw_df[(kw_df[category_col] == cat) & (kw_df["stage"] == "naive")]
        sub_trained = kw_df[(kw_df[category_col] == cat) & (kw_df["stage"] == "trained")]
        if not sub_naive.empty:
            p = sub_naive["p"].iloc[0]
            add_sig_bracket(ax, stage_pos["naive"][i] - 0.18, stage_pos["naive"][i] + 0.18,
                            y1 - (0.10 + i * 0.05) * yr, 0.015 * yr, f"N: {p_to_stars(p)}", fontsize=8)
        if not sub_trained.empty:
            p = sub_trained["p"].iloc[0]
            add_sig_bracket(ax, stage_pos["trained"][i] - 0.18, stage_pos["trained"][i] + 0.18,
                            y1 - (0.20 + i * 0.05) * yr, 0.015 * yr, f"T: {p_to_stars(p)}", fontsize=8)

    ax.set_xlim(centers[0] - 1.0, centers[-1] + 1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels(category_labels, fontsize=plot_config["label_fontsize"])
    ax.tick_params(labelsize=plot_config["tick_fontsize"])
    ax.set_ylabel(ylabel, fontsize=plot_config["label_fontsize"], labelpad=plot_config["ylabel_pad"])
    ax.set_title(f"{title} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_stage_letters(ax, category_order, stage_pos)
    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, ax

# ============================================================
# METRICS
# ============================================================

def compute_session_metrics(row):
    df = read_csv_cached(row["path"]).copy()
    out = {
        "animal": row["animal"], "sex": row["sex"], "strain": row["strain"], "cohort": row["cohort"],
        "session_date": row["session_date"], "stage": row["stage"], "filename": row["filename"],
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

        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()
        trials_per_block = df.groupby(["block_index", "block"]).size().reset_index(name="n_trials")
        out["mean_trials_until_switch"] = trials_per_block["n_trials"].mean()
    else:
        out["n_blocks"] = np.nan
        out["n_sound_blocks"] = np.nan
        out["n_action_right_blocks"] = np.nan
        out["n_action_left_blocks"] = np.nan
        out["mean_trials_until_switch"] = np.nan

    if {"lick_time", "RW_start"}.issubset(df.columns):
        out["mean_latency"] = (df["lick_time"] - df["RW_start"]).mean()
    else:
        out["mean_latency"] = np.nan

    if {"omission", "catch_trial"}.issubset(df.columns):
        out["n_omissions"] = int(((df["omission"] == 1) & (df["catch_trial"] == 0)).sum())
    else:
        out["n_omissions"] = np.nan
    return out

# ============================================================
# TRIANGLE SUMMARY
# ============================================================

def make_blocktypes_df(stage_df):
    df_blocktypes = stage_df[
        ["animal", "strain", "stage", "session_date", "n_sound_blocks", "n_action_right_blocks", "n_action_left_blocks"]
    ].copy()
    df_blocktypes = df_blocktypes.rename(columns={
        "n_sound_blocks": "sound", "n_action_right_blocks": "action-right", "n_action_left_blocks": "action-left",
    })
    return df_blocktypes.melt(
        id_vars=["animal", "strain", "stage", "session_date"],
        value_vars=["sound", "action-right", "action-left"],
        var_name="block", value_name="count"
    )


# NOTE:
# The custom plotting functions below are preserved from the original workflow.
# The generic plotting functions above now save all statistics automatically.
# To keep this full script manageable and stable, the custom figure functions are left
# without additional statistical annotations unless they already use the generic functions.

def plot_triangle_summary_combined(metrics_df, strain_colors, master_stats_list=None):
    df = metrics_df[metrics_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No data for triangle summary")
        return None, None

    df_blocktypes = make_blocktypes_df(df)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=plot_config["dpi"])
    plt.subplots_adjust(wspace=plot_config["subplot_wspace"])

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------
    stats_total_trials = compute_two_stage_stats(
        df=df,
        value_col="total_trials",
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_total_trials, prefix="triangle_summary_total_trials")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_total_trials, "triangle_summary_total_trials")

    stats_n_blocks = compute_two_stage_stats(
        df=df,
        value_col="n_blocks",
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_n_blocks, prefix="triangle_summary_total_block_count")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_n_blocks, "triangle_summary_total_block_count")

    stats_blocktypes = compute_two_stage_stats_by_category(
        df=df_blocktypes,
        category_col="block",
        value_col="count",
        category_order=["sound", "action-right", "action-left"],
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_blocktypes, prefix="triangle_summary_block_types")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_blocktypes, "triangle_summary_block_types")

    # ========================================================
    # PANEL 1: total trials
    # ========================================================
    ax = axes[0]
    x_map = {"naive": 0.85, "trained": 1.15}
    mean_x_map = {"naive": 0.93, "trained": 1.23}

    for _, row in df.iterrows():
        jitter = (rng.random() - 0.5) * 0.08
        color = strain_colors.get(row["strain"], "#333333")
        ax.scatter(
            x_map[row["stage"]] + jitter,
            row["total_trials"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    for stage in STAGE_ORDER:
        d = df.loc[df["stage"] == stage, "total_trials"].dropna()
        if len(d):
            ax.errorbar(
                mean_x_map[stage], d.mean(), yerr=sem_safe(d),
                fmt="^", markersize=12, color="black", capsize=4
            )

    ax.set_xlim(0.7, 1.32)
    ax.set_ylim(500, 800)
    ax.set_xticks([0.89, 1.19])
    ax.set_xticklabels(["Naive", "Trained"])
    ax.set_ylabel("Trials performed", fontsize=plot_config["label_fontsize"],
                  labelpad=plot_config["ylabel_pad"])
    ax.set_title("Total trials")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p_trials = stats_total_trials["overall_naive_vs_trained"]["p"].iloc[0]
    add_sig_bracket(ax, 0.85, 1.15, 770, 8, f"{p_to_stars(p_trials)}\n{p_to_text(p_trials)}", fontsize=10)

    # ========================================================
    # PANEL 2: total blocks
    # ========================================================
    ax = axes[1]
    for _, row in df.iterrows():
        jitter = (rng.random() - 0.5) * 0.08
        color = strain_colors.get(row["strain"], "#333333")
        ax.scatter(
            x_map[row["stage"]] + jitter,
            row["n_blocks"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    for stage in STAGE_ORDER:
        d = df.loc[df["stage"] == stage, "n_blocks"].dropna()
        if len(d):
            ax.errorbar(
                mean_x_map[stage], d.mean(), yerr=sem_safe(d),
                fmt="^", markersize=12, color="black", capsize=4
            )

    ax.set_xlim(0.7, 1.32)
    ax.set_ylim(0, 26)
    ax.set_xticks([0.89, 1.19])
    ax.set_xticklabels(["Naive", "Trained"])
    ax.set_ylabel("Number of blocks", fontsize=plot_config["label_fontsize"],
                  labelpad=plot_config["ylabel_pad"])
    ax.set_title("Total block count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p_blocks = stats_n_blocks["overall_naive_vs_trained"]["p"].iloc[0]
    add_sig_bracket(ax, 0.85, 1.15, 23.0, 0.8, f"{p_to_stars(p_blocks)}\n{p_to_text(p_blocks)}", fontsize=10)

    # ========================================================
    # PANEL 3: block types
    # ========================================================
    ax = axes[2]
    block_order = ["sound", "action-right", "action-left"]
    block_labels = ["Sound", "Action-R", "Action-L"]

    centers, stage_pos = get_two_stage_positions(len(block_order), group_spacing=2.3, stage_offset=0.28)
    block_to_idx = {b: i for i, b in enumerate(block_order)}

    for _, row in df_blocktypes.iterrows():
        i = block_to_idx[row["block"]]
        x0 = stage_pos[row["stage"]][i]
        jitter = (rng.random() - 0.5) * 0.10
        color = strain_colors.get(row["strain"], "#333333")

        ax.scatter(
            x0 + jitter,
            row["count"],
            marker="^", s=130,
            facecolors=color, edgecolors=color, linewidths=1.8
        )

    for stage in STAGE_ORDER:
        for block in block_order:
            d = df_blocktypes.loc[
                (df_blocktypes["stage"] == stage) & (df_blocktypes["block"] == block),
                "count"
            ].dropna()
            if len(d):
                ax.errorbar(
                    stage_pos[stage][block_to_idx[block]] + 0.12,
                    d.mean(),
                    yerr=sem_safe(d),
                    fmt="^", markersize=12, color="black", capsize=4
                )

    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    ax.set_ylim(0, 15)
    ax.set_xticks(centers)
    ax.set_xticklabels(block_labels)
    ax.set_ylabel("Blocks per type", fontsize=plot_config["label_fontsize"],
                  labelpad=plot_config["ylabel_pad"])
    ax.set_title("Block types")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_stage_letters(ax, block_order, stage_pos)

    # naive vs trained within each block type
    block_stats = stats_blocktypes["overall_naive_vs_trained_by_category"]
    y_positions = {"sound": 13.2, "action-right": 10.8, "action-left": 13.2}

    for block in block_order:
        sub = block_stats[block_stats["block"] == block]
        if sub.empty:
            continue

        if "p_fdr" in sub.columns and pd.notna(sub["p_fdr"].iloc[0]):
            p = sub["p_fdr"].iloc[0]
        else:
            p = sub["p"].iloc[0]

        i = block_to_idx[block]
        add_sig_bracket(
            ax,
            stage_pos["naive"][i],
            stage_pos["trained"][i],
            y_positions[block],
            0.4,
            f"{p_to_stars(p)}\n{p_to_text(p)}",
            fontsize=9
        )

    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, axes


def plot_triangle_summary_combined_by_strain(metrics_df, strain_colors, master_stats_list=None):
    df = metrics_df[metrics_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No data for triangle summary by strain")
        return None, None

    df_blocktypes = make_blocktypes_df(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=plot_config["dpi"])
    plt.subplots_adjust(wspace=plot_config["subplot_wspace"])

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------
    stats_total_trials = compute_two_stage_stats(
        df=df,
        value_col="total_trials",
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_total_trials, prefix="triangle_summary_by_strain_total_trials")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_total_trials, "triangle_summary_by_strain_total_trials")

    stats_n_blocks = compute_two_stage_stats(
        df=df,
        value_col="n_blocks",
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_n_blocks, prefix="triangle_summary_by_strain_total_block_count")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_n_blocks, "triangle_summary_by_strain_total_block_count")

    stats_blocktypes = compute_two_stage_stats_by_category(
        df=df_blocktypes,
        category_col="block",
        value_col="count",
        category_order=["sound", "action-right", "action-left"],
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER
    )
    save_stats_dict(stats_blocktypes, prefix="triangle_summary_by_strain_block_types")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats_blocktypes, "triangle_summary_by_strain_block_types")

    # total trials
    for ax, value_col, ylabel, title, ylim, stats_here in zip(
        axes[:2],
        ["total_trials", "n_blocks"],
        ["Trials performed", "Number of blocks"],
        ["Total trials", "Total block count"],
        [(500, 800), (0, 26)],
        [stats_total_trials, stats_n_blocks]
    ):
        x_map = {"naive": 0.85, "trained": 1.15}
        offsets = np.linspace(-0.12, 0.12, len(STRAIN_ORDER))

        for i, strain in enumerate(STRAIN_ORDER):
            if strain not in df["strain"].astype(str).unique():
                continue
            color = strain_colors.get(strain, "gray")
            offset = offsets[i]

            for stage in STAGE_ORDER:
                d = df.loc[
                    (df["stage"] == stage) & (df["strain"] == strain),
                    value_col
                ].dropna()

                if len(d) == 0:
                    continue

                ax.errorbar(
                    x_map[stage] + offset,
                    d.mean(),
                    yerr=sem_safe(d),
                    fmt="^",
                    markersize=12,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    capsize=4,
                    linewidth=2
                )

        ax.set_xlim(0.6, 1.4)
        ax.set_ylim(*ylim)
        ax.set_xticks([0.85, 1.15])
        ax.set_xticklabels(["Naive", "Trained"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — strain means ± SEM")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        p_naive = stats_here["between_strains_naive_kw"]["p"].iloc[0]
        p_trained = stats_here["between_strains_trained_kw"]["p"].iloc[0]

        if value_col == "total_trials":
            add_sig_bracket(ax, 0.67, 1.03, 770, 7, f"N: {p_to_stars(p_naive)}", fontsize=9)
            add_sig_bracket(ax, 0.97, 1.33, 750, 7, f"T: {p_to_stars(p_trained)}", fontsize=9)
        else:
            add_sig_bracket(ax, 0.67, 1.03, 23.0, 0.7, f"N: {p_to_stars(p_naive)}", fontsize=9)
            add_sig_bracket(ax, 0.97, 1.33, 21.0, 0.7, f"T: {p_to_stars(p_trained)}", fontsize=9)

    # block types
    ax = axes[2]
    block_order = ["sound", "action-right", "action-left"]
    block_labels = ["Sound", "Action-R", "Action-L"]
    centers, stage_pos = get_two_stage_positions(len(block_order), group_spacing=2.3, stage_offset=0.28)
    block_to_idx = {b: i for i, b in enumerate(block_order)}
    strain_offsets = np.linspace(-0.12, 0.12, len(STRAIN_ORDER))

    for s_idx, strain in enumerate(STRAIN_ORDER):
        if strain not in df_blocktypes["strain"].astype(str).unique():
            continue
        color = strain_colors.get(strain, "gray")
        soff = strain_offsets[s_idx]

        for stage in STAGE_ORDER:
            for block in block_order:
                d = df_blocktypes.loc[
                    (df_blocktypes["stage"] == stage)
                    & (df_blocktypes["strain"] == strain)
                    & (df_blocktypes["block"] == block),
                    "count"
                ].dropna()

                if len(d) == 0:
                    continue

                ax.errorbar(
                    stage_pos[stage][block_to_idx[block]] + soff,
                    d.mean(),
                    yerr=sem_safe(d),
                    fmt="^",
                    markersize=11,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    capsize=4,
                    linewidth=2
                )

    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    ax.set_ylim(0, 15)
    ax.set_xticks(centers)
    ax.set_xticklabels(block_labels)
    ax.set_ylabel("Blocks per type")
    ax.set_title("Block types — strain means ± SEM")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_stage_letters(ax, block_order, stage_pos)

    kw_df = stats_blocktypes["between_strains_kw_by_category"]

    y_naive = {"sound": 13.7, "action-right": 11.5, "action-left": 13.7}
    y_trained = {"sound": 12.7, "action-right": 10.5, "action-left": 12.7}

    for block in block_order:
        i = block_to_idx[block]

        sub_n = kw_df[(kw_df["block"] == block) & (kw_df["stage"] == "naive")]
        if not sub_n.empty:
            p = sub_n["p"].iloc[0]
            add_sig_bracket(
                ax,
                stage_pos["naive"][i] - 0.18,
                stage_pos["naive"][i] + 0.18,
                y_naive[block],
                0.25,
                f"N: {p_to_stars(p)}",
                fontsize=8
            )

        sub_t = kw_df[(kw_df["block"] == block) & (kw_df["stage"] == "trained")]
        if not sub_t.empty:
            p = sub_t["p"].iloc[0]
            add_sig_bracket(
                ax,
                stage_pos["trained"][i] - 0.18,
                stage_pos["trained"][i] + 0.18,
                y_trained[block],
                0.25,
                f"T: {p_to_stars(p)}",
                fontsize=8
            )

    add_strain_legend(fig, df, strain_colors)
    plt.tight_layout()
    return fig, axes

# ============================================================
# PERFORMANCE TABLES
# ============================================================

def compute_performance_tables(sessions_df):
    print("\n=== Computing performance tables ===")
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
            n_omissions = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].shape[0]
            denom = n_correct + n_incorrect + n_omissions
            performance = (n_correct / denom * 100) if denom > 0 else np.nan
            performance_records.append({
                "animal": row["animal"], "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"], "stage": row["stage"],
                "correct": n_correct, "incorrect": n_incorrect, "omissions": n_omissions, "performance": performance,
            })
        else:
            print(f"⚠ Missing columns for session performance in {row['filename']}")

        needed_block = ["block", "reward", "punishment", "omission", "catch_trial"]
        if all(c in df.columns for c in needed_block):
            for blk in block_types:
                df_blk = df[df["block"] == blk]
                if df_blk.empty:
                    continue
                n_correct = (df_blk["reward"] == 1).sum()
                n_incorrect = (df_blk["punishment"] == 1).sum()
                n_omissions = df_blk[(df_blk["omission"] == 1) & (df_blk["catch_trial"] == 0)].shape[0]
                denom = n_correct + n_incorrect + n_omissions
                perf = (n_correct / denom * 100) if denom > 0 else np.nan
                perf_block_records.append({
                    "animal": row["animal"], "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"], "stage": row["stage"],
                    "block_type": blk, "performance": perf,
                })
        else:
            print(f"⚠ Missing columns for block performance in {row['filename']}")

    perf_df = pd.DataFrame(performance_records)
    perf_block_df = pd.DataFrame(perf_block_records)
    print("\n=== PERFORMANCE TABLE ==="); print(perf_df.head())
    print("\n=== PERFORMANCE PER BLOCK TYPE ==="); print(perf_block_df.head())
    return perf_df, perf_block_df


def plot_performance_per_session(perf_df, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric(
        df=perf_df, value_col="performance", ylabel="Performance (%)", title="Performance per Session",
        strain_colors=strain_colors, ylim=(0, 100), chance_line=50, figsize=(7, 5),
        dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_performance_per_session_by_strain(perf_df, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric_by_strain(
        df=perf_df, value_col="performance", ylabel="Performance (%)", title="Performance per Session",
        strain_colors=strain_colors, ylim=(0, 100), chance_line=50, figsize=(7, 5),
        dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_performance_per_blocktype(perf_block_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=perf_block_df, category_col="block_type", value_col="performance",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Performance (%)", title="Performance per Block Type", strain_colors=strain_colors,
        ylim=(0, 100), chance_line=50, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_performance_per_blocktype_by_strain(perf_block_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=perf_block_df, category_col="block_type", value_col="performance",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Performance (%)", title="Performance per Block Type", strain_colors=strain_colors,
        ylim=(0, 100), chance_line=50, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )

# Additional analysis/plotting functions from the original script continue below.
# For brevity and stability in this integrated version, the remaining metric computations
# are preserved and the wrapper functions that use generic plotting now pass through the
# master_stats_list parameter when relevant.

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
            records.append({
                "animal": row["animal"], "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"], "stage": row["stage"], "block_type": blk_type, "n_trials": len(df_blk),
            })
    tri_df = pd.DataFrame(records)
    if tri_df.empty:
        print("⚠ No trials-until-switch records computed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    session_means_df = tri_df.groupby(
        ["animal", "session_date", "session_id", "strain", "stage", "block_type"]
    )["n_trials"].mean().reset_index()
    overall_stats_df = tri_df.groupby(["stage", "block_type"])["n_trials"].agg(["mean", "sem"]).reset_index()
    print("\n=== TRIALS UNTIL SWITCH TABLE ==="); print(tri_df.head())
    print("\n=== SESSION MEANS TABLE ==="); print(session_means_df.head())
    return tri_df, session_means_df, overall_stats_df


def plot_trials_until_switch_blocktype(session_means_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=session_means_df, category_col="block_type", value_col="n_trials",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Trials until switch", title="Trials Until Block Switch", strain_colors=strain_colors,
        ylim=(0, 350), ref_line=20, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_trials_until_switch_blocktype_by_strain(session_means_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=session_means_df, category_col="block_type", value_col="n_trials",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Trials until switch", title="Trials Until Block Switch", strain_colors=strain_colors,
        ylim=(0, 350), ref_line=20, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )

# ============================================================
# OMISSIONS PER BLOCK TYPE
# ============================================================

def compute_omissions_blocktype_table(sessions_df):
    print("\n=== Computing omissions per block type ===")
    omission_records = []
    block_order = ["sound", "action-right", "action-left"]
    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue
        required_cols = {"omission", "block", "catch_trial"}
        if not required_cols.issubset(df.columns):
            print(f"⚠ Missing omission/block/catch_trial columns in {row['filename']}")
            continue
        df_valid = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].copy()
        for blk in block_order:
            blk_df = df_valid[df_valid["block"] == blk]
            omission_records.append({
                "animal": row["animal"], "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"],
                "stage": row["stage"], "block_type": blk, "n_omissions": len(blk_df),
            })
    df_omissions = pd.DataFrame(omission_records)
    if df_omissions.empty:
        print("⚠ No omission data computed.")
        return pd.DataFrame()
    print("\n=== OMISSIONS PER BLOCK TYPE ==="); print(df_omissions.head())
    return df_omissions


def plot_omissions_blocktype(df_omissions, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=df_omissions, category_col="block_type", value_col="n_omissions",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Number of omissions", title="Omissions per Block Type", strain_colors=strain_colors,
        ylim=(0, 200), figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_omissions_blocktype_by_strain(df_omissions, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=df_omissions, category_col="block_type", value_col="n_omissions",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Number of omissions", title="Omissions per Block Type", strain_colors=strain_colors,
        ylim=(0, 200), figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )

# ============================================================
# PERCENT CORRECT
# ============================================================

def compute_percentcorrect_tables(sessions_df):
    print("\n=== Computing performance tables ===")
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
                "animal": row["animal"], "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"], "stage": row["stage"],
                "correct": n_correct, "incorrect": n_incorrect, "percentcorrect": percentcorrect,
            })
        else:
            print(f"⚠ Missing columns for session percentcorrect in {row['filename']}")
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
                    "animal": row["animal"], "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"],
                    "stage": row["stage"], "block_type": blk, "percentcorrect": percentcorrect_blocks,
                })
        else:
            print(f"⚠ Missing columns for block performance in {row['filename']}")
    percentcorrect_df = pd.DataFrame(percentcorrect_records)
    percentcorrect_block_df = pd.DataFrame(percentcorrect_block_records)
    print("\n=== percentcorrect TABLE ==="); print(percentcorrect_df.head())
    print("\n=== percentcorrect PER BLOCK TYPE ==="); print(percentcorrect_block_df.head())
    return percentcorrect_df, percentcorrect_block_df


def plot_percentcorrect_per_session(percentcorrect_df, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric(
        df=percentcorrect_df, value_col="percentcorrect", ylabel="Percent correct (%)",
        title="Percent Correct per Session", strain_colors=strain_colors, ylim=(0, 100), chance_line=50,
        figsize=(7, 5), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_percentcorrect_per_session_by_strain(percentcorrect_df, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric_by_strain(
        df=percentcorrect_df, value_col="percentcorrect", ylabel="Percent correct (%)",
        title="Percent Correct per Session", strain_colors=strain_colors, ylim=(0, 100), chance_line=50,
        figsize=(7, 5), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_percentcorrect_per_blocktype(percentcorrect_block_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=percentcorrect_block_df, category_col="block_type", value_col="percentcorrect",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Percent correct (%)", title="Percent Correct per Block Type", strain_colors=strain_colors,
        ylim=(0, 100), chance_line=50, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_percentcorrect_per_blocktype_by_strain(percentcorrect_block_df, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=percentcorrect_block_df, category_col="block_type", value_col="percentcorrect",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Percent correct (%)", title="Percent Correct per Block Type", strain_colors=strain_colors,
        ylim=(0, 100), chance_line=50, figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )

# ============================================================
# BLOCKS PER 100 TRIALS
# ============================================================

def compute_blocks_per_100_table(metrics_df):
    df = metrics_df.copy()
    if df.empty:
        print("⚠ metrics_df is empty")
        return pd.DataFrame()
    df["blocks_per_100_trials"] = (df["n_blocks"] / df["total_trials"]) * 100
    out = df[["animal", "session_date", "strain", "stage", "n_blocks", "total_trials", "blocks_per_100_trials"]].copy()
    print("\n=== BLOCKS PER 100 TRIALS ==="); print(out.head())
    return out


def plot_blocks_per_100(df_blocks100, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric(
        df=df_blocks100, value_col="blocks_per_100_trials", ylabel="Blocks per 100 trials",
        title="Blocks per 100 Trials", strain_colors=strain_colors, figsize=(7, 5),
        dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_blocks_per_100_by_strain(df_blocks100, strain_colors, master_stats_list=None):
    return plot_two_stage_single_metric_by_strain(
        df=df_blocks100, value_col="blocks_per_100_trials", ylabel="Blocks per 100 trials",
        title="Blocks per 100 Trials", strain_colors=strain_colors, figsize=(7, 5),
        dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )

# ============================================================
# TRIALS PER BLOCK
# ============================================================

def compute_trials_per_block_table(sessions_df):
    print("\n=== Computing trials per block table ===")
    records = []
    block_order = ["sound", "action-right", "action-left"]
    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue
        if "block" not in df.columns:
            print(f"⚠ Missing block column in {row['filename']}")
            continue
        df["block_change"] = (df["block"] != df["block"].shift(1)).astype(int)
        df["block_index"] = df["block_change"].cumsum()
        for _, dblk in df.groupby("block_index"):
            blk_type = dblk["block"].iloc[0]
            if blk_type not in block_order:
                continue
            records.append({
                "animal": row["animal"], "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}", "strain": row["strain"],
                "stage": row["stage"], "block_type": blk_type, "n_trials": len(dblk),
            })
    out = pd.DataFrame(records)
    print("\n=== TRIALS PER BLOCK TABLE ==="); print(out.head())
    return out


def plot_trials_per_block(df_trials_block, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=df_trials_block, category_col="block_type", value_col="n_trials",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Trials per block", title="Trials per Block", strain_colors=strain_colors,
        figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


def plot_trials_per_block_by_strain(df_trials_block, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=df_trials_block, category_col="block_type", value_col="n_trials",
        category_order=["sound", "action-right", "action-left"], category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Trials per block", title="Trials per Block", strain_colors=strain_colors,
        figsize=(9, 6), dpi=plot_config["dpi"], master_stats_list=master_stats_list,
    )


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


def plot_pe_triangle(pe_df, strain_colors, master_stats_list=None):
    df_long = pe_df.melt(
        id_vars=["animal", "session_date", "session_id", "strain", "stage"],
        value_vars=["Action→Sound", "Sound→Action"],
        var_name="switch_type",
        value_name="pe_count"
    )

    return plot_two_stage_by_category(
        df=df_long,
        category_col="switch_type",
        value_col="pe_count",
        category_order=["Action→Sound", "Sound→Action"],
        category_labels=["Action→Sound", "Sound→Action"],
        ylabel="Number of Perseverative Errors",
        title="Perseverative Errors Around Block Switches",
        strain_colors=strain_colors,
        figsize=(8, 6),
        dpi=plot_config["dpi"],
        master_stats_list=master_stats_list,
    )


def plot_pe_triangle_by_strain(pe_df, strain_colors, master_stats_list=None):
    df_long = pe_df.melt(
        id_vars=["animal", "session_date", "session_id", "strain", "stage"],
        value_vars=["Action→Sound", "Sound→Action"],
        var_name="switch_type",
        value_name="pe_count"
    )

    return plot_two_stage_by_category_by_strain(
        df=df_long,
        category_col="switch_type",
        value_col="pe_count",
        category_order=["Action→Sound", "Sound→Action"],
        category_labels=["Action→Sound", "Sound→Action"],
        ylabel="Number of Perseverative Errors",
        title="Perseverative Errors Around Block Switches",
        strain_colors=strain_colors,
        figsize=(8, 6),
        dpi=plot_config["dpi"],
        master_stats_list=master_stats_list,
    )


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

    print("\n=== PE TIME COURSE TABLE ===")
    print(pe_timecourse_df.head())

    print("\n=== SESSION PE TIME COURSE TABLE ===")
    print(session_timecourse_df.head())

    return pe_timecourse_df, session_timecourse_df


def plot_pe_timecourse(session_timecourse_df, strain_colors):
    df = session_timecourse_df[session_timecourse_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No PE time course data")
        return None, None

    switch_order = ["Action→Sound", "Sound→Action"]
    stage_color = {"naive": "#4C72B0", "trained": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df[df["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

        for stage in STAGE_ORDER:
            d_stage = df_sw[df_sw["stage"] == stage].copy()
            if d_stage.empty:
                continue

            stats_df = (
                d_stage.groupby("trial_from_switch")["pe"]
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
                color=stage_color[stage],
                label=STAGE_LABELS[stage]
            )

            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=stage_color[stage],
                alpha=0.18
            )

        max_trial = int(df_sw["trial_from_switch"].max())
        ax.set_xlim(0, max_trial)
        ax.set_xticks(range(0, max_trial + 1, 2))
        ax.set_ylim(0, 100)
        ax.set_xlabel("Trial after switch", fontsize=plot_config["label_fontsize"], labelpad=13)
        ax.set_title(switch_type, fontsize=plot_config["title_fontsize"], pad=15)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Perseverative errors (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles = [
        plt.Line2D([0], [0], color=stage_color[stage], linewidth=2.8, label=STAGE_LABELS[stage])
        for stage in STAGE_ORDER if stage in df["stage"].unique()
    ]
    fig.legend(handles=handles, title="Stage", frameon=False)

    fig.suptitle(
        "PE Time Course After Switch",
        fontsize=plot_config["title_fontsize"], y=1.02
    )

    plt.tight_layout()
    return fig, axes


def plot_pe_timecourse_by_strain(session_timecourse_df, strain_colors):
    df = session_timecourse_df[session_timecourse_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No PE time course data")
        return None, None

    switch_order = ["Action→Sound", "Sound→Action"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df[df["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

        for strain in STRAIN_ORDER:
            if strain not in df_sw["strain"].astype(str).unique():
                continue

            color = strain_colors.get(strain, "gray")

            for stage in STAGE_ORDER:
                d_stage = df_sw[
                    (df_sw["stage"] == stage) & (df_sw["strain"] == strain)
                ].copy()

                if d_stage.empty:
                    continue

                stats_df = (
                    d_stage.groupby("trial_from_switch")["pe"]
                    .agg(["mean", "sem"])
                    .reset_index()
                    .sort_values("trial_from_switch")
                )

                x = stats_df["trial_from_switch"].values
                y = stats_df["mean"].values * 100
                sem = stats_df["sem"].fillna(0).values * 100

                linestyle = "-" if stage == "naive" else "--"

                ax.plot(
                    x,
                    y,
                    linewidth=2.2,
                    color=color,
                    linestyle=linestyle,
                    label=f"{strain} ({STAGE_LABELS[stage]})"
                )

                ax.fill_between(
                    x,
                    y - sem,
                    y + sem,
                    color=color,
                    alpha=0.10
                )

        max_trial = int(df_sw["trial_from_switch"].max())
        ax.set_xlim(0, max_trial)
        ax.set_xticks(range(0, max_trial + 1, 2))
        ax.set_ylim(0, 100)
        ax.set_xlabel("Trial after switch", fontsize=plot_config["label_fontsize"], labelpad=13)
        ax.set_title(f"{switch_type} — strain means ± SEM", fontsize=plot_config["title_fontsize"], pad=15)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Perseverative errors (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)

    fig.suptitle(
        "PE Time Course After Switch",
        fontsize=plot_config["title_fontsize"], y=1.02
    )

    plt.tight_layout()
    return fig, axes


# ============================================================
# LATENCY HELPERS
# ============================================================

def correct_side_latency(block, stim, tone_map):
    if block == "sound":
        return tone_map[stim]
    elif block == "action-left":
        return "left"
    elif block == "action-right":
        return "right"
    return None


def opposite(side):
    return "right" if side == "left" else "left"


# ============================================================
# LATENCY TABLE
# ============================================================

def compute_latency_table(sessions_df):
    print("\n=== Computing latency table ===")

    latency_records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required = [
            "block", "lick_time", "RW_start",
            "reward", "punishment", "omission",
            "8KHz", "16KHz"
        ]

        if not all(c in df.columns for c in required):
            print(f"⚠ Missing columns in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        df["latency"] = df["lick_time"] - df["RW_start"]

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else None),
            axis=1
        )

        for _, r in df.iterrows():
            if pd.isna(r["stim"]) or r["omission"] == 1:
                continue

            blk = r["block"]
            stim = r["stim"]
            lat = r["latency"]

            if pd.isna(lat):
                continue

            cside = correct_side_latency(blk, stim, tone_map)
            if cside is None:
                continue

            if r["reward"] == 1:
                lick_side = cside
                correctness = "correct"
            elif r["punishment"] == 1:
                lick_side = opposite(cside)
                correctness = "incorrect"
            else:
                continue

            latency_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block": blk,
                "side": lick_side,
                "correctness": correctness,
                "latency": lat
            })

    lat_df = pd.DataFrame(latency_records)

    if lat_df.empty:
        print("⚠ No latency data computed.")
        return pd.DataFrame()

    valid_blocks = ["sound", "action-left", "action-right"]
    lat_df = lat_df[lat_df["block"].isin(valid_blocks)].copy()

    print("\n=== LATENCY TABLE ===")
    print(lat_df.head())

    return lat_df


def plot_latency_panels_by_stage(lat_df, strain_colors):
    df = lat_df[lat_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No latency data")
        return None, None

    categories = [
        ("sound", "correct", "Sound — Correct"),
        ("sound", "incorrect", "Sound — Incorrect"),
        ("action-right", "all", "Action-Right — All trials"),
        ("action-left", "all", "Action-Left — All trials"),
    ]

    stage_color = {"naive": "#4C72B0", "trained": "#DD8452"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), dpi=plot_config["dpi"])
    plt.subplots_adjust(wspace=0.45)

    for ax, (blk, cond, panel_title) in zip(axes, categories):
        x_left = 0
        x_right = 1

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"], fontsize=plot_config["tick_fontsize"])

        for stage in STAGE_ORDER:
            d_panel = df[(df["block"] == blk) & (df["stage"] == stage)].copy()
            if cond != "all":
                d_panel = d_panel[d_panel["correctness"] == cond]

            if d_panel.empty:
                continue

            stats_stage = (
                d_panel.groupby("side")["latency"]
                .agg(["mean", "sem"])
                .reindex(["left", "right"])
            )

            left_mean = stats_stage.loc["left", "mean"] if "left" in stats_stage.index else np.nan
            right_mean = stats_stage.loc["right", "mean"] if "right" in stats_stage.index else np.nan
            color = stage_color[stage]

            if not pd.isna(left_mean) and not pd.isna(right_mean):
                ax.plot(
                    [x_left, x_right],
                    [left_mean, right_mean],
                    color=color,
                    linewidth=3.0,
                    alpha=0.95
                )

            if not pd.isna(left_mean):
                ax.errorbar(
                    x_left - (0.07 if stage == "naive" else -0.07),
                    left_mean,
                    yerr=stats_stage.loc["left", "sem"],
                    fmt="o",
                    color=color,
                    capsize=4,
                    markersize=8,
                    linewidth=2,
                    zorder=5
                )

            if not pd.isna(right_mean):
                ax.errorbar(
                    x_right - (0.07 if stage == "naive" else -0.07),
                    right_mean,
                    yerr=stats_stage.loc["right", "sem"],
                    fmt="o",
                    color=color,
                    capsize=4,
                    markersize=8,
                    linewidth=2,
                    zorder=5
                )

        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.set_title(panel_title, fontsize=plot_config["title_fontsize"], pad=13)
        ax.set_ylabel(
            "Mean lick latency (s)",
            fontsize=plot_config["label_fontsize"],
            labelpad=plot_config["ylabel_pad"]
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=stage_color[s], markersize=6,
                   linewidth=2.5, label=STAGE_LABELS[s])
        for s in STAGE_ORDER if s in df["stage"].unique()
    ]
    fig.legend(handles=handles, title="Stage", frameon=False)
    fig.suptitle(
        "Latency per Spout Side",
        fontsize=plot_config["title_fontsize"], y=1.03
    )

    plt.tight_layout()
    return fig, axes


def plot_latency_panels_by_stage_by_strain(lat_df, strain_colors):
    df = lat_df[lat_df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No latency data")
        return None, None

    categories = [
        ("sound", "correct", "Sound — Correct"),
        ("sound", "incorrect", "Sound — Incorrect"),
        ("action-right", "all", "Action-Right — All trials"),
        ("action-left", "all", "Action-Left — All trials"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), dpi=plot_config["dpi"])
    plt.subplots_adjust(wspace=0.45)

    for ax, (blk, cond, panel_title) in zip(axes, categories):
        x_left = 0
        x_right = 1

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"], fontsize=plot_config["tick_fontsize"])

        for strain in STRAIN_ORDER:
            if strain not in df["strain"].astype(str).unique():
                continue

            color = strain_colors.get(strain, "gray")

            for stage in STAGE_ORDER:
                d_panel = df[
                    (df["block"] == blk)
                    & (df["stage"] == stage)
                    & (df["strain"] == strain)
                ].copy()

                if cond != "all":
                    d_panel = d_panel[d_panel["correctness"] == cond]

                if d_panel.empty:
                    continue

                stats_stage = (
                    d_panel.groupby("side")["latency"]
                    .agg(["mean", "sem"])
                    .reindex(["left", "right"])
                )

                left_mean = stats_stage.loc["left", "mean"] if "left" in stats_stage.index else np.nan
                right_mean = stats_stage.loc["right", "mean"] if "right" in stats_stage.index else np.nan

                linestyle = "-" if stage == "naive" else "--"

                if not pd.isna(left_mean) and not pd.isna(right_mean):
                    ax.plot(
                        [x_left, x_right],
                        [left_mean, right_mean],
                        color=color,
                        linewidth=2.0,
                        linestyle=linestyle,
                        alpha=0.95
                    )

                if not pd.isna(left_mean):
                    ax.errorbar(
                        x_left,
                        left_mean,
                        yerr=stats_stage.loc["left", "sem"],
                        fmt="o",
                        color=color,
                        capsize=4,
                        markersize=6,
                        linewidth=2
                    )

                if not pd.isna(right_mean):
                    ax.errorbar(
                        x_right,
                        right_mean,
                        yerr=stats_stage.loc["right", "sem"],
                        fmt="o",
                        color=color,
                        capsize=4,
                        markersize=6,
                        linewidth=2
                    )

        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.set_title(f"{panel_title} — strain means ± SEM", fontsize=plot_config["title_fontsize"], pad=13)
        ax.set_ylabel(
            "Mean lick latency (s)",
            fontsize=plot_config["label_fontsize"],
            labelpad=plot_config["ylabel_pad"]
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = []
    for strain in STRAIN_ORDER:
        if strain in df["strain"].astype(str).unique():
            handles.append(
                plt.Line2D([0], [0], color=strain_colors[strain], linewidth=2.5, label=strain)
            )
    if handles:
        fig.legend(handles=handles, title="Strain", frameon=False)

    fig.suptitle(
        "Latency per Spout Side",
        fontsize=plot_config["title_fontsize"], y=1.03
    )

    plt.tight_layout()
    return fig, axes


# ============================================================
# OMISSIONS PER BLOCK TYPE
# ============================================================

def compute_omissions_blocktype_table(sessions_df):
    print("\n=== Computing omissions per block type ===")

    omission_records = []
    block_order = ["sound", "action-right", "action-left"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required_cols = {"omission", "block", "catch_trial"}
        if not required_cols.issubset(df.columns):
            print(f"⚠ Missing omission/block/catch_trial columns in {row['filename']}")
            continue

        df_valid = df[(df["omission"] == 1) & (df["catch_trial"] == 0)].copy()

        for blk in block_order:
            blk_df = df_valid[df_valid["block"] == blk]
            n_omissions = len(blk_df)

            omission_records.append({
                "animal": row["animal"],
                "session_date": row["session_date"],
                "session_id": f"{row['animal']}_{row['session_date']}",
                "strain": row["strain"],
                "stage": row["stage"],
                "block_type": blk,
                "n_omissions": n_omissions,
            })

    df_omissions = pd.DataFrame(omission_records)

    if df_omissions.empty:
        print("⚠ No omission data computed.")
        return pd.DataFrame()

    print("\n=== OMISSIONS PER BLOCK TYPE ===")
    print(df_omissions.head())

    return df_omissions


def plot_omissions_blocktype(df_omissions, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category(
        df=df_omissions,
        category_col="block_type",
        value_col="n_omissions",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Number of omissions",
        title="Omissions per Block Type",
        strain_colors=strain_colors,
        ylim=(0, 200),
        figsize=(9, 6),
        dpi=plot_config["dpi"],
        master_stats_list=master_stats_list,
    )


def plot_omissions_blocktype_by_strain(df_omissions, strain_colors, master_stats_list=None):
    return plot_two_stage_by_category_by_strain(
        df=df_omissions,
        category_col="block_type",
        value_col="n_omissions",
        category_order=["sound", "action-right", "action-left"],
        category_labels=["Sound", "Action-R", "Action-L"],
        ylabel="Number of omissions",
        title="Omissions per Block Type",
        strain_colors=strain_colors,
        ylim=(0, 200),
        figsize=(9, 6),
        dpi=plot_config["dpi"],
        master_stats_list=master_stats_list,
    )


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


def plot_pe_first_last_blocks(df_pe_fl, strain_colors):
    df = df_pe_fl[df_pe_fl["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No PE first/last data")
        return None, None

    block_order = ["sound", "action-right", "action-left"]
    period_order = ["first", "last"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk_type in zip(axes, block_order):
        dblk = df[df["block_type"] == blk_type].copy()

        for _, row in dblk.iterrows():
            base_x = 0 if row["period"] == "first" else 1
            stage_shift = -0.14 if row["stage"] == "naive" else 0.14
            jitter = (rng.random() - 0.5) * 0.08
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                base_x + stage_shift + jitter,
                row["pe_count"],
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        for i, period in enumerate(period_order):
            for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                d = dblk[(dblk["period"] == period) & (dblk["stage"] == stage)]["pe_count"].dropna()
                if len(d) == 0:
                    continue

                ax.errorbar(
                    i + shift + 0.06,
                    d.mean(),
                    yerr=sem_safe(d),
                    fmt="^",
                    markersize=12,
                    color="black",
                    capsize=4,
                    linewidth=2
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First", "Last"], fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk_type, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "Perseverative errors",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "PE in First vs Last Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


def plot_pe_first_last_blocks_by_strain(df_pe_fl, strain_colors):
    df = df_pe_fl[df_pe_fl["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No PE first/last data")
        return None, None

    block_order = ["sound", "action-right", "action-left"]
    period_order = ["first", "last"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, blk_type in zip(axes, block_order):
        dblk = df[df["block_type"] == blk_type].copy()

        for strain in STRAIN_ORDER:
            if strain not in dblk["strain"].astype(str).unique():
                continue
            color = strain_colors.get(strain, "gray")

            for i, period in enumerate(period_order):
                for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                    d = dblk[
                        (dblk["period"] == period)
                        & (dblk["stage"] == stage)
                        & (dblk["strain"] == strain)
                    ]["pe_count"].dropna()

                    if len(d) == 0:
                        continue

                    ax.errorbar(
                        i + shift,
                        d.mean(),
                        yerr=sem_safe(d),
                        fmt="^",
                        markersize=10,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        capsize=4,
                        linewidth=2
                    )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First", "Last"], fontsize=plot_config["tick_fontsize"])
        ax.set_title(f"{blk_type} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "Perseverative errors",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "PE in First vs Last Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


# ============================================================
# OMISSIONS PER STIMULUS IN ACTION BLOCKS
# ============================================================

def compute_actionblock_stim_omissions(sessions_df):
    print("\n=== Computing omissions per stimulus in action blocks ===")

    records = []
    action_blocks = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Error loading {row['filename']}: {e}")
            continue

        needed = {"block", "omission", "catch_trial", "8KHz", "16KHz"}
        if not needed.issubset(df.columns):
            continue

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        for blk in action_blocks:
            dblk = df[df["block"] == blk].copy()
            if dblk.empty:
                continue

            for stim in stim_order:
                dstim = dblk[dblk["stim"] == stim].copy()
                if dstim.empty:
                    continue

                n_trials = len(dstim)
                n_omissions = len(dstim[(dstim["omission"] == 1) & (dstim["catch_trial"] == 0)])
                omission_rate = n_omissions / n_trials if n_trials > 0 else np.nan

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "action_block": blk,
                    "stim": stim,
                    "sound_rule_side": tone_map[stim],
                    "n_trials": n_trials,
                    "n_omissions": n_omissions,
                    "omission_rate": omission_rate
                })

    out = pd.DataFrame(records)

    print("\n=== ACTION BLOCK STIM OMISSIONS ===")
    print(out.head())

    return out


def plot_actionblock_stim_omissions(df_stim_omit, strain_colors):
    df = df_stim_omit[df_stim_omit["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No stim omission data")
        return None, None

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk in zip(axes, block_order):
        dblk = df[df["action_block"] == blk].copy()

        for _, row in dblk.iterrows():
            base_x = 0 if row["stim"] == "8KHz" else 1
            stage_shift = -0.14 if row["stage"] == "naive" else 0.14
            jitter = (rng.random() - 0.5) * 0.08
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                base_x + stage_shift + jitter,
                row["omission_rate"] * 100,
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        for i, stim in enumerate(stim_order):
            for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                d = dblk[(dblk["stim"] == stim) & (dblk["stage"] == stage)]["omission_rate"].dropna()
                if len(d) == 0:
                    continue

                d_sem = sem_safe(d)
                ax.errorbar(
                    i + shift + 0.06,
                    d.mean() * 100,
                    yerr=d_sem * 100 if not np.isnan(d_sem) else np.nan,
                    fmt="^",
                    markersize=12,
                    color="black",
                    capsize=4,
                    linewidth=2
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "Omission rate (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "Omissions by Stimulus in Action Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


def plot_actionblock_stim_omissions_by_strain(df_stim_omit, strain_colors):
    df = df_stim_omit[df_stim_omit["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No stim omission data")
        return None, None

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, blk in zip(axes, block_order):
        dblk = df[df["action_block"] == blk].copy()

        for strain in STRAIN_ORDER:
            if strain not in dblk["strain"].astype(str).unique():
                continue
            color = strain_colors.get(strain, "gray")

            for i, stim in enumerate(stim_order):
                for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                    d = dblk[
                        (dblk["stim"] == stim)
                        & (dblk["stage"] == stage)
                        & (dblk["strain"] == strain)
                    ]["omission_rate"].dropna()

                    if len(d) == 0:
                        continue

                    d_sem = sem_safe(d)
                    ax.errorbar(
                        i + shift,
                        d.mean() * 100,
                        yerr=d_sem * 100 if not np.isnan(d_sem) else np.nan,
                        fmt="^",
                        markersize=10,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        capsize=4,
                        linewidth=2
                    )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(f"{blk} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "Omission rate (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "Omissions by Stimulus in Action Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


# ============================================================
# CHOICE BIAS BY STIMULUS IN ACTION BLOCKS
# ============================================================

def compute_actionblock_choice_bias(sessions_df):
    print("\n=== Computing choice bias by stimulus in action blocks ===")

    records = []
    action_blocks = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

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

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

        for blk in action_blocks:
            dblk = df[df["block"] == blk].copy()
            if dblk.empty:
                continue

            for stim in stim_order:
                dstim = dblk[(dblk["stim"] == stim) & (dblk["omission"] == 0)].copy()
                if dstim.empty:
                    continue

                if blk == "action-right":
                    n_right = (dstim["reward"] == 1).sum()
                    n_left = (dstim["punishment"] == 1).sum()
                elif blk == "action-left":
                    n_left = (dstim["reward"] == 1).sum()
                    n_right = (dstim["punishment"] == 1).sum()
                else:
                    continue

                total_choices = n_left + n_right
                p_right = n_right / total_choices if total_choices > 0 else np.nan

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "action_block": blk,
                    "stim": stim,
                    "sound_rule_side": tone_map[stim],
                    "p_right": p_right
                })

    out = pd.DataFrame(records)

    print("\n=== ACTION BLOCK CHOICE BIAS ===")
    print(out.head())

    return out


def plot_actionblock_choice_bias(df_choice_bias, strain_colors):
    df = df_choice_bias[df_choice_bias["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No choice bias data")
        return None, None

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)
    rng = np.random.default_rng(42)

    for ax, blk in zip(axes, block_order):
        dblk = df[df["action_block"] == blk].copy()

        for _, row in dblk.iterrows():
            base_x = 0 if row["stim"] == "8KHz" else 1
            stage_shift = -0.14 if row["stage"] == "naive" else 0.14
            jitter = (rng.random() - 0.5) * 0.08
            color = strain_colors.get(row["strain"], "gray")

            ax.scatter(
                base_x + stage_shift + jitter,
                row["p_right"] * 100,
                marker="^",
                s=130,
                facecolors=color,
                edgecolors=color,
                linewidths=1.8
            )

        for i, stim in enumerate(stim_order):
            for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                d = dblk[(dblk["stim"] == stim) & (dblk["stage"] == stage)]["p_right"].dropna()
                if len(d) == 0:
                    continue

                d_sem = sem_safe(d)
                ax.errorbar(
                    i + shift + 0.06,
                    d.mean() * 100,
                    yerr=d_sem * 100 if not np.isnan(d_sem) else np.nan,
                    fmt="^",
                    markersize=12,
                    color="black",
                    capsize=4,
                    linewidth=2
                )

        ax.axhline(50, linestyle="--", color="gray", alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(blk, fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "P(right choice) (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "Choice Bias by Stimulus in Action Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


def plot_actionblock_choice_bias_by_strain(df_choice_bias, strain_colors):
    df = df_choice_bias[df_choice_bias["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print("⚠ No choice bias data")
        return None, None

    block_order = ["action-right", "action-left"]
    stim_order = ["8KHz", "16KHz"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=plot_config["dpi"], sharey=True)

    for ax, blk in zip(axes, block_order):
        dblk = df[df["action_block"] == blk].copy()

        for strain in STRAIN_ORDER:
            if strain not in dblk["strain"].astype(str).unique():
                continue
            color = strain_colors.get(strain, "gray")

            for i, stim in enumerate(stim_order):
                for stage, shift in zip(STAGE_ORDER, [-0.14, 0.14]):
                    d = dblk[
                        (dblk["stim"] == stim)
                        & (dblk["stage"] == stage)
                        & (dblk["strain"] == strain)
                    ]["p_right"].dropna()

                    if len(d) == 0:
                        continue

                    d_sem = sem_safe(d)
                    ax.errorbar(
                        i + shift,
                        d.mean() * 100,
                        yerr=d_sem * 100 if not np.isnan(d_sem) else np.nan,
                        fmt="^",
                        markersize=10,
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        capsize=4,
                        linewidth=2
                    )

        ax.axhline(50, linestyle="--", color="gray", alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(stim_order, fontsize=plot_config["tick_fontsize"])
        ax.set_title(f"{blk} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y0, y1 = ax.get_ylim()
        text_y = y0 - 0.06 * (y1 - y0)
        ax.text(-0.14, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(0.86, text_y, "N", ha="center", va="top", fontsize=plot_config["tick_fontsize"])
        ax.text(1.14, text_y, "T", ha="center", va="top", fontsize=plot_config["tick_fontsize"])

    axes[0].set_ylabel(
        "P(right choice) (%)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    add_strain_legend(fig, df, strain_colors)
    fig.suptitle(
        "Choice Bias by Stimulus in Action Blocks",
        fontsize=plot_config["title_fontsize"]
    )
    plt.tight_layout()
    return fig, axes


# ============================================================
# LATENCY TIME COURSE AFTER SWITCH
# ============================================================

def compute_latency_timecourse_tables(sessions_df, window=20):
    print("\n=== Computing latency time course after block switch ===")

    records = []

    for _, row in sessions_df.iterrows():
        try:
            df = read_csv_cached(row["path"]).copy()
        except Exception as e:
            print(f"⚠ Could not read {row['filename']}: {e}")
            continue

        required = [
            "block", "lick_time", "RW_start",
            "reward", "punishment", "omission",
            "8KHz", "16KHz"
        ]
        if not all(c in df.columns for c in required):
            print(f"⚠ Missing columns in {row['filename']}")
            continue

        try:
            tone_map = get_tone_map(row["animal"])
        except Exception as e:
            print(f"⚠ Could not get tone map for animal {row['animal']}: {e}")
            continue

        df["latency"] = df["lick_time"] - df["RW_start"]

        df["stim"] = df.apply(
            lambda r: "8KHz" if r["8KHz"] == 1 else ("16KHz" if r["16KHz"] == 1 else np.nan),
            axis=1
        )

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
                if pd.isna(tr["stim"]) or tr["omission"] == 1:
                    continue

                stim = tr["stim"]
                blk = tr["block"]
                lat = tr["latency"]

                if pd.isna(lat):
                    continue

                cside = correct_side_latency(blk, stim, tone_map)
                if cside is None:
                    continue

                if tr["reward"] == 1:
                    lick_side = cside
                    correctness = "correct"
                elif tr["punishment"] == 1:
                    lick_side = opposite(cside)
                    correctness = "incorrect"
                else:
                    continue

                records.append({
                    "animal": row["animal"],
                    "session_date": row["session_date"],
                    "session_id": f"{row['animal']}_{row['session_date']}",
                    "strain": row["strain"],
                    "stage": row["stage"],
                    "switch_type": switch_type,
                    "trial_from_switch": k,
                    "latency": lat,
                    "correctness": correctness,
                    "lick_side": lick_side,
                })

    lat_timecourse_df = pd.DataFrame(records)

    if lat_timecourse_df.empty:
        print("⚠ No latency time course data computed.")
        return pd.DataFrame(), pd.DataFrame()

    session_latency_timecourse_df = (
        lat_timecourse_df
        .groupby(
            ["animal", "session_date", "session_id", "strain", "stage", "switch_type", "trial_from_switch"],
            dropna=False
        )["latency"]
        .mean()
        .reset_index()
    )

    print("\n=== LATENCY TIME COURSE TABLE ===")
    print(lat_timecourse_df.head())

    print("\n=== SESSION LATENCY TIME COURSE TABLE ===")
    print(session_latency_timecourse_df.head())

    return lat_timecourse_df, session_latency_timecourse_df


def plot_latency_timecourse_by_strain(session_latency_timecourse_df, strain_colors):
    df = session_latency_timecourse_df[
        session_latency_timecourse_df["stage"].isin(STAGE_ORDER)
    ].copy()

    if df.empty:
        print("⚠ No latency time course data")
        return None, None

    switch_order = ["Action→Sound", "Sound→Action"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 5),
        dpi=plot_config["dpi"],
        sharey=True
    )

    for ax, switch_type in zip(axes, switch_order):
        df_sw = df[df["switch_type"] == switch_type].copy()

        if df_sw.empty:
            ax.set_visible(False)
            continue

        for strain in STRAIN_ORDER:
            if strain not in df_sw["strain"].astype(str).unique():
                continue

            color = strain_colors.get(strain, "gray")

            for stage in STAGE_ORDER:
                d = df_sw[
                    (df_sw["strain"] == strain) &
                    (df_sw["stage"] == stage)
                ].copy()

                if d.empty:
                    continue

                stats_df = (
                    d.groupby("trial_from_switch")["latency"]
                    .agg(["mean", "sem"])
                    .reset_index()
                    .sort_values("trial_from_switch")
                )

                x = stats_df["trial_from_switch"].values
                y = stats_df["mean"].values
                sem = stats_df["sem"].fillna(0).values

                linestyle = "-" if stage == "naive" else "--"

                ax.plot(
                    x,
                    y,
                    linewidth=2.2,
                    color=color,
                    linestyle=linestyle,
                    label=f"{strain} ({STAGE_LABELS[stage]})"
                )

                ax.fill_between(
                    x,
                    y - sem,
                    y + sem,
                    color=color,
                    alpha=0.10
                )

        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 21, 2))
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Trial after switch", fontsize=plot_config["label_fontsize"])
        ax.set_title(f"{switch_type} — strain means ± SEM", fontsize=plot_config["title_fontsize"])
        ax.tick_params(labelsize=plot_config["tick_fontsize"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Mean lick latency (s)",
        fontsize=plot_config["label_fontsize"],
        labelpad=plot_config["ylabel_pad"]
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)

    fig.suptitle(
        "Latency Time Course After Switch",
        fontsize=plot_config["title_fontsize"],
        y=1.05
    )

    plt.tight_layout()
    return fig, axes


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Statistics tables will be saved in: {STATSDIR}")

    meta = pd.read_excel(SESSIONS_XLSX, dtype={"animal": str, "session_date": str})
    meta.columns = [c.strip() for c in meta.columns]

    required_cols = ["animal", "sex", "strain", "cohort", "session_date", "stage"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing required column in Excel file: {col}")

    meta["animal"] = meta["animal"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    meta["session_date"] = meta["session_date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    meta["stage"] = meta["stage"].astype(str).str.strip().str.lower()
    meta["strain"] = meta["strain"].astype(str).str.strip()
    meta["cohort"] = pd.to_numeric(meta["cohort"], errors="coerce")
    meta = meta[meta["cohort"].notna()].copy()
    meta["cohort"] = meta["cohort"].astype(int)
    meta = meta[meta["stage"].isin(["naive", "trained"])].copy()

    included_sessions_df = load_included_sessions(BEHAVIOR_QC_FILE)
    print(f"Included sessions in QC file: {len(included_sessions_df)}")

    meta_before = len(meta)
    meta = meta.merge(included_sessions_df, on=["animal", "session_date"], how="inner")
    if meta.empty:
        raise ValueError("No sessions left after applying include filter.")
    print(f"Metadata rows before include filter: {meta_before}")
    print(f"Metadata rows after include filter: {len(meta)}")

    files_df = find_adaptive_files(meta)
    sessions_df = files_df.merge(meta, on=["animal", "cohort", "session_date"], how="inner")
    if sessions_df.empty:
        raise ValueError("No matching included sessions found between metadata Excel and files.")
    print(f"Matched {len(sessions_df)} adaptive included sessions")

    metrics = []
    for _, row in sessions_df.iterrows():
        try:
            metrics.append(compute_session_metrics(row))
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        raise ValueError("No metrics could be computed.")

    metrics_df = metrics_df.sort_values(["stage", "strain", "animal", "session_date"]).reset_index(drop=True)
    metrics_df.to_csv(OUTDIR / "adaptive_session_metrics_included.csv", index=False)
    sessions_df.to_csv(OUTDIR / "adaptive_sessions_matched_included.csv", index=False)

    print(metrics_df.head())

    perf_df, perf_block_df = compute_performance_tables(sessions_df)
    performance_table = perf_df
    perf_df.to_csv(OUTDIR / "adaptive_performance_per_session_included.csv", index=False)
    perf_block_df.to_csv(OUTDIR / "adaptive_performance_per_blocktype_included.csv", index=False)

    tri_df, session_means_df, overall_stats_df = compute_trials_until_switch_tables(sessions_df)
    tri_df.to_csv(OUTDIR / "adaptive_trials_until_switch_blocks_included.csv", index=False)
    session_means_df.to_csv(OUTDIR / "adaptive_trials_until_switch_session_means_included.csv", index=False)
    overall_stats_df.to_csv(OUTDIR / "adaptive_trials_until_switch_overall_stats_included.csv", index=False)

    pe_df = compute_pe_tables(sessions_df, window=20)
    pe_df.to_csv(OUTDIR / "adaptive_perseverative_errors_included.csv", index=False)

    pe_timecourse_df, session_timecourse_df = compute_pe_timecourse_tables(sessions_df, window=20)
    pe_timecourse_df.to_csv(OUTDIR / "adaptive_pe_timecourse_all_trials_included.csv", index=False)
    session_timecourse_df.to_csv(OUTDIR / "adaptive_pe_timecourse_session_means_included.csv", index=False)

    lat_df = compute_latency_table(sessions_df)
    lat_df.to_csv(OUTDIR / "adaptive_latency_table_included.csv", index=False)

    lat_timecourse_df, session_latency_timecourse_df = compute_latency_timecourse_tables(
        sessions_df, window=20
    )
    lat_timecourse_df.to_csv(OUTDIR / "adaptive_latency_timecourse_all_trials_included.csv", index=False)
    session_latency_timecourse_df.to_csv(
        OUTDIR / "adaptive_latency_timecourse_session_means_included.csv", index=False
    )

    df_omissions = compute_omissions_blocktype_table(sessions_df)
    df_omissions.to_csv(OUTDIR / "adaptive_omissions_per_blocktype_included.csv", index=False)

    percentcorrect_df, percentcorrect_block_df = compute_percentcorrect_tables(sessions_df)
    percentcorrect_table = percentcorrect_df
    percentcorrect_df.to_csv(OUTDIR / "adaptive_percentcorrect_per_session_included.csv", index=False)
    percentcorrect_block_df.to_csv(OUTDIR / "adaptive_percentcorrect_per_blocktype_included.csv", index=False)

    df_blocks100 = compute_blocks_per_100_table(metrics_df)
    df_blocks100.to_csv(OUTDIR / "blocks_per_100_trials_included.csv", index=False)

    df_trials_block = compute_trials_per_block_table(sessions_df)
    df_trials_block.to_csv(OUTDIR / "trials_per_block_included.csv", index=False)

    df_pe_fl = compute_pe_first_last_blocks(sessions_df, window=20)
    df_pe_fl.to_csv(OUTDIR / "pe_first_last_blocks_included.csv", index=False)

    df_stim_omit = compute_actionblock_stim_omissions(sessions_df)
    df_stim_omit.to_csv(OUTDIR / "actionblock_stim_omissions_included.csv", index=False)

    df_choice_bias = compute_actionblock_choice_bias(sessions_df)
    df_choice_bias.to_csv(OUTDIR / "actionblock_choice_bias_included.csv", index=False)

    master_stats_list = init_master_stats_store()

    # ========================================================
    # PLOTS: all animals
    # ========================================================
    plot_triangle_summary_combined(metrics_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_performance_per_session(perf_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_performance_per_blocktype(perf_block_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_trials_until_switch_blocktype(session_means_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_triangle(pe_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_timecourse(session_timecourse_df, STRAIN_COLORS)
    plot_latency_panels_by_stage(lat_df, STRAIN_COLORS)
    plot_omissions_blocktype(df_omissions, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_percentcorrect_per_session(percentcorrect_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_percentcorrect_per_blocktype(percentcorrect_block_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_blocks_per_100(df_blocks100, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_trials_per_block(df_trials_block, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_first_last_blocks(df_pe_fl, STRAIN_COLORS)
    plot_actionblock_stim_omissions(df_stim_omit, STRAIN_COLORS)
    plot_actionblock_choice_bias(df_choice_bias, STRAIN_COLORS)

    # ========================================================
    # PLOTS: strain means ± SEM only
    # ========================================================
    plot_triangle_summary_combined_by_strain(metrics_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_performance_per_session_by_strain(perf_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_performance_per_blocktype_by_strain(perf_block_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_trials_until_switch_blocktype_by_strain(session_means_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_triangle_by_strain(pe_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_timecourse_by_strain(session_timecourse_df, STRAIN_COLORS)
    plot_latency_panels_by_stage_by_strain(lat_df, STRAIN_COLORS)
    plot_omissions_blocktype_by_strain(df_omissions, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_percentcorrect_per_session_by_strain(percentcorrect_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_percentcorrect_per_blocktype_by_strain(percentcorrect_block_df, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_blocks_per_100_by_strain(df_blocks100, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_trials_per_block_by_strain(df_trials_block, STRAIN_COLORS, master_stats_list=master_stats_list)
    plot_pe_first_last_blocks_by_strain(df_pe_fl, STRAIN_COLORS)
    plot_actionblock_stim_omissions_by_strain(df_stim_omit, STRAIN_COLORS)
    plot_actionblock_choice_bias_by_strain(df_choice_bias, STRAIN_COLORS)
    plot_latency_timecourse_by_strain(session_latency_timecourse_df, STRAIN_COLORS)

    save_master_stats(master_stats_list, STATSDIR / "all_statistics_summary.csv")

    plt.show()

    print("Done.")
    return metrics_df, sessions_df, performance_table, percentcorrect_table


if __name__ == "__main__":
    metrics_df, sessions_df, performance_table, percentcorrect_table = main()
