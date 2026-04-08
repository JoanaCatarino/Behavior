from pathlib import Path
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp


# ============================================================
# STATS / ANNOTATION HELPERS FOR YOUR ADAPTIVE TASK SCRIPT
# ============================================================
# Drop this block into your script after your generic helpers.
# It assumes these globals already exist in your main script:
#   - OUTDIR
#   - STRAIN_ORDER
#   - STAGE_ORDER
#   - plot_config
#   - sem_safe
#   - get_two_stage_positions
#   - add_stage_letters
#   - add_strain_legend
# ============================================================

STATSDIR = OUTDIR / "stats"
STATSDIR.mkdir(parents=True, exist_ok=True)


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
        return {
            "n1": len(x),
            "n2": len(y),
            "U": np.nan,
            "p": np.nan,
            "note": "not enough observations",
        }

    stat, p = mannwhitneyu(x, y, alternative="two-sided")
    return {
        "n1": len(x),
        "n2": len(y),
        "U": stat,
        "p": p,
        "note": "",
    }


def safe_kruskal_from_groups(group_dict, group_order):
    used = []
    arrays = []

    for g in group_order:
        vals = pd.Series(group_dict.get(g, [])).dropna().values
        if len(vals) > 0:
            used.append(g)
            arrays.append(vals)

    if len(arrays) < 2:
        return {
            "H": np.nan,
            "p": np.nan,
            "groups_used": used,
            "note": "fewer than 2 groups with data",
        }

    stat, p = kruskal(*arrays)
    return {
        "H": stat,
        "p": p,
        "groups_used": used,
        "note": "",
    }


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
        rows.append({
            "group1": g1,
            "group2": g2,
            "p_adjusted": mat.loc[g1, g2],
        })

    return pd.DataFrame(rows)


def compute_two_stage_stats(df, value_col, stage_col="stage", strain_col="strain", strain_order=None):
    """
    For one metric:
    1) naive vs trained overall -> Mann-Whitney U
    2) naive vs trained within each strain -> Mann-Whitney U + FDR
    3) between strains within naive -> Kruskal-Wallis + Dunn
    4) between strains within trained -> Kruskal-Wallis + Dunn
    """
    out = {}

    # overall naive vs trained
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

    # naive vs trained within each strain
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

    # between strains inside naive
    naive_df = df[df[stage_col] == "naive"].copy()
    naive_groups = {
        s: naive_df.loc[naive_df[strain_col] == s, value_col].dropna().values
        for s in strain_order
    }
    kw_naive = safe_kruskal_from_groups(naive_groups, strain_order)
    out["between_strains_naive_kw"] = pd.DataFrame([{
        "stage": "naive",
        "H": kw_naive["H"],
        "p": kw_naive["p"],
        "groups_used": ", ".join(kw_naive["groups_used"]),
        "note": kw_naive["note"],
    }])
    out["between_strains_naive_dunn"] = dunn_posthoc_long(
        naive_df, group_col=strain_col, value_col=value_col, group_order=strain_order
    )

    # between strains inside trained
    trained_df = df[df[stage_col] == "trained"].copy()
    trained_groups = {
        s: trained_df.loc[trained_df[strain_col] == s, value_col].dropna().values
        for s in strain_order
    }
    kw_trained = safe_kruskal_from_groups(trained_groups, strain_order)
    out["between_strains_trained_kw"] = pd.DataFrame([{
        "stage": "trained",
        "H": kw_trained["H"],
        "p": kw_trained["p"],
        "groups_used": ", ".join(kw_trained["groups_used"]),
        "note": kw_trained["note"],
    }])
    out["between_strains_trained_dunn"] = dunn_posthoc_long(
        trained_df, group_col=strain_col, value_col=value_col, group_order=strain_order
    )

    return out


def compute_two_stage_stats_by_category(
    df,
    category_col,
    value_col,
    category_order,
    stage_col="stage",
    strain_col="strain",
    strain_order=None,
):
    """
    Same stats, repeated inside each category (for block-type plots etc.).
    """
    all_overall_rows = []
    all_per_strain_rows = []
    all_kw_rows = []
    all_dunn_rows = []

    for cat in category_order:
        sub = df[df[category_col] == cat].copy()
        if sub.empty:
            continue

        stats = compute_two_stage_stats(
            sub,
            value_col=value_col,
            stage_col=stage_col,
            strain_col=strain_col,
            strain_order=strain_order,
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

    # FDR across category-wise naive-vs-trained tests
    if not out["overall_naive_vs_trained_by_category"].empty:
        valid = out["overall_naive_vs_trained_by_category"]["p"].notna()
        if valid.any():
            reject, p_corr, _, _ = multipletests(
                out["overall_naive_vs_trained_by_category"].loc[valid, "p"],
                method="fdr_bh"
            )
            out["overall_naive_vs_trained_by_category"].loc[valid, "p_fdr"] = p_corr
            out["overall_naive_vs_trained_by_category"].loc[valid, "reject_fdr_0_05"] = reject

    return out


def save_stats_dict(stats_dict, prefix):
    saved_paths = []
    for name, df_stats in stats_dict.items():
        if isinstance(df_stats, pd.DataFrame) and not df_stats.empty:
            outpath = STATSDIR / f"{sanitize_filename(prefix)}_{name}.csv"
            df_stats.to_csv(outpath, index=False)
            saved_paths.append(outpath)
    return saved_paths


def flatten_stats_dict(stats_dict, plot_name):
    rows = []
    for table_name, df_stats in stats_dict.items():
        if not isinstance(df_stats, pd.DataFrame) or df_stats.empty:
            continue
        tmp = df_stats.copy()
        tmp.insert(0, "stats_table", table_name)
        tmp.insert(0, "plot_name", plot_name)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


# ============================================================
# AGGREGATION HELPERS FOR ONE MASTER CSV WITH ALL STATS
# ============================================================

def init_master_stats_store():
    return []


def append_stats_to_master(master_stats_list, stats_dict, plot_name):
    flat = flatten_stats_dict(stats_dict, plot_name)
    if not flat.empty:
        master_stats_list.append(flat)


def save_master_stats(master_stats_list, outpath=None):
    if outpath is None:
        outpath = STATSDIR / "all_statistics_summary.csv"

    if not master_stats_list:
        pd.DataFrame().to_csv(outpath, index=False)
        return outpath

    master_df = pd.concat(master_stats_list, ignore_index=True, sort=False)
    master_df.to_csv(outpath, index=False)
    return outpath


# ============================================================
# DROP-IN REPLACEMENTS FOR YOUR 4 GENERIC PLOTTING FUNCTIONS
# ============================================================
# IMPORTANT:
#   Replace the 4 plotting functions in your script with these.
#   Each one accepts an extra optional argument: master_stats_list
#   In main(), create: master_stats_list = init_master_stats_store()
#   Then pass master_stats_list=master_stats_list to each plot call.
#   At the very end call: save_master_stats(master_stats_list)
# ============================================================


def plot_two_stage_single_metric(
    df,
    value_col,
    ylabel,
    title,
    strain_colors,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(6, 5),
    dpi=300,
    master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats(
        df=df,
        value_col=value_col,
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER,
    )
    save_stats_dict(stats, prefix=title)
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats, title)

    overall_p = stats["overall_naive_vs_trained"]["p"].iloc[0]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x_map = {"naive": 0.85, "trained": 1.15}
    mean_x_map = {"naive": 0.93, "trained": 1.23}

    for _, row in df.iterrows():
        stage = row["stage"]
        jitter = (rng.random() - 0.5) * 0.08
        color = strain_colors.get(row["strain"], "gray")

        ax.scatter(
            x_map[stage] + jitter,
            row[value_col],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
            alpha=0.95,
        )

    for stage in STAGE_ORDER:
        d = df.loc[df["stage"] == stage, value_col].dropna()
        if len(d) == 0:
            continue

        ax.errorbar(
            mean_x_map[stage],
            d.mean(),
            yerr=sem_safe(d),
            fmt="^",
            markersize=12,
            color="black",
            capsize=4,
            linewidth=2,
            zorder=5,
        )

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="black", alpha=0.6)

    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)

    ax.set_xlim(0.7, 1.32)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1
    bracket_y = y1 - 0.10 * yr
    h = 0.03 * yr
    add_sig_bracket(
        ax,
        x_map["naive"],
        x_map["trained"],
        bracket_y,
        h,
        f"{p_to_stars(overall_p)}\n{p_to_text(overall_p)}",
        fontsize=10,
    )

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
    df,
    category_col,
    value_col,
    category_order,
    category_labels,
    ylabel,
    title,
    strain_colors,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(9, 6),
    dpi=300,
    master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()

    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats_by_category(
        df=df,
        category_col=category_col,
        value_col=value_col,
        category_order=category_order,
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER,
    )
    save_stats_dict(stats, prefix=title)
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats, title)

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

        ax.scatter(
            x0 + jitter,
            row[value_col],
            marker="^",
            s=130,
            facecolors=color,
            edgecolors=color,
            linewidths=1.8,
            alpha=0.95,
        )

    for stage in STAGE_ORDER:
        for cat in category_order:
            d = df.loc[(df["stage"] == stage) & (df[category_col] == cat), value_col].dropna()
            if len(d) == 0:
                continue

            ax.errorbar(
                stage_pos[stage][cat_to_idx[cat]] + 0.12,
                d.mean(),
                yerr=sem_safe(d),
                fmt="^",
                markersize=12,
                color="black",
                capsize=4,
                linewidth=2,
                zorder=5,
            )

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
        y = y1 - (0.10 + i * 0.08) * yr

        add_sig_bracket(
            ax,
            stage_pos["naive"][i],
            stage_pos["trained"][i],
            y,
            0.02 * yr,
            f"{p_to_stars(p)}\n{p_to_text(p)}",
            fontsize=9,
        )

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
    df,
    value_col,
    ylabel,
    title,
    strain_colors,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(8, 5),
    dpi=300,
    master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats(
        df=df,
        value_col=value_col,
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER,
    )
    save_stats_dict(stats, prefix=f"{title}_by_strain")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats, f"{title}_by_strain")

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
                linewidth=2,
            )

    if chance_line is not None:
        ax.axhline(chance_line, linestyle="--", color="black", alpha=0.6)

    if ref_line is not None:
        ax.axhline(ref_line, linestyle="--", color="black", alpha=0.7)

    ax.set_xlim(0.55, 1.45)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1

    add_sig_bracket(
        ax,
        x_map["naive"] - 0.18,
        x_map["naive"] + 0.18,
        y1 - 0.10 * yr,
        0.02 * yr,
        f"KW {p_to_stars(naive_kw_p)}\n{p_to_text(naive_kw_p)}",
        fontsize=9,
    )

    add_sig_bracket(
        ax,
        x_map["trained"] - 0.18,
        x_map["trained"] + 0.18,
        y1 - 0.23 * yr,
        0.02 * yr,
        f"KW {p_to_stars(trained_kw_p)}\n{p_to_text(trained_kw_p)}",
        fontsize=9,
    )

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
    df,
    category_col,
    value_col,
    category_order,
    category_labels,
    ylabel,
    title,
    strain_colors,
    ylim=None,
    chance_line=None,
    ref_line=None,
    figsize=(10, 6),
    dpi=300,
    master_stats_list=None,
):
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if df.empty:
        print(f"⚠ No data for {title}")
        return None, None

    stats = compute_two_stage_stats_by_category(
        df=df,
        category_col=category_col,
        value_col=value_col,
        category_order=category_order,
        stage_col="stage",
        strain_col="strain",
        strain_order=STRAIN_ORDER,
    )
    save_stats_dict(stats, prefix=f"{title}_by_strain")
    if master_stats_list is not None:
        append_stats_to_master(master_stats_list, stats, f"{title}_by_strain")

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
                d = df.loc[
                    (df["stage"] == stage)
                    & (df["strain"] == strain)
                    & (df[category_col] == cat),
                    value_col,
                ].dropna()

                if len(d) == 0:
                    continue

                x = stage_pos[stage][cat_to_idx[cat]] + strain_offset

                ax.errorbar(
                    x,
                    d.mean(),
                    yerr=sem_safe(d),
                    fmt="^",
                    markersize=11,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    capsize=4,
                    linewidth=2,
                )

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
            add_sig_bracket(
                ax,
                stage_pos["naive"][i] - 0.18,
                stage_pos["naive"][i] + 0.18,
                y1 - (0.10 + i * 0.10) * yr,
                0.015 * yr,
                f"N: {p_to_stars(p)}",
                fontsize=8,
            )

        if not sub_trained.empty:
            p = sub_trained["p"].iloc[0]
            add_sig_bracket(
                ax,
                stage_pos["trained"][i] - 0.18,
                stage_pos["trained"][i] + 0.18,
                y1 - (0.16 + i * 0.10) * yr,
                0.015 * yr,
                f"T: {p_to_stars(p)}",
                fontsize=8,
            )

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
# MAIN CHANGES YOU SHOULD MAKE IN YOUR EXISTING SCRIPT
# ============================================================
# 1) Add this near the start of main():
#       master_stats_list = init_master_stats_store()
#
# 2) Change every plot call that uses one of the 4 generic wrappers so it passes:
#       master_stats_list=master_stats_list
#
#    Example:
#       plot_performance_per_session(perf_df, STRAIN_COLORS, master_stats_list=master_stats_list)
#
# 3) At the end of main(), before plt.show():
#       save_master_stats(master_stats_list, STATSDIR / "all_statistics_summary.csv")
#
# 4) Update wrapper functions so they accept master_stats_list=None and forward it.
#    Example:
#       def plot_performance_per_session(perf_df, strain_colors, master_stats_list=None):
#           return plot_two_stage_single_metric(..., master_stats_list=master_stats_list)
# ============================================================
