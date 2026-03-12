# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 16:26:02 2026

@author: JoanaCatarino
"""

# colors.py
"""
Centralized color definitions for the PFC–Str behavior project.

Import in scripts with:
    from colors import STRAIN_COLORS
"""

# =========================================================
# STRAIN COLORS (used for dots in plots)
# =========================================================

STRAIN_COLORS = {
    "Tlx3":        "#020202",   # black
    "Fezf2":       "#6D6A71",   # gray
    "Fmr1_Fezf2":  "#185856",   # blue
    "Fmr1_Tlx3":   "#341F39"    # purple
}

# =========================================================
# TASK COLORS (optional, for bar plots by task)
# =========================================================

TASK_COLORS = {
    "free_licking":          "#9DB8AE",
    "two_choice":            "#5B9E9A",
    "two_choice_blocks":     "#9490B4",
    "adaptive_sensorimotor": "#815D9F",
}

# =========================================================
# COHORT COLORS (if you ever want to separate cohorts visually)
# =========================================================

COHORT_COLORS = {
    1: "#2E706C",   # teal
    2: "#C7651B",   # orange
}

# =========================================================
# BLOCK TYPE COLORS (for 10 / 5 / 3 trial blocks)
# =========================================================

BLOCK_COLORS = {
    10: "#5B9E9A",
    5:  "#ED7D31",
    3:  "#6A5ACD",
}
