# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:32:33 2026

@author: JoanaCatarino
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
h5_file = 'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/out_dlc/999770/day3/999770_day32025-12-06T10_04_09DLC_Resnet50_test_manul_wholevideoMar12shuffle1_snapshot_040.h5'

fps = 30
start_s = 1000
duration_s = 10
likelihood_thresh = 0.5
smooth_window = 7
save_name = "dlc_clean_regions"

colors = {
    "Nose":   "#4C78A8",
    "Eye":    "#B279A2",
    "Pupil":  "#72B7B2",
    "Tongue": "#E45756",
}

# =========================================================
# LOAD
# =========================================================
df = pd.read_hdf(h5_file)
scorer = df.columns.get_level_values(0)[0]

start_f = int(start_s * fps)
end_f = int((start_s + duration_s) * fps)
snippet = df.iloc[start_f:end_f].copy()
t = np.arange(len(snippet)) / fps

# =========================================================
# HELPERS
# =========================================================
def smooth_nan(x, window=5):
    if window <= 1:
        return x.copy()
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()

def interp_short_gaps(x, max_gap=2):
    s = pd.Series(x)
    isnan = s.isna()
    if not isnan.any():
        return s.to_numpy()

    groups = (isnan != isnan.shift()).cumsum()
    gap_sizes = isnan.groupby(groups).transform("sum")

    out = s.interpolate(limit=max_gap, limit_direction="both")
    out[(isnan) & (gap_sizes > max_gap)] = np.nan
    return out.to_numpy()

def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isnan(mad) or mad == 0:
        sd = np.nanstd(x)
        if np.isnan(sd) or sd == 0:
            return np.zeros_like(x)
        return (x - med) / sd
    return (x - med) / (1.4826 * mad)

def get_xy(bodypart):
    x = snippet[(scorer, bodypart, "x")].to_numpy(dtype=float)
    y = snippet[(scorer, bodypart, "y")].to_numpy(dtype=float)
    l = snippet[(scorer, bodypart, "likelihood")].to_numpy(dtype=float)

    x[l < likelihood_thresh] = np.nan
    y[l < likelihood_thresh] = np.nan

    x = interp_short_gaps(x, max_gap=2)
    y = interp_short_gaps(y, max_gap=2)
    return x, y

def finalize_signal(x):
    x = smooth_nan(x, smooth_window)
    return robust_zscore(x)

def valid_fraction(x):
    return np.mean(~np.isnan(x))

# =========================================================
# SIGNALS
# =========================================================

# Nose: use nosecenter vertical displacement
_, y_nose = get_xy("nosecenter")
nose = finalize_signal(-y_nose)

# Eye: eyelid opening
_, y_eyetop = get_xy("eyeltop")
_, y_eyebottom = get_xy("eyebottom")
eye = finalize_signal(np.abs(y_eyebottom - y_eyetop))

# Pupil: average of horizontal and vertical diameters
x_pl, _ = get_xy("pupilleft")
x_pr, _ = get_xy("pupilright")
_, y_pt = get_xy("pupiltop")
_, y_pb = get_xy("pupilbottom")

pupil_h = np.abs(x_pr - x_pl)
pupil_v = np.abs(y_pb - y_pt)
pupil = finalize_signal((pupil_h + pupil_v) / 2)

# Tongue: distance from mouth to tonguetip
x_mouth, y_mouth = get_xy("mouth")
x_tip, y_tip = get_xy("tonguetip")
tongue = np.sqrt((x_tip - x_mouth)**2 + (y_tip - y_mouth)**2)
tongue = finalize_signal(tongue)

signals = {
    "Nose": nose,
    "Eye": eye,
    "Pupil": pupil,
    "Tongue": tongue,
}

# =========================================================
# OPTIONAL: hide bad signals automatically
# =========================================================
# If a signal has too little valid data, replace with NaNs so it doesn't fake a flat line.
raw_signals_for_validity = {
    "Nose": y_nose,
    "Eye": np.abs(y_eyebottom - y_eyetop),
    "Pupil": (pupil_h + pupil_v) / 2,
    "Tongue": np.sqrt((x_tip - x_mouth)**2 + (y_tip - y_mouth)**2),
}

for name, raw in raw_signals_for_validity.items():
    if valid_fraction(raw) < 0.25:
        signals[name] = np.full_like(signals[name], np.nan)
        print(f"{name}: too little valid tracking in this snippet")

# =========================================================
# PLOT
# =========================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.4, 3.0))

order = ["Nose", "Eye", "Pupil", "Tongue"]
offset_step = 2.2

yticks = []
yticklabels = []

for i, name in enumerate(order):
    offset = (len(order) - 1 - i) * offset_step
    ax.plot(t, signals[name] + offset, color=colors[name], lw=1.6, solid_capstyle="round")
    yticks.append(offset)
    yticklabels.append(name)

ax.set_xlim(t[0], t[-1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Time (s)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
fig.savefig(f"{save_name}.svg", bbox_inches="tight")
fig.savefig(f"{save_name}.png", dpi=400, bbox_inches="tight")

plt.show()





#%%



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
h5_file = 'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/out_dlc/999770/day3/999770_day32025-12-06T10_04_09DLC_Resnet50_test_manul_wholevideoMar12shuffle1_snapshot_040.h5'

fps = 30
start_s = 200
duration_s =10

general_thresh = 0.5
tongue_thresh = 0.05
smooth_window = 7

colors = {
    "Nose":   "#4C78A8",
    "Eye":    "#B279A2",
    "Pupil":  "#72B7B2",
    "Tongue": "#E45756",
}

# =========================================================
# LOAD
# =========================================================
df = pd.read_hdf(h5_file)
scorer = df.columns.get_level_values(0)[0]

start_f = int(start_s * fps)
end_f = int((start_s + duration_s) * fps)
snippet = df.iloc[start_f:end_f].copy()
t = np.arange(len(snippet)) / fps

# =========================================================
# HELPERS
# =========================================================
def smooth_nan(x, window=5):
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()

def interp_short_gaps(x, max_gap=3):
    s = pd.Series(x)
    isnan = s.isna()
    if not isnan.any():
        return s.to_numpy()

    groups = (isnan != isnan.shift()).cumsum()
    gap_sizes = isnan.groupby(groups).transform("sum")

    out = s.interpolate(limit=max_gap, limit_direction="both")
    out[(isnan) & (gap_sizes > max_gap)] = np.nan
    return out.to_numpy()

def robust_zscore(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.full_like(x, np.nan)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if np.isnan(mad) or mad == 0:
        sd = np.nanstd(x)
        if np.isnan(sd) or sd == 0:
            return np.full_like(x, np.nan)
        return (x - med) / sd

    return (x - med) / (1.4826 * mad)

def finalize_signal(x):
    x = interp_short_gaps(x, max_gap=3)
    x = smooth_nan(x, smooth_window)
    x = robust_zscore(x)
    return x

def get_xy(bodypart, thresh):
    x = snippet[(scorer, bodypart, "x")].to_numpy(dtype=float)
    y = snippet[(scorer, bodypart, "y")].to_numpy(dtype=float)
    l = snippet[(scorer, bodypart, "likelihood")].to_numpy(dtype=float)

    x[l < thresh] = np.nan
    y[l < thresh] = np.nan
    return x, y, l

# =========================================================
# SIGNALS
# =========================================================

# Nose
_, y_nose, l_nose = get_xy("nosecenter", general_thresh)
nose = finalize_signal(-y_nose)

# Eye opening
_, y_eyetop, l_eyetop = get_xy("eyeltop", general_thresh)
_, y_eyebottom, l_eyebottom = get_xy("eyebottom", general_thresh)
eye_raw = np.abs(y_eyebottom - y_eyetop)
eye = finalize_signal(eye_raw)

# Pupil size
x_pl, _, l_pl = get_xy("pupilleft", general_thresh)
x_pr, _, l_pr = get_xy("pupilright", general_thresh)
_, y_pt, l_pt = get_xy("pupiltop", general_thresh)
_, y_pb, l_pb = get_xy("pupilbottom", general_thresh)

pupil_raw = (np.abs(x_pr - x_pl) + np.abs(y_pb - y_pt)) / 2
pupil = finalize_signal(pupil_raw)

# Tongue protrusion = distance mouth -> tonguetip
x_m, y_m, l_m = get_xy("mouth", tongue_thresh)
x_t, y_t, l_t = get_xy("tonguetip", tongue_thresh)

valid_tongue = (l_m >= tongue_thresh) & (l_t >= tongue_thresh)
tongue_raw = np.full(len(x_m), np.nan)
tongue_raw[valid_tongue] = np.sqrt(
    (x_t[valid_tongue] - x_m[valid_tongue])**2 +
    (y_t[valid_tongue] - y_m[valid_tongue])**2
)
tongue = finalize_signal(tongue_raw)

signals = {
    "Nose": nose,
    "Eye": eye,
    "Pupil": pupil,
    "Tongue": tongue,
}

validity = {
    "Nose": np.mean(l_nose >= general_thresh),
    "Eye": min(np.mean(l_eyetop >= general_thresh), np.mean(l_eyebottom >= general_thresh)),
    "Pupil": min(np.mean(l_pl >= general_thresh), np.mean(l_pr >= general_thresh),
                 np.mean(l_pt >= general_thresh), np.mean(l_pb >= general_thresh)),
    "Tongue": np.mean(valid_tongue),
}

print("Signal validity:")
for k, v in validity.items():
    print(f"{k}: {v:.3f}")

# =========================================================
# PLOT
# =========================================================
order = ["Nose", "Eye", "Pupil", "Tongue"]
good_order = [name for name in order if not np.all(np.isnan(signals[name]))]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
offset_step = 2.2

yticks = []
yticklabels = []

for i, name in enumerate(good_order):
    offset = (len(good_order) - 1 - i) * offset_step
    ax.plot(t, signals[name] + offset, color=colors[name], lw=1.8, solid_capstyle="round")
    yticks.append(offset)
    yticklabels.append(name)

ax.set_xlim(t[0], t[-1])
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xlabel("Time (s)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()

save_dir = r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn'
file_base = f"dlc_snippet_{start_s}s_{duration_s}s"

fig.savefig(f"{save_dir}/{file_base}.pdf", bbox_inches="tight")
fig.savefig(f"{save_dir}/{file_base}.svg", bbox_inches="tight")
fig.savefig(f"{save_dir}/{file_base}.png", dpi=400, bbox_inches="tight")