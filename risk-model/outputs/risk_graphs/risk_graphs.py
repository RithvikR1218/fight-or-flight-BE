"""Static charts of how a tile's 0-1 risk is calculated, plus the cell heatmap
at four hours. Writes PNGs into outputs/risk_graphs/ (or --out).

Usage (from the repo root):
    python -m algorithm.scoring.risk_graphs
    python -m algorithm.scoring.risk_graphs --cache path/to/risk_cache.json --out some/dir

Charts:
    1_severity.png         per-crime severity (0-10) by crime type, from crime.db
    2_recency.png          recency weight 1/(1+age/tau) and its exponential-mix fit
    3_time_of_day.png      time-of-day gaussian weight around the scored hour
    4_risk_scale.png       model probability -> everyday base risk (log scale, 0..0.85)
    5_danger_boost.png     recent serious crimes -> boost, and base + (1-base)*boost
    6_bluelight.png        score drop vs distance-weighted phone count, and vs distance
    7_risk_distribution.png  served risk over all tiles at 2 AM / noon / 10 PM
    8_hourly_profiles.png  24-hour risk at a few named places
    9_cell_heatmap.png     every H3 tile's risk at 2 AM, 10 AM, 6 PM and 10 PM
"""
import argparse
import json
import sqlite3
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from algorithm.common import config, geo
from algorithm.features.build_dataset import recency_mixture, recency_weight, tod_kernel
from algorithm.scoring import risk_scale

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, RED, ORANGE, GREEN = "#2563eb", "#dc2626", "#ea580c", "#047857"
# Same colour ramp as the interactive map (risk_map.py).
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", [(0, "#fdf1e8"), (0.35, "#f6c39c"), (0.6, "#ef8a5a"), (0.85, "#d6304b"),
             (0.9, "#8a1155"), (1.0, "#1a0b24")])

PLACES = {
    "Tommy Trojan (campus, 0 for routing)": (34.0205, -118.2856),
    "Figueroa & Adams": (34.0326, -118.2741),
    "Vermont & Exposition": (34.0183, -118.2915),
    "MacArthur Park": (34.0577, -118.2770),
    "7th St/Metro Center": (34.0486, -118.2587),
    "Leimert Park": (34.0094, -118.3321),
}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9ca3af")
    ax.tick_params(colors=MUTED)


def _save(fig, out, name):
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    print("wrote", p)


def severity_chart(out):
    from algorithm.common.severity import incident_severity
    con = sqlite3.connect(config.DB_PATH)
    rows = con.execute("SELECT crm_cd, crm_cd_desc, weapon_used_cd FROM incidents").fetchall()
    con.close()
    sev = np.array([incident_severity(a, b, c) for a, b, c in rows])
    by_desc = {}
    for (cd, desc, _), s in zip(rows, sev):
        d = by_desc.setdefault(desc or "?", [0, 0.0]); d[0] += 1; d[1] += s
    top = sorted(by_desc.items(), key=lambda kv: -kv[1][0])[:18]
    top = sorted(top, key=lambda kv: kv[1][1] / kv[1][0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    names = [k[:42] for k, _ in top]
    vals = [v[1] / v[0] for _, v in top]
    axes[0].barh(names, vals, color=[RED if v >= config.DANGER_MIN_SEVERITY else ORANGE for v in vals])
    axes[0].axvline(config.DANGER_MIN_SEVERITY, color=INK, ls="--", lw=1)
    axes[0].text(config.DANGER_MIN_SEVERITY + 0.1, 0.2, "danger-boost threshold", color=INK, fontsize=8.5)
    _style(axes[0], "Mean severity of the 18 most common crime types", "severity (0-10; armed x1.5, cap 6)", "")
    axes[0].tick_params(axis="y", labelsize=8)
    bins = np.arange(0, 11, 0.5)
    axes[1].hist(sev, bins=bins, color=BLUE, edgecolor="white")
    _style(axes[1], f"Severity of all {len(sev):,} incidents", "severity", "incidents")
    _save(fig, out, "1_severity.png")


def recency_chart(out):
    age = np.logspace(-1, np.log10(1500), 400)
    mix = recency_mixture()
    fit = sum(a * 0.5 ** (age / hl) for hl, a in mix.items())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(age, recency_weight(age), color=BLUE, lw=2.5, label=f"target 1 / (1 + age / {config.RECENCY_TAU_DAYS:g} d)")
    ax.plot(age, fit, color=ORANGE, lw=1.5, ls="--", label="fit: mix of exponentials (used by the model)")
    for d, lab in [(1, "1 day"), (7, "1 week"), (30, "1 month"), (365, "1 year")]:
        w = float(recency_weight(d)); ax.scatter([d], [w], color=INK, s=18, zorder=3)
        ax.annotate(f"{lab}: {w:.2f}", (d, w), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.legend(frameon=False, fontsize=9)
    _style(ax, "Recency: how much a past crime still counts", "age of crime (days, log scale)", "weight")
    _save(fig, out, "2_recency.png")


def tod_chart(out):
    K = tod_kernel()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for h, c in [(2, "#6366f1"), (12, ORANGE), (22, RED)]:
        ax.plot(range(24), K[h], marker="o", ms=3.5, lw=2, color=c, label=f"scoring {h:02d}:00")
    ax.set_xticks(range(0, 24, 2)); ax.legend(frameon=False, fontsize=9)
    _style(ax, f"Time of day: crimes near the scored hour count more (sigma = {config.TOD_SIGMA_HOURS:g} h, wraps midnight)",
           "hour the past crime happened", "weight")
    _save(fig, out, "3_time_of_day.png")


def scale_chart(out, meta):
    lo, hi = risk_scale.anchors_from_meta(meta)
    p = np.logspace(np.log10(lo) - 1, np.log10(hi) + 0.5, 400)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(p, risk_scale.base_risk(p, meta), color=BLUE, lw=2.5, label="base risk (log scale)")
    if "risk_quantiles" in meta:
        ax.plot(p, risk_scale.percentile(p, meta), color=MUTED, lw=1.5, ls=":", label="old score: 2023 percentile")
    for x, lab in [(lo, f"p5 of 2023\n{lo:.1e}"), (hi, f"p99.9 of 2023\n{hi:.3f}")]:
        ax.axvline(x, color=INK, ls="--", lw=1); ax.text(x * 1.15, 0.08, lab, fontsize=8.5, color=INK)
    ax.axhline(config.RISK_BASE_MAX, color=RED, lw=1, ls="--")
    ax.text(p[0] * 1.2, config.RISK_BASE_MAX + 0.02, f"cap {config.RISK_BASE_MAX} without a danger boost", color=RED, fontsize=8.5)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=9, loc="center left")
    _style(ax, "Model probability -> everyday base risk", "calibrated P(incident in tile this hour), log scale", "base risk")
    _save(fig, out, "4_risk_scale.png")


def boost_chart(out):
    x = np.linspace(0, 2.5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].plot(x, risk_scale.boost_from_x(x), color=RED, lw=2.5)
    for xx, lab in [(1.0, "1 robbery (sev 5) in the tile, just now"), (0.5, "same, 1 ring away"), (0.25, "same, 2 rings away")]:
        b = float(risk_scale.boost_from_x(xx)); axes[0].scatter([xx], [b], color=INK, s=18, zorder=3)
        axes[0].annotate(f"{lab}: {b:.2f}", (xx, b), xytext=(8, -12), textcoords="offset points", fontsize=8.5)
    _style(axes[0], f"Danger boost = 1 - exp(-x / {config.DANGER_SCALE})", "x = sum sev/5 x ring weight x exp(-age_h / 6)", "boost")
    b = np.linspace(0, 1, 200)
    for base, c in [(0.2, "#93c5fd"), (0.5, BLUE), (0.85, "#1e3a8a")]:
        axes[1].plot(b, risk_scale.combine(base, b), color=c, lw=2, label=f"base {base}")
    axes[1].axhline(0.9, color=RED, ls="--", lw=1); axes[1].text(0.02, 0.915, "> 0.9 = immediate danger", color=RED, fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1], "Served risk = base + (1 - base) x boost", "boost", "served risk")
    hrs = np.linspace(0, 24, 200)
    ax2 = axes[0].inset_axes([0.58, 0.12, 0.38, 0.38])
    ax2.plot(hrs, np.exp(-hrs / config.DANGER_TAU_HOURS), color=ORANGE); ax2.set_title("age decay (h)", fontsize=8)
    ax2.tick_params(labelsize=7)
    _save(fig, out, "5_danger_boost.png")


def bluelight_chart(out):
    n = np.linspace(0, 7, 300)
    drop = np.minimum(config.BLUELIGHT_SCORE_DROP_PER_PHONE * n, config.BLUELIGHT_SCORE_DROP_MAX)
    d = np.linspace(0, 600, 300)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].plot(n, drop, color=BLUE, lw=2.5)
    _style(axes[0], f"Score drop = min({config.BLUELIGHT_SCORE_DROP_PER_PHONE} x phones, {config.BLUELIGHT_SCORE_DROP_MAX})",
           "distance-weighted phone count  sum exp(-d / 120 m)", "subtracted from base risk")
    for k, c in [(1, "#93c5fd"), (2, BLUE), (4, "#1e3a8a")]:
        axes[1].plot(d, np.minimum(config.BLUELIGHT_SCORE_DROP_PER_PHONE * k * np.exp(-d / config.BLUELIGHT_RANGE_M),
                                   config.BLUELIGHT_SCORE_DROP_MAX), color=c, lw=2, label=f"{k} phone{'s' if k > 1 else ''} at that distance")
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1], "Score drop vs distance to the phones", "distance (m)", "score drop")
    _save(fig, out, "6_bluelight.png")


def distribution_chart(out, cache):
    R = np.array([v["risk"] for v in cache["cells"].values()])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bins = np.linspace(0, 1, 51)
    for h, c in [(2, "#6366f1"), (12, ORANGE), (22, RED)]:
        ax.hist(R[:, h], bins=bins, histtype="step", lw=2, color=c, label=f"{h:02d}:00 (median {np.median(R[:, h]):.2f})")
    ax.axvline(0.9, color=INK, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, f"Served risk over {len(R):,} tiles on {cache['date']}", "risk", "tiles")
    _save(fig, out, "7_risk_distribution.png")


def profiles_chart(out, cache, zero):
    res = int(cache["h3_res"])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    cols = ["#047857", "#dc2626", "#ea580c", "#7c3aed", "#0891b2", "#be185d"]
    for (name, (lat, lon)), c in zip(PLACES.items(), cols):
        cell = h3.latlng_to_cell(lat, lon, res)
        v = cache["cells"].get(cell)
        if v is None:
            continue
        r = [0.0] * 24 if cell in zero else v["risk"]
        ax.plot(range(24), r, lw=2, color=c, label=name)
    ax.axhline(0.9, color=INK, ls="--", lw=1)
    ax.set_xticks(range(0, 24, 2)); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax, f"24-hour risk at a few places ({cache['date']})", "hour", "risk")
    _save(fig, out, "8_hourly_profiles.png")


def heatmap_chart(out, cache, zero):
    cells = list(cache["cells"])
    R = np.array([cache["cells"][c]["risk"] for c in cells])
    zmask = np.array([c in zero for c in cells])
    R[zmask] = 0.0
    kx = np.cos(np.radians(config.CENTER[0]))
    poly = lambda c: [((lng - config.CENTER[1]) * kx * 111.32, (lat - config.CENTER[0]) * 110.54)
                      for lat, lng in h3.cell_to_boundary(c)]
    P = [poly(c) for c in cells]
    ND = [poly(c) for c in cache.get("no_data_cells", [])]
    phones = np.array([((b - config.CENTER[1]) * kx * 111.32, (a - config.CENTER[0]) * 110.54)
                       for a, b, _ in geo.bluelight_phones()])
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), gridspec_kw={"hspace": 0.08, "wspace": 0.02})
    for ax, h in zip(axes.flat, [2, 10, 18, 22]):
        ax.add_collection(PolyCollection(ND, facecolor="#d4d4d4", edgecolor="none"))
        pc = PolyCollection(P, array=R[:, h], cmap=RISK_CMAP, edgecolor="none")
        pc.set_clim(0, 1); ax.add_collection(pc)
        ax.add_collection(PolyCollection([p for p, z in zip(P, zmask) if z], facecolor="none", edgecolor=GREEN, lw=0.6))
        ax.scatter(phones[:, 0], phones[:, 1], s=1.5, color=BLUE)
        lim = config.RADIUS_M / 1000 + 0.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{h:02d}:00  (median {np.median(R[~zmask, h]):.2f}, {np.mean(R[:, h] > 0.9) * 100:.2f}% > 0.9)",
                     fontsize=12, color=INK, loc="left")
    cb = fig.colorbar(pc, ax=axes, shrink=0.5, pad=0.02, label="risk (0-1)")
    cb.set_ticks([0, 0.3, 0.6, 0.85, 0.9, 1])
    fig.suptitle(f"Risk per H3 res-{cache['h3_res']} tile, {cache['date']} (history as of {cache.get('history_as_of')}). "
                 "Grey = no LAPD data, green = USC campus/Village (0), blue = blue-light phones",
                 fontsize=11.5, color=INK, x=0.03, ha="left", y=0.93)
    p = out / "9_cell_heatmap.png"
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="risk_cache.json (default: config.CACHE_PATH)")
    ap.add_argument("--out", default=None, help="default: outputs/risk_graphs next to the risk model")
    args = ap.parse_args()
    cache = json.loads(Path(args.cache or config.CACHE_PATH).read_text())
    meta = json.loads(config.META_PATH.read_text())
    out = Path(args.out) if args.out else config.ROOT / "outputs" / "risk_graphs"
    out.mkdir(parents=True, exist_ok=True)
    from algorithm.scoring.risk_map import zero_risk_cells
    zero = zero_risk_cells(cache.get("h3_res", config.H3_RES))
    severity_chart(out); recency_chart(out); tod_chart(out)
    scale_chart(out, meta); boost_chart(out); bluelight_chart(out)
    distribution_chart(out, cache); profiles_chart(out, cache, zero); heatmap_chart(out, cache, zero)


if __name__ == "__main__":
    main()
