#!/usr/bin/env python3
"""Speed x number of frames: where does a longer window still help?

  python scripts/analyze_tsweep.py results --out results/report_tsweep_speed

All T are scored on the same test windows (frame index >= max T - 1), so the windows do not differ with T.
Writes mae_speed_T.csv (model, T, speed bin, MAE mean/std over seeds), best_T.csv, and plots.
Physical reference (BEV, 10 Hz): displacement per frame = v / 10 [m] = v * 20 [px at 5 mm/px];
the BEV covers 0.30-1.00 m ahead, so a floor point stays in view for 0.70 / (v / 10) frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FPS, BEV_DEPTH = 10.0, 0.70
BINS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="early2d,c3d")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted(Path(a.results).glob("*_T*_s*")):
        if not (d / "summary.json").exists():
            continue
        s = json.load(open(d / "summary.json"))
        if s["model"] in a.models.split(",") and s.get("input", "bev") == "bev":
            runs.append((s, d))
    tmax = max(s["T"] for s, _ in runs)
    rows = []
    for s, d in runs:
        df = pd.read_csv(d / "test_pred.csv")
        df = df[df.frame >= tmax - 1]                 # the same windows for every T
        b = pd.cut(df.v, BINS, include_lowest=True, labels=[f"{x:.1f}-{x + 0.1:.1f}" for x in BINS[:-1]])
        for lab, e in df.err.abs().groupby(b, observed=False):
            rows.append(dict(model=s["model"], T=s["T"], seed=s["seed"], speed=str(lab), mae=e.mean(), n=len(e)))
        rows.append(dict(model=s["model"], T=s["T"], seed=s["seed"], speed="all", mae=df.err.abs().mean(), n=len(df)))
    raw = pd.DataFrame(rows)
    tab = raw.groupby(["model", "T", "speed"]).agg(mae=("mae", "mean"), std=("mae", "std"), seeds=("seed", "nunique"),
                                                   n=("n", "first")).reset_index()
    tab.to_csv(out / "mae_speed_T.csv", index=False)

    # smallest T whose MAE is within 5 % of the best T, per model and speed bin
    best = []
    for (m, sp), g in tab.groupby(["model", "speed"]):
        g = g.sort_values("T")
        bmin = g.mae.min()
        best.append(dict(model=m, speed=sp, best_T=int(g.loc[g.mae.idxmin(), "T"]), best_mae=bmin,
                         enough_T=int(g[g.mae <= bmin * 1.05]["T"].min())))
    best = pd.DataFrame(best)
    best.to_csv(out / "best_T.csv", index=False)

    speeds = [f"{x:.1f}-{x + 0.1:.1f}" for x in BINS[:-1]]
    with open(out / "summary.md", "w") as f:
        for m in a.models.split(","):
            t = tab[tab.model == m].pivot(index="speed", columns="T", values="mae").reindex(speeds + ["all"])
            f.write(f"\n### {m}: MAE [m/s] (rows: speed [m/s], columns: T; mean over seeds)\n\n")
            f.write("| speed | " + " | ".join(f"T={c}" for c in t.columns) + " | best T | enough T |\n")
            f.write("|---" * (len(t.columns) + 3) + "|\n")
            bm = best[best.model == m].set_index("speed")
            for sp, r in t.iterrows():
                f.write(f"| {sp} | " + " | ".join(f"{v:.3f}" for v in r.values) +
                        f" | {bm.best_T.get(sp, '')} | {bm.enough_T.get(sp, '')} |\n")
        f.write(f"\nwindows: frame >= {tmax - 1} only; enough T = smallest T within 5 % of the best\n")
    print(open(out / "summary.md").read())

    # plots: MAE vs T per speed bin, and best T vs speed with the field-of-view limit
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, m in zip(axs, a.models.split(",")):
        for i, sp in enumerate(speeds):
            g = tab[(tab.model == m) & (tab.speed == sp)].sort_values("T")
            ax.plot(g["T"], g.mae, "o-", color=cmap(i / (len(speeds) - 1)), label=f"{sp} m/s", ms=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(tab["T"].unique()), [str(t) for t in sorted(tab["T"].unique())])
        ax.set_title(m); ax.set_xlabel("frames T (10 Hz)"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("MAE [m/s]")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "mae_vs_T.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    mid = BINS[:-1] + 0.05
    for m, c in zip(a.models.split(","), ["tab:blue", "tab:orange"]):
        bm = best[best.model == m].set_index("speed").reindex(speeds)
        ax.plot(mid, bm.enough_T, "o-", color=c, label=f"{m}: enough T")
    v = np.linspace(0.05, 1.0, 100)
    ax.plot(v, BEV_DEPTH / (v / FPS), "k--", lw=1, label="frames a floor point stays in the BEV")
    ax.set_yscale("log", base=2); ax.set_ylim(1, 64)
    ax.set_xlabel("speed [m/s]"); ax.set_ylabel("frames"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "enough_T.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
