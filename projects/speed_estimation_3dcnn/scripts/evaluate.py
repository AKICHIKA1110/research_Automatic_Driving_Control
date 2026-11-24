#!/usr/bin/env python3
"""Compare trained models by condition (from their test_pred.csv).

  python scripts/evaluate.py results/single2d_T8 results/early2d_T8 results/c3d_T8 --out results/report_T8

Writes <out>/by_condition.csv, <out>/by_condition.md and plots (<out>/*.png).
Conditions:
  line      dashed center line / solid lines only
  seams     floor-board seams (move with the floor) / none
  road      straight (|kappa| < 0.05 1/m) / curve
  hard      solid lines, no seams, straight: only the line shape could tell the speed, and it does not change
  speed     0-0.1 / 0.1-0.4 / 0.4-0.7 / 0.7-1.0 m/s
  alias     dashed lines only: displacement per frame / dash period (> 0.5 = beyond Nyquist, wagon-wheel effect)
  accel     |a| > 0.3 m/s^2 (speed changing) / steady
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

FPS = 10.0


def conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    straight = df.kappa.abs() < 0.05
    ratio = df.v / FPS / df.dash_period
    c = {
        "all": pd.Series(True, index=df.index),
        "line: dashed": df.dashed, "line: solid only": ~df.dashed,
        "seams: yes": df.seams, "seams: no": ~df.seams,
        "road: straight": straight, "road: curve": ~straight,
        "hard: solid+no seams+straight": ~df.dashed & ~df.seams & straight,
        "speed 0-0.1": df.v < 0.1, "speed 0.1-0.4": (df.v >= 0.1) & (df.v < 0.4),
        "speed 0.4-0.7": (df.v >= 0.4) & (df.v < 0.7), "speed 0.7-1.0": df.v >= 0.7,
        "alias: dashed, ratio<0.4": df.dashed & (ratio < 0.4),
        "alias: dashed, ratio 0.4-0.5": df.dashed & (ratio >= 0.4) & (ratio < 0.5),
        "alias: dashed, ratio>=0.5": df.dashed & (ratio >= 0.5),
        "accel: |a|>0.3": df.accel.abs() > 0.3, "accel: steady": df.accel.abs() <= 0.3,
    }
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="result dirs (several seeds of one model/T are averaged)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    for r in a.runs:
        s = json.load(open(Path(r) / "summary.json"))
        name = f"{s['model']} T{s['T']}" + (" cam" if s.get("input") == "cam" else "") + (f" lr{s['lr']:g}" if s.get("lr", 2e-3) != 2e-3 else "")
        groups.setdefault(name, []).append((s, pd.read_csv(Path(r) / "test_pred.csv")))
    base = next(iter(groups.values()))[0][1]
    const = base.v.mean()   # reference: always answer the mean speed of the test set
    cond_names = list(conditions(base).keys())

    def fmt(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            return "-"
        return f"{np.mean(vals):.3f}" + (f" ± {np.std(vals):.3f}" if len(vals) > 1 else "")

    rows = []
    for cname in cond_names:
        mask = conditions(base)[cname]
        row = {"condition": cname, "n": int(mask.sum()),
               "const": f"{(base.v[mask] - const).abs().mean():.3f}" if mask.any() else "-"}
        for name, runs in groups.items():
            vals = []
            for _, df in runs:
                m = conditions(df)[cname]
                vals.append(float(df.err[m].abs().mean()) if m.any() else np.nan)
            row[name] = fmt(vals)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "by_condition.csv", index=False)
    with open(out / "by_condition.md", "w") as f:
        f.write("| model | params | MMAC | seeds | test MAE [m/s] | test RMSE [m/s] |\n|---|---|---|---|---|---|\n")
        for name, runs in groups.items():
            s0 = runs[0][0]
            f.write(f"| {name} | {s0['params'] / 1e3:.0f}k | {s0['macs'] / 1e6:.0f} | {len(runs)} | "
                    f"{fmt([r[0]['test_mae'] for r in runs])} | {fmt([r[0]['test_rmse'] for r in runs])} |\n")
        f.write("\nMAE [m/s] by condition (mean ± std over seeds; const = always the mean speed)\n\n")
        cols = list(tab.columns)
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in tab.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    print(open(out / "by_condition.md").read())

    first = {name: runs[0][1] for name, runs in groups.items()}   # plots: first seed of each model
    fig, axs = plt.subplots(1, len(first), figsize=(4 * len(first), 4), squeeze=False)
    for ax, (name, df) in zip(axs[0], first.items()):
        ax.hist2d(df.v, df.pred, bins=60, range=[[0, 1.05], [-0.1, 1.15]], cmap="viridis", cmin=1)
        ax.plot([0, 1.05], [0, 1.05], "r--", lw=1)
        ax.set_title(f"{name}  MAE {df.err.abs().mean():.3f}", fontsize=10)
        ax.set_xlabel("true speed [m/s]"); ax.set_ylabel("predicted [m/s]")
    fig.tight_layout(); fig.savefig(out / "pred_vs_true.png", dpi=110)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 0.8, 17)
    for name, runs in groups.items():
        curves = []
        for _, df in runs:
            d = df[df.dashed]
            idx = np.digitize(d.v / FPS / d.dash_period, bins)
            curves.append([d.err[idx == i].abs().mean() if (idx == i).sum() > 30 else np.nan
                           for i in range(1, len(bins))])
        ax.plot((bins[:-1] + bins[1:]) / 2, np.nanmean(curves, 0), "o-", label=name, ms=3)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("displacement per frame / dash period"); ax.set_ylabel("MAE [m/s]")
    ax.set_title("dashed courses: error vs aliasing ratio", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "alias.png", dpi=110)


if __name__ == "__main__":
    main()
