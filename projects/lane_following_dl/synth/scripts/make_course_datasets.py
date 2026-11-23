#!/usr/bin/env python3
"""Generate one dataset per course, each saved in its own folder.

  python3 scripts/make_course_datasets.py                       # all config/courses/*.yaml → data/courses/<course>/
  python3 scripts/make_course_datasets.py --courses config/courses/oval.yaml config/courses/bean.yaml
  python3 scripts/make_course_datasets.py --project project.yaml   # camera / variations from a GUI project
  python3 scripts/make_course_datasets.py --speeds 0.5 --offsets 0 --headings 0   # fewer runs per course

Each course folder has the same layout as a GUI dataset (bev/, camera/, labels.csv, project.yaml, LABELS.txt).
<out>/index.csv lists course, runs and images of every folder.
"""
import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lane_synth.dataset import default_project, generate, make_jobs  # noqa: E402
from lane_synth.generator import load_config  # noqa: E402


def rel(p):
    """Path relative to the synth folder when inside it (keeps index.csv valid if the folder moves)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", nargs="+", default=sorted(glob.glob(str(ROOT / "config/courses/*.yaml"))))
    ap.add_argument("--out", default=str(ROOT / "data/courses"))
    ap.add_argument("--project", default=None, help="GUI project YAML for config and variations (default: defaults)")
    ap.add_argument("--directions", nargs="+", default=None, choices=["forward", "reverse"])
    ap.add_argument("--speeds", nargs="+", type=float, default=None, help="[m/s]")
    ap.add_argument("--offsets", nargs="+", type=float, default=None, help="lateral offsets [m]")
    ap.add_argument("--headings", nargs="+", type=float, default=None, help="heading offsets [deg]")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-camera", action="store_true", help="do not save the camera-view images")
    ap.add_argument("--overwrite", action="store_true", help="regenerate courses whose folder already has labels.csv")
    a = ap.parse_args()

    if a.project:
        base = yaml.safe_load(open(a.project))
    else:
        base = default_project(load_config())
    d = base["dataset"]
    for key, val in (("directions", a.directions), ("speeds", a.speeds), ("offsets_y", a.offsets),
                     ("heading_offsets_deg", a.headings), ("workers", a.workers)):
        if val is not None:
            d[key] = val
    if a.no_camera:
        d["save_camera"] = False
    d["mode"] = "course"

    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    index = []
    t_all = time.time()
    for i, cpath in enumerate(a.courses, 1):
        course = yaml.safe_load(open(cpath))
        name = course.get("name") or Path(cpath).stem
        course["name"] = name
        out = out_root / name
        if (out / "labels.csv").exists() and not a.overwrite:
            with open(out / "labels.csv") as f:
                rows = list(csv.DictReader(f))
            n, runs = len(rows), len({r["run"] for r in rows})
            print(f"[{i}/{len(a.courses)}] {name}: exists ({runs} runs, {n} images), skipped (--overwrite to redo)")
            index.append((name, rel(cpath), runs, n, rel(out)))
            continue
        project = dict(base, course=course, dataset=dict(d, out_dir=str(out)))
        runs = len(make_jobs(project))
        t = time.time()

        def prog(done, total, frames):
            print(f"\r[{i}/{len(a.courses)}] {name}: {done}/{total} runs, {frames} images, {time.time() - t:.0f} s",
                  end="", flush=True)

        n = generate(project, prog)
        print()
        index.append((name, rel(cpath), runs, n, rel(out)))

    with open(out_root / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["course", "course_yaml", "runs", "images", "folder"])
        w.writerows(index)
    print(f"\n{len(index)} courses, {sum(r[3] for r in index)} images, {time.time() - t_all:.0f} s → {out_root}")


if __name__ == "__main__":
    main()
