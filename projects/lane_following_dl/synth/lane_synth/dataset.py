"""Dataset generation shared by the GUI and the command line.

A *project* (dict / YAML) holds everything:

  config:  the generator config (camera, bev, labels, track, driving, perturb, detection_noise)
  course:  a course dict (segments...) or null for random courses
  dataset:
    mode: course | random
    out_dir: path
    directions: [forward, reverse]          # course mode
    speeds: [0.5]                           # course mode, one run per combination
    offsets_y: [0.0]                        # default: centerline only
    heading_offsets_deg: [0.0]
    random_samples: 20000                   # random mode
    workers: 8
    save_camera: true                       # also save the camera-view image

Output (out_dir):
  bev/<run>_<frame>.png      BEV line image (input of the curvature CNN)
  camera/<run>_<frame>.png   camera-view line image (if save_camera)
  labels.csv                 one row per image (see LABEL_DOC)
  project.yaml               the project used
"""
from __future__ import annotations

import csv
import itertools
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

LABEL_DOC = """\
labels.csv columns
  file          image file name (same name in bev/ and camera/)
  run, frame    run id and frame number in the run
  course        course name (course mode) or random course id
  speed, offset_y, heading_offset_deg   trajectory settings of the run
  s             arc length of the vehicle along the target path [m]
  kappa_<s>     curvature of the target path at arc length <s> m ahead of the vehicle [1/m]
                (left turn positive) -- the main label
  e_y           lateral offset of the vehicle (rear axle) from the target path [m] (left +)
  e_psi         heading error vehicle - path [rad] (left +)
  px_<s>, py_<s>, pvalid_<s>   centerline point at arc length <s> ahead in the vehicle frame
                (x forward, y left) [m]; pvalid = 1 if it can be inferred from the image
"""


def default_project(cfg: dict) -> dict:
    # Default: drive on the lane centerline only (no lateral / heading offset).
    # The target trajectory fed to the MPC is still under discussion; add offsets explicitly when needed.
    return dict(config=cfg, course=None, dataset=dict(
        mode="course", out_dir="data/dataset", directions=["forward", "reverse"],
        speeds=[0.5], offsets_y=[0.0], heading_offsets_deg=[0.0],
        random_samples=20000, workers=max(1, (os.cpu_count() or 2) - 1), save_camera=True))


def label_header(cfg):
    ks = [f"kappa_{s:g}" for s in cfg["labels"]["preview_s"]]
    ps = cfg["labels"].get("path_s", [])
    pts = [f"{a}_{s:g}" for s in ps for a in ("px", "py", "pvalid")]
    return ["file", "run", "frame", "course", "speed", "offset_y", "heading_offset_deg", "s"] + ks + \
        ["e_y", "e_psi"] + pts


def make_jobs(project: dict):
    """Split the work into jobs (dicts) for worker processes."""
    d = project["dataset"]
    jobs = []
    if d["mode"] == "course":
        if not project.get("course"):
            raise ValueError("course mode needs a course")
        combos = list(itertools.product(d["directions"], d["speeds"], d["offsets_y"], d["heading_offsets_deg"]))
        for run, (dr, v, oy, oh) in enumerate(combos):
            jobs.append(dict(kind="course", run=run, direction=dr, speed=float(v), offset_y=float(oy),
                             heading_offset_deg=float(oh)))
    else:
        n, chunk = int(d["random_samples"]), 500
        for i, start in enumerate(range(0, n, chunk)):
            jobs.append(dict(kind="random", run=i, n=min(chunk, n - start), seed=i))
    return jobs


def run_job(args):
    """Worker: render one job, write images and a partial CSV. Returns #frames."""
    project, job, part_dir = args
    # import here so that spawned workers import the package fresh
    from .generator import Generator
    from .track import course_from_dict

    cfg = project["config"]
    d = project["dataset"]
    out = Path(d["out_dir"])
    g = Generator(cfg, seed=1000 + job["run"] if job["kind"] == "course" else job["seed"])
    rows = []
    if job["kind"] == "course":
        course = course_from_dict(project["course"], cfg["track"])
        name = project["course"].get("name", "course")
        frames = g.run(course, 0, job["direction"] == "forward",
                       dict(speed=job["speed"], offset_y=job["offset_y"],
                            heading_offset_deg=job["heading_offset_deg"]))
        meta = (name, job["speed"], job["offset_y"], job["heading_offset_deg"])
    else:
        it = g.sequences()
        frames = (next(it) for _ in range(job["n"]))
        meta = None
    for i, s in enumerate(frames):
        fname = f"{job['run']:05d}_{i:05d}.png"
        cv2.imwrite(str(out / "bev" / fname), s["bev"])
        if d.get("save_camera", True):
            cv2.imwrite(str(out / "camera" / fname), s["cam"])
        m = meta or (f"random{job['run']:04d}_{s['seq']}", s["v"], 0.0, 0.0)
        pts = []
        for (px, py), pv in zip(s["path_xy"], s["path_valid"]):
            pts += [f"{px:.5f}", f"{py:.5f}", int(pv)]
        rows.append([fname, job["run"], i, m[0], f"{m[1]:.3f}", f"{m[2]:.4f}", f"{m[3]:.2f}", f"{s['s']:.4f}"]
                    + [f"{k:.5f}" for k in s["kappa"]] + [f"{s['e_y']:.5f}", f"{s['e_psi']:.6f}"] + pts)
    with open(Path(part_dir) / f"part_{job['run']:05d}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows)


def generate(project: dict, progress=None, should_stop=None) -> int:
    """Generate the dataset. progress(done_jobs, total_jobs, frames); should_stop() → bool."""
    d = project["dataset"]
    out = Path(d["out_dir"])
    (out / "bev").mkdir(parents=True, exist_ok=True)
    if d.get("save_camera", True):
        (out / "camera").mkdir(parents=True, exist_ok=True)
    part_dir = out / ".parts"
    part_dir.mkdir(exist_ok=True)
    with open(out / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, allow_unicode=True, sort_keys=False)
    (out / "LABELS.txt").write_text(LABEL_DOC)
    jobs = make_jobs(project)
    total, done, frames = len(jobs), 0, 0
    workers = max(1, min(int(d.get("workers", 1)), total))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for n in pool.imap_unordered(run_job, [(project, j, str(part_dir)) for j in jobs]):
            done += 1
            frames += n
            if progress:
                progress(done, total, frames)
            if should_stop and should_stop():
                pool.terminate()
                return frames
    # merge the partial CSVs in run order
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(label_header(project["config"]))
        for j in sorted(part_dir.glob("part_*.csv")):
            with open(j) as pf:
                f.write(pf.read())
            j.unlink()
    part_dir.rmdir()
    return frames
