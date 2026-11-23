#!/usr/bin/env python3
"""Generate a dataset from a project file saved by the GUI (no display needed).

  python3 scripts/make_dataset.py project.yaml
  python3 scripts/make_dataset.py project.yaml --out data/run2 --workers 12
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lane_synth.dataset import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    with open(a.project) as f:
        project = yaml.safe_load(f)
    if a.out:
        project["dataset"]["out_dir"] = a.out
    if a.workers:
        project["dataset"]["workers"] = a.workers
    t = time.time()

    def prog(done, total, frames):
        print(f"\r{done}/{total} jobs, {frames} images, {time.time() - t:.0f} s", end="", flush=True)

    n = generate(project, prog)
    print(f"\n{n} images → {project['dataset']['out_dir']}")


if __name__ == "__main__":
    main()
