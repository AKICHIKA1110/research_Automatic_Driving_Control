#!/usr/bin/env python3
"""Launch the dataset GUI.  python3 scripts/gui.py [project.yaml]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":   # required: dataset workers are started with "spawn"
    from lane_synth.gui import main
    main()
