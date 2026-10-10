"""Download the OGBench datasets used in the paper.

Usage:
    python scripts/download_datasets.py                  # all ten domains
    python scripts/download_datasets.py cube-double scene

The standard (1M) datasets are fetched with ogbench.download_datasets. The
large variants (antmaze-giant 10M, cube-triple 10M, puzzle-4x4 10M,
cube-quadruple 100M) are the OGBench "-10m-v0" / "-100m-v0" releases; they are
downloaded as directories of .npz shards that main.py cycles through via
--ogbench_dataset_dir. All data goes to ~/.ogbench/data (or $OGBENCH_DATA).
"""
import os
import sys

import ogbench
from ogbench.utils import download_datasets

DATASETS = {
    "antmaze-large":       ["antmaze-large-navigate-singletask-v0"],
    "antmaze-giant":       ["antmaze-giant-navigate-singletask-v0", "antmaze-giant-navigate-10m-v0"],
    "humanoidmaze-medium": ["humanoidmaze-medium-navigate-singletask-v0"],
    "humanoidmaze-large":  ["humanoidmaze-large-navigate-singletask-v0"],
    "cube-double":         ["cube-double-play-singletask-v0"],
    "cube-triple":         ["cube-triple-play-singletask-v0", "cube-triple-play-10m-v0"],
    "cube-quadruple":      ["cube-quadruple-play-singletask-v0", "cube-quadruple-play-100m-v0"],
    "scene":               ["scene-play-singletask-v0"],
    "puzzle-3x3":          ["puzzle-3x3-play-singletask-v0"],
    "puzzle-4x4":          ["puzzle-4x4-play-singletask-v0", "puzzle-4x4-play-10m-v0"],
}

if __name__ == "__main__":
    domains = sys.argv[1:] or list(DATASETS)
    data_dir = os.environ.get("OGBENCH_DATA", os.path.expanduser("~/.ogbench/data"))
    names = [n for d in domains for n in DATASETS[d]]
    print(f"downloading {len(names)} dataset(s) to {data_dir}")
    download_datasets(names, dataset_dir=data_dir)
    print("done")
