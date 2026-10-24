"""Summarize flow-map runs: final evaluation return per task (mean, std over
seeds) and the interquartile mean (IQM) over all runs of a suite with a 95%
stratified-bootstrap interval (seeds resampled within tasks; DMC returns
divided by 1000, humanoid returns raw).

    python scripts/summarize_flowmap.py                     # outputs/flowmap_k*
    python scripts/summarize_flowmap.py outputs/flowmap_k2

Each folder holds <task>__s<seed>/metrics.npz (scripts/run_flowmap.sh).
"""

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

HUMANOID = ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain",
            "T1JoystickFlatTerrain", "T1JoystickRoughTerrain")
RUN = re.compile(r"^(\w+?)__s(\d+)$")


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float).ravel())
    lo, hi = int(np.floor(0.25 * len(x))), int(np.ceil(0.75 * len(x)))
    return float(x[lo:hi].mean()) if hi > lo else float(np.mean(x))


def strat_boot_iqm(per_task, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    stats = [iqm(np.concatenate([rng.choice(v, len(v)) for v in per_task])) for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


def summarize(folder):
    final = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(folder, "*", "metrics.npz"))):
        m = RUN.match(os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        try:
            ret = np.asarray(np.load(path)["eval/episode_return"]).ravel()
        except (OSError, EOFError, KeyError, ValueError) as ex:
            print(f"skipping unreadable {path}: {ex!r}")
            continue
        final[m.group(1)][int(m.group(2))] = float(ret[-1])
    if not final:
        print(f"{folder}: no runs")
        return
    print(f"\n== {folder}: final evaluation return ==")
    print(f"{'task':<26}{'mean':>9}{'std':>9}{'runs':>6}")
    for task in sorted(final, key=lambda t: (t in HUMANOID, t)):
        v = np.array(list(final[task].values()))
        print(f"{task:<26}{v.mean():>9.1f}{v.std():>9.1f}{len(v):>6}")
    for name, tasks, scale in (("DMC (/1000)", [t for t in final if t not in HUMANOID], 1000.0),
                               ("humanoid", [t for t in final if t in HUMANOID], 1.0)):
        if not tasks:
            continue
        per_task = [np.array(list(final[t].values())) / scale for t in sorted(tasks)]
        lo, hi = strat_boot_iqm(per_task)
        runs = sum(len(v) for v in per_task)
        print(f"IQM {name:<12} {iqm(np.concatenate(per_task)):.3f} [{lo:.3f}, {hi:.3f}]"
              f"  ({len(tasks)} tasks, {runs} runs)")


if __name__ == "__main__":
    folders = sys.argv[1:] or sorted(glob.glob(os.path.join("outputs", "flowmap_k*")))
    for f in folders:
        summarize(f)
