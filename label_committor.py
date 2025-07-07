#!/usr/bin/env python
"""
label_committor.py
------------------
Estimate the committor p_B(x) for a batch of lattice snapshots.

Each snapshot is fired `--n_fire` times; a firing runs up to
`--max_steps` Monte-Carlo sweeps and stops as soon as the configuration
reaches basin A (up) or basin B (down).  The committor is the fraction
of firings that hit basin B first.

Outputs an HDF5 file with three datasets:
  snapshot : int8  array (N, L, L, L)
  pB       : float array (N,)         — estimated committor
  stderr   : float array (N,)         — binomial standard error
"""

from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
import h5py

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # fallback if tqdm is not installed
    def tqdm(x, **kw): return x

from ising3d import Ising3D

# ----------------------------------------------------------------------
#  command-line arguments
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",    required=True,
                    help="HDF5 file that contains lattice snapshots")
parser.add_argument("--dataset",  default="snapshots",
                    help="dataset name inside INPUT that stores "
                         "(N,L,L,L) spin arrays")
parser.add_argument("--n_fire",   type=int, default=100,
                    help="# independent trial runs per snapshot")
parser.add_argument("--max_steps", type=int, default=500,
                    help="max MC sweeps per trial")
parser.add_argument("--h",        type=float, default=-0.08,
                    help="field h used during trial runs")
parser.add_argument("--T",        type=float, default=3.5,
                    help="temperature T used during trial runs")
parser.add_argument("--save",     default="labels.h5",
                    help="output HDF5 filename")
args = parser.parse_args()

# ----------------------------------------------------------------------
#  basin indicator helpers
# ----------------------------------------------------------------------
def in_A(m: float) -> bool:   # metastable up basin
    return m > +0.80

def in_B(m: float) -> bool:   # stable down basin
    return m < -0.60

# ----------------------------------------------------------------------
#  load snapshots
# ----------------------------------------------------------------------
in_path = Path(args.input)
if not in_path.exists():
    sys.exit(f"[ERROR] file not found: {in_path}")

with h5py.File(in_path, "r") as h5:
    if args.dataset not in h5:
        sys.exit(f"[ERROR] dataset '{args.dataset}' not found in {args.input}")
    snaps = h5[args.dataset][:]
N, L, _, _ = snaps.shape
print(f"Loaded {N} snapshots of lattice size {L}³")

# ----------------------------------------------------------------------
#  main committor loop
# ----------------------------------------------------------------------
labels  = np.empty(N, dtype=np.float32)
stderr  = np.empty(N, dtype=np.float32)

for idx, s0 in enumerate(tqdm(snaps, desc="labelling")):
    wins = 0
    for _ in range(args.n_fire):
        mdl = Ising3D(L=L, h=args.h, T=args.T)
        mdl.s[...] = s0                       # copy spin configuration

        for _ in range(args.max_steps):
            m = mdl.magnetisation()
            if in_A(m):                     # basin A reached → trial lost
                break
            if in_B(m):                     # basin B reached → trial wins
                wins += 1
                break
            mdl.sweep()                     # advance one MC sweep

    pB = wins / args.n_fire
    se = np.sqrt(pB * (1.0 - pB) / args.n_fire)  # binomial stderr
    labels[idx] = pB
    stderr[idx] = se

# ----------------------------------------------------------------------
#  write output
# ----------------------------------------------------------------------
out_path = Path(args.save)
with h5py.File(out_path, "w") as h5:
    h5.create_dataset("snapshot", data=snaps,  compression="gzip")
    h5.create_dataset("pB",       data=labels)
    h5.create_dataset("stderr",   data=stderr)
    h5.attrs.update(
        L=L, n_snapshots=N, n_fire=args.n_fire,
        max_steps=args.max_steps, h=args.h, T=args.T,
        basin_A="m > +0.8", basin_B="m < -0.6"
    )

print(f"✅  Saved {N} labels to {out_path.resolve()}")

with h5py.File("labels_L16.h5") as h5:
    pB = h5["pB"][:]
plt.hist(pB, bins=20); plt.xlabel("p_B"); plt.ylabel("count")
plt.title("Distribution of committor labels"); plt.show()
