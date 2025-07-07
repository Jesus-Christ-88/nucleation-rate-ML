"""
run_bruteforce.py  —  generate a single brute-force trajectory
--------------------------------------------------------------

Examples
--------
# quick smoke test (no snapshots, no plot)
python run_bruteforce.py --L 16 --max_sweeps 5000

# save snapshots every 200 sweeps and plot the magnetisation
python run_bruteforce.py --L 16 --h -0.08 --T 2.5 \
                         --max_sweeps 20000         \
                         --snap_every 200          \
                         --out traj_L16.h5         \
                         --plot
"""

import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from ising3d import Ising3D

# -------------------------- CLI arguments ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--L",          type=int,   default=16,      help="lattice size")
parser.add_argument("--J",          type=float, default=1.0,     help="coupling J")
parser.add_argument("--h",          type=float, default=-0.12,   help="field h (<0)")
parser.add_argument("--T",          type=float, default=3.0,     help="temperature T")
parser.add_argument("--max_sweeps", type=int,   default=500000,  help="hard cutoff")
parser.add_argument("--snap_every", type=int,   default=200,
                    help="save snapshot every N sweeps (0 = no snapshots)")
parser.add_argument("--out",        type=str,   default="traj.h5",
                    help="output HDF5 file")
parser.add_argument("--plot",       action="store_true",
                    help="plot magnetisation trace at the end")
args = parser.parse_args()

# -------------------------- initialise model ------------------------
ising = Ising3D(L=args.L, J=args.J, h=args.h, T=args.T)

m_trace   = [ising.magnetisation()]
snapshots = []                       # fill only if snap_every > 0
t0        = time.time()
flip_sweep = None

# -------------------------- main MC loop ----------------------------
for sweep in range(1, args.max_sweeps + 1):
    ising.sweep()
    m = ising.magnetisation()
    m_trace.append(m)

    # store snapshot?
    if args.snap_every and sweep % args.snap_every == 0:
        snapshots.append(ising.s.copy())


run_time = time.time() - t0
print(f"Finished {sweep} sweeps in {run_time:.1f} s "
      f"({'flipped' if flip_sweep else 'no flip'})")

# -------------------------- write HDF5 ------------------------------
out_path = Path(args.out)
with h5py.File(out_path, "w") as h5:
    h5.attrs.update(
        L=args.L, J=args.J, h=args.h, T=args.T,
        flipped=bool(flip_sweep), flip_sweep=flip_sweep or -1,
        snap_every=args.snap_every
    )
    h5.create_dataset("magnetisation", data=np.asarray(m_trace))
    if snapshots:                                       # save only if list non-empty
        h5.create_dataset("snapshots",
                          data=np.asarray(snapshots, dtype=np.int8),
                          compression="gzip")

print(f"Trace written to {out_path.resolve()}")

# ------------------------ plot ---------------------------
if args.plot:
    import matplotlib.pyplot as plt
    plt.plot(m_trace)
    plt.xlabel("sweep")
    plt.ylabel("m")
    plt.title(out_path.name)
    plt.show()
