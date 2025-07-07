from ising3d import Ising3D
import numpy as np, contextlib
import matplotlib.pyplot as plt

np.random.seed(0)                      # reproducibility
n_sweeps = 100
accepted = 0
mdl = Ising3D(L=25, J=1.0, h=-0.2, T=5)
E0 = mdl.total_energy()
mdl.sweep()
E1 = mdl.total_energy()
print("Energy before sweep:", E0)
print("Energy after  sweep:", E1)
print("ΔE =", E1 - E0)

for _ in range(n_sweeps * mdl.L**3):
    i, j, k = np.random.randint(0, mdl.L, 3)
    dE = mdl._dE_single_flip(i, j, k)
    if dE <= 0 or np.random.rand() < np.exp(-mdl.beta * dE):
        accepted += 1
        mdl.s[i, j, k] *= -1
print(f"Acceptance rate ≈ {accepted / (n_sweeps * mdl.L**3):.2f}")

ms = []
for _ in range(200):
    mdl.sweep()
    ms.append(mdl.magnetisation())
plt.plot(ms)
plt.xlabel("sweep"); plt.ylabel("m")
plt.title("Magnetisation vs. MC sweeps")
plt.show()