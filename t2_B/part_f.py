import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 100  # Number of bosons
epsilon = 1.0  # Energy unit
kB = 1.0  # Boltzmann constant (scaled for simplicity)
T_range = np.linspace(0.1, 10, 100)  # Temperature range
beta_range = 1 / (kB * T_range)  # Compute beta

# Compute average particle numbers in quantum case
n_e_Q = 1 / (np.exp(beta_range * epsilon) - 1)
n_0_Q = N - n_e_Q

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_range, n_0_Q, label="Ground state ⟨n₀⟩", linewidth=2)
plt.plot(T_range, n_e_Q, label="Excited state ⟨nε⟩", linewidth=2, linestyle='dashed')
plt.xlabel("Temperature (T)")
plt.ylabel("Average Particle Number")
plt.title("Average Number of Bosons in Each State (Quantum Case)")
plt.legend()
plt.grid()
plt.savefig("part_f.png")
