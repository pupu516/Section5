import numpy as np
import matplotlib.pyplot as plt

# Define parameters for a Bose system with BEC
T_c = 2.0  # Critical temperature for BEC
T_range_BEC = np.linspace(0.1, 3 * T_c, 100)  # Temperature range around Tc

# Compute ground state occupation with BEC behavior
n_0_BEC = np.where(T_range_BEC < T_c, (1 - (T_range_BEC / T_c) ** (3 / 2)), 0)
Cv_BEC = np.gradient(n_0_BEC, T_range_BEC)  # Specific heat approximation

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_range_BEC, n_0_BEC, label="Ground state occupation ⟨n₀⟩ (BEC)", linewidth=2)
plt.plot(T_range_BEC, Cv_BEC, label="Specific heat Cv", linestyle='dashed')
plt.axvline(x=T_c, color='r', linestyle='dashed', label="Critical temperature Tc")
plt.xlabel("Temperature (T)")
plt.ylabel("Computed Values")
plt.title("Bose System With BEC - Phase Transition")
plt.legend()
plt.grid()
plt.savefig("part_j.png")
