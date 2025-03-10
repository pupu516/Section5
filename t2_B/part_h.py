import numpy as np
import matplotlib.pyplot as plt

# Define parameters for near-degenerate system
num_levels = 50  # Number of near-degenerate energy levels
energy_levels = np.linspace(0, 1, num_levels)  # Small separations between levels
mu_fixed = -0.5  # Fixed chemical potential
T_range = np.linspace(0.1, 10, 100)  # Temperature range
beta_range = 1 / (T_range)  # Compute beta assuming kB = 1

# Compute ground state occupation for near-degenerate system
n_0_near = np.array([sum(1 / (np.exp(beta * (energy_levels - mu_fixed)) - 1)) for beta in beta_range])
log_n_0 = np.log(n_0_near)  # Log of ground state occupation
grad_n_0 = np.gradient(n_0_near, T_range)  # Gradient w.r.t. temperature
Cv = np.gradient(grad_n_0, T_range)  # Specific heat approximation

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_range, n_0_near, label="Ground state occupation ⟨n₀⟩", linewidth=2)
plt.plot(T_range, log_n_0, label="log(⟨n₀⟩)", linestyle='dashed')
plt.plot(T_range, -grad_n_0, label="-∂⟨n₀⟩/∂T", linestyle='dotted')
plt.plot(T_range, Cv, label="Specific heat Cv", linestyle='dashdot')
plt.xlabel("Temperature (T)")
plt.ylabel("Computed Values")
plt.title("Near-Degenerate Bose System Behavior")
plt.legend()
plt.grid()
plt.savefig("part_h.png")
