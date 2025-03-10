import numpy as np
import matplotlib.pyplot as plt

# Define parameters for grand canonical ensemble
mu_values = np.linspace(-2, 2, 100)  # Range of chemical potential values
beta_fixed = 1.0  # Fix beta for visualization

# Compute average particle number in the grand canonical ensemble
n_avg_ground = 1 / (np.exp(beta_fixed * mu_values) - 1)
n_avg_excited = 1 / (np.exp(beta_fixed * (epsilon - mu_values)) - 1)
n_avg_total = n_avg_ground + n_avg_excited

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(mu_values, n_avg_total, label="Total ⟨n⟩", linewidth=2)
plt.plot(mu_values, n_avg_ground, label="Ground state ⟨n₀⟩", linestyle='dashed')
plt.plot(mu_values, n_avg_excited, label="Excited state ⟨nε⟩", linestyle='dotted')
plt.axvline(x=epsilon, color='r', linestyle='dashed', label="mu = epsilon (critical point)")
plt.xlabel("Chemical Potential (μ)")
plt.ylabel("Average Particle Number")
plt.title("Average Particle Number vs. Chemical Potential")
plt.legend()
plt.grid()
plt.savefig("part_g.png")
