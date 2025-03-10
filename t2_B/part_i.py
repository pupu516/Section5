import numpy as np
import matplotlib.pyplot as plt

# Define parameters for a system without BEC
num_levels_no_BEC = 50  # Many accessible excited states
energy_gaps = np.linspace(1, 10, num_levels_no_BEC)  # Large energy gaps
T_range_no_BEC = np.linspace(0.1, 10, 100)  # Temperature range
beta_range_no_BEC = 1 / (T_range_no_BEC)  # Compute beta assuming kB = 1

# Compute occupation numbers for a system without BEC
n_no_BEC = np.array([sum(1 / (np.exp(beta * energy_gaps) - 1)) for beta in beta_range_no_BEC])

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(T_range_no_BEC, n_no_BEC, label="Total Occupation (No BEC)", linewidth=2)
plt.xlabel("Temperature (T)")
plt.ylabel("Average Particle Number")
plt.title("Bose System Without BEC - Smooth Occupation Distribution")
plt.legend()
plt.grid()
plt.savefig("part_i.png")
