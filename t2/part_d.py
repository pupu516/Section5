import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
import matplotlib.animation as animation

# System Parameters
m1, m2 = 1.0, 1.0  # Masses
L1, L2 = 1.0, 1.0  # Rod Lengths
g = 9.81           # Gravity

# Equations of Motion
def equations(t, y):
    theta1, theta2, p1, p2 = y
    
    # Mass matrix
    M = np.array([
        [(m1 + m2) * L1**2, m2 * L1 * L2 * np.cos(theta1 - theta2)],
        [m2 * L1 * L2 * np.cos(theta1 - theta2), m2 * L2**2]
    ])
    
    # Solve for angular velocities
    theta_dot = np.linalg.solve(M, np.array([p1, p2]))
    
    # Compute time derivatives of p1, p2
    p1_dot = - (m1 + m2) * g * L1 * np.sin(theta1) - m2 * L1 * L2 * theta_dot[1]**2 * np.sin(theta1 - theta2)
    p2_dot = - m2 * g * L2 * np.sin(theta2) + m2 * L1 * L2 * theta_dot[0]**2 * np.sin(theta1 - theta2)
    
    return [theta_dot[0], theta_dot[1], p1_dot, p2_dot]

# Initialize a Phase Space Cloud
N = 100  # Number of trajectories
perturbation = 1e-3  # Small perturbation
initial_conditions = np.array([np.pi / 2, np.pi / 2, 0, 0])
trajectories = [initial_conditions + perturbation * np.random.randn(4) for _ in range(N)]

# Time Interval
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve for all trajectories
solutions = [solve_ivp(equations, t_span, y0, t_eval=t_eval, method='Radau') for y0 in trajectories]

# Extract Phase Space Data
phase_space_points = np.array([[sol.y[1], sol.y[3]] for sol in solutions])  # (theta2, p2)

# Compute Convex Hull Areas Over Time
convex_hull_areas = []
for i in range(len(t_eval)):
    points = np.array([phase_space_points[j][:, i] for j in range(N)])
    hull = ConvexHull(points)
    convex_hull_areas.append(hull.volume)

# Plot Convex Hull Area Growth
plt.figure(figsize=(8, 6))
plt.plot(t_eval, convex_hull_areas, label='Convex Hull Area')
plt.xlabel('Time')
plt.ylabel('Phase Space Volume')
plt.title('Phase Space Expansion')
plt.legend()
plt.grid()
plt.savefig("part_d.png")

# Animate Phase Space Cloud Evolution
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim([-100, 100])
points_plot, = ax.plot([], [], 'bo', alpha=0.5)

# Animation Function
def animate(i):
    points = np.array([phase_space_points[j][:, i] for j in range(N)])
    points_plot.set_data(points[:, 0], points[:, 1])  # Show all points at each frame
    return points_plot,

ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=50, blit=True)
plt.show()

