import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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

# Initial Conditions
y0 = [np.pi / 2, np.pi / 2, 0, 0]  # (theta1, theta2, p1, p2)

# Time Interval
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 5000)  # Increase resolution

# Solve ODE with implicit solver for stability
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval, method='Radau')

# Extract Results
t = solution.t
theta1 = (solution.y[0] + np.pi) % (2 * np.pi) - np.pi  # Wrapping angles
theta2 = (solution.y[1] + np.pi) % (2 * np.pi) - np.pi
p2 = solution.y[3]  # Corrected momentum extraction
x1, y1 = L1 * np.sin(theta1), -L1 * np.cos(theta1)
x2, y2 = x1 + L2 * np.sin(theta2), y1 - L2 * np.cos(theta2)

# Plot Phase Space
plt.figure(figsize=(8, 6))
plt.plot(theta2, p2, label='Phase Space (theta2, p2)', alpha=0.7)
plt.xlabel('Theta2 (rad)')
plt.ylabel('Momentum p2')
plt.legend()
plt.title('Phase Space Trajectory')
plt.xlim([-np.pi, np.pi])  # Limit theta2 range to improve visualization
plt.ylim([-100, 100])  # Shrink momentum axis range
plt.grid()
plt.savefig("part_c.png")

# Animate the Double Pendulum in Real Space
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=10, blit=True)
plt.show()
