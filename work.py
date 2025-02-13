import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit

### Task 1: Density of States

#### Part a) Density of States for 2D Harmonic Oscillator

# Analytical derivation of density of states g(E)
def density_of_states_analytical(E, m, omega):
    # For a 2D harmonic oscillator, g(E) = E / (hbar * omega)^2
    # Since hbar = 1, g(E) = E / omega^2
    return E / (omega**2)

# Numerical calculation of density of states
def density_of_states_numerical(E, m, omega, num_points=1000):
    # Mesh the phase space volume and calculate the variation in energy
    px = np.linspace(-np.sqrt(2 * m * E), np.sqrt(2 * m * E), num_points)
    py = np.linspace(-np.sqrt(2 * m * E), np.sqrt(2 * m * E), num_points)
    x = np.linspace(-np.sqrt(2 * E / (m * omega**2)), np.sqrt(2 * E / (m * omega**2)), num_points)
    y = np.linspace(-np.sqrt(2 * E / (m * omega**2)), np.sqrt(2 * E / (m * omega**2)), num_points)
    
    # Calculate the phase space volume
    phase_space_volume = np.sum(px**2 + py**2 <= 2 * m * E) * np.sum(x**2 + y**2 <= 2 * E / (m * omega**2))
    
    # Calculate the density of states
    g_E = phase_space_volume / (E * num_points**4)
    return g_E

# Example usage
m = 1.0
omega = 1.0
E = 10.0
g_analytical = density_of_states_analytical(E, m, omega)
g_numerical = density_of_states_numerical(E, m, omega)
print(f"Analytical g(E): {g_analytical}, Numerical g(E): {g_numerical}")

#### Part b) Partition Function via Density of States

def partition_function(beta, m, omega, E_max=1000, num_points=1000):
    E_values = np.linspace(0, E_max, num_points)
    g_E_values = density_of_states_analytical(E_values, m, omega)
    Z = np.trapz(g_E_values * np.exp(-beta * E_values), E_values)
    return Z

# Example usage
beta = 1.0
Z = partition_function(beta, m, omega)
print(f"Partition function Z(beta): {Z}")

#### Part c) Density of States for 2D Non-linear Harmonic Oscillator

def density_of_states_nonlinear(E, m, omega, lambda_, num_points=1000):
    # Numerical calculation for non-linear potential
    px = np.linspace(-np.sqrt(2 * m * E), np.sqrt(2 * m * E), num_points)
    py = np.linspace(-np.sqrt(2 * m * E), np.sqrt(2 * m * E), num_points)
    x = np.linspace(-np.sqrt(2 * E / (m * omega**2)), np.sqrt(2 * E / (m * omega**2)), num_points)
    y = np.linspace(-np.sqrt(2 * E / (m * omega**2)), np.sqrt(2 * E / (m * omega**2)), num_points)
    
    # Calculate the phase space volume for non-linear potential
    phase_space_volume = np.sum(px**2 + py**2 <= 2 * m * E) * np.sum(x**2 + y**2 + lambda_ * (x**2 + y**2)**2 <= 2 * E / (m * omega**2))
    
    # Calculate the density of states
    g_E = phase_space_volume / (E * num_points**4)
    return g_E

# Example usage
lambda_ = 0.1
g_nonlinear = density_of_states_nonlinear(E, m, omega, lambda_)
print(f"Density of states for non-linear potential: {g_nonlinear}")

### Task 2: Double Pendulum Dynamics

#### Part a) Lagrangian and Equations of Motion

def double_pendulum_equations(t, state, m1, m2, L1, L2, g):
    theta1, p1, theta2, p2 = state
    
    # Equations of motion for the double pendulum
    delta_theta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta_theta)**2
    den2 = (m1 + m2) * L2 - m2 * L2 * np.cos(delta_theta)**2
    
    dtheta1_dt = p1 / (m1 * L1**2)
    dtheta2_dt = p2 / (m2 * L2**2)
    
    dp1_dt = -m2 * L1 * L2 * dtheta2_dt**2 * np.sin(delta_theta) - (m1 + m2) * g * L1 * np.sin(theta1)
    dp2_dt = m2 * L1 * L2 * dtheta1_dt**2 * np.sin(delta_theta) - m2 * g * L2 * np.sin(theta2)
    
    return [dtheta1_dt, dp1_dt, dtheta2_dt, dp2_dt]

# Example usage
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
g = 9.81
initial_state = [np.pi / 2, 0, np.pi / 2, 0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

sol = solve_ivp(double_pendulum_equations, t_span, initial_state, args=(m1, m2, L1, L2, g), t_eval=t_eval)

#### Part b) Hamiltonian

def hamiltonian(state, m1, m2, L1, L2, g):
    theta1, p1, theta2, p2 = state
    T = (p1**2 / (2 * m1 * L1**2)) + (p2**2 / (2 * m2 * L2**2))
    V = -m1 * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return T + V

# Example usage
H = hamiltonian(initial_state, m1, m2, L1, L2, g)
print(f"Hamiltonian: {H}")

#### Part c) Phase Space Trajectory and Real Space Dynamics

# Plot phase space trajectory (theta2, p2)
plt.plot(sol.y[2], sol.y[3])
plt.xlabel("theta2")
plt.ylabel("p2")
plt.title("Phase Space Trajectory")
plt.show()

# Plot real space dynamics (Cartesian coordinates)
x1 = L1 * np.sin(sol.y[0])
y1 = -L1 * np.cos(sol.y[0])
x2 = x1 + L2 * np.sin(sol.y[2])
y2 = y1 - L2 * np.cos(sol.y[2])

plt.plot(x1, y1, label="Mass 1")
plt.plot(x2, y2, label="Mass 2")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Double Pendulum Dynamics")
plt.legend()
plt.show()

#### Part d) Phase Space Density

# Initialize a point cloud with similar initial conditions
num_points = 100
initial_conditions = np.array([initial_state + np.random.normal(0, 0.1, 4) for _ in range(num_points)])

# Integrate the equations of motion for each initial condition
solutions = [solve_ivp(double_pendulum_equations, t_span, ic, args=(m1, m2, L1, L2, g), t_eval=t_eval) for ic in initial_conditions]

# Extract theta2 and p2 for each solution
theta2_p2 = np.array([sol.y[2:4, -1] for sol in solutions])

# Compute the convex hull of the point cloud
hull = ConvexHull(theta2_p2)

# Plot the convex hull
plt.plot(theta2_p2[:, 0], theta2_p2[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(theta2_p2[simplex, 0], theta2_p2[simplex, 1], 'k-')
plt.xlabel("theta2")
plt.ylabel("p2")
plt.title("Phase Space Density")
plt.show()
