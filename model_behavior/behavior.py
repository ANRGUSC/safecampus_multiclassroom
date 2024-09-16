import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint
# Parameters for the simulation
max_students = 100  # Maximum number of students per classroom
num_classrooms = 5  # Number of classrooms
time_steps = 30  # Simulation duration (in time steps)

# Transmission rates (can be adjusted)
phi = 0.005  # Within-classroom transmission rate
beta = 0.01  # Community transmission rate

# Community risk values for each classroom over time (randomized for simulation)
np.random.seed(42)  # For reproducibility
community_risk = np.random.uniform(0.5, 1.0, (num_classrooms, time_steps))

# Number of students allowed in each classroom (assumed to be max_students for simplicity)
allowed_students = np.full((num_classrooms, time_steps), max_students)

# Number of students attending both classrooms (k_ij matrix, random values for cross-classroom students)
cross_classroom_students = np.random.randint(0, 20, size=(num_classrooms, num_classrooms))

# Initialize infection dynamics for each classroom (initially 1 infected individual in each classroom)
infected = np.zeros((num_classrooms, time_steps))
infected[:, 0] = 1  # Start with 1 infected student per classroom


# Function to calculate R0 for each classroom
def calculate_R0(classroom_idx, t):
    u_i = allowed_students[classroom_idx, t]
    c_i = community_risk[classroom_idx, t]

    # Within-classroom term
    within_classroom = phi + beta * c_i * u_i

    # Cross-classroom term
    cross_classroom = 0
    for j in range(num_classrooms):
        if j != classroom_idx:
            u_j = allowed_students[j, t]
            c_j = community_risk[j, t]
            k_ij = cross_classroom_students[classroom_idx, j]
            cross_classroom += k_ij * (phi + beta * c_j * u_j)

    # Calculate R0
    R0 = within_classroom * (u_i - np.sum(cross_classroom_students[classroom_idx])) + cross_classroom
    return R0


def plot_threshold_behavior_R0():
    allowed_values = np.arange(0, max_students + 10, 10)
    R0_values = []

    for allowed in allowed_values:
        R0_sum = 0
        for i in range(num_classrooms):
            allowed_students[i, 0] = allowed
            R0_sum += calculate_R0(i, 0)
        R0_values.append(R0_sum / num_classrooms)

    R0_values = np.array(R0_values)

    plt.figure(figsize=(10, 6))
    plt.plot(allowed_values, R0_values, label=r'$R_0$', color='black', linewidth=2)
    plt.axhline(y=1, color='red', linestyle='--', label=r'$R_0 = 1$')
    plt.fill_between(allowed_values, 0, R0_values, where=(R0_values < 1), color='blue', alpha=0.3, label='DFE region')
    plt.fill_between(allowed_values, 0, R0_values, where=(R0_values >= 1), color='red', alpha=0.3, label='EE region')
    plt.xlabel(r'Allowed Population $N_i$')
    plt.ylabel(r'$R_0$')
    plt.title(r'Threshold Behavior of $R_0$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("R0_threshold_behavior.png")
    plt.show()


# Function to simulate infection dynamics and calculate R0 over time
def simulate_infection_dynamics():
    R0_values = np.zeros((num_classrooms, time_steps))

    for t in range(1, time_steps):
        for i in range(num_classrooms):
            R0_i = calculate_R0(i, t)
            R0_values[i, t] = R0_i

            u_i = allowed_students[i, t]
            i_i = infected[i, t - 1]
            c_i = community_risk[i, t]

            alpha_i = phi * i_i / u_i + beta * c_i - i_i / u_i

            cross_infection = 0
            for j in range(num_classrooms):
                if j != i:
                    k_ij = cross_classroom_students[i, j]
                    u_j = allowed_students[j, t]
                    i_j = infected[j, t - 1]
                    c_j = community_risk[j, t]
                    alpha_j = phi * i_j / u_j + beta * c_j - i_j / u_j
                    cross_infection += k_ij * (alpha_i + alpha_j)

            delta_i = (u_i - np.sum(cross_classroom_students[i]) - i_i) * alpha_i + cross_infection
            infected[i, t] = max(0, min(u_i, i_i + delta_i))

    return R0_values



# Plot the threshold behavior showing regions of EE and DFE
def plot_threshold_behavior(R0_values):
    plt.figure(figsize=(10, 6))
    for i in range(num_classrooms):
        plt.plot(R0_values[i], label=f"Classroom {i + 1}")

    plt.axhline(y=1, color='r', linestyle='--', label='$R_0 = 1$ (Threshold)')
    plt.fill_between(np.arange(time_steps), 0, 1, color='green', alpha=0.1, label="DFE Region (R_0 < 1)")
    plt.fill_between(np.arange(time_steps), 1, np.max(R0_values), color='orange', alpha=0.1,
                     label="EE Region (R_0 > 1)")

    plt.xlabel("Time Steps")
    plt.ylabel("$R_0$")
    plt.title("Threshold Behavior of the Model (R_0 over time)")
    plt.legend()
    plt.grid(True)
    plt.savefig("threshold_behavior.png")
    plt.show()


# Plot the infection dynamics
def plot_infection_dynamics():
    plt.figure(figsize=(10, 6))
    for i in range(num_classrooms):
        plt.plot(infected[i], label=f"Classroom {i + 1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Infected Students")
    plt.title("Infection Dynamics in Multiple Classrooms")
    plt.legend()
    plt.grid(True)
    plt.savefig("infection_dynamics.png")
    plt.show()


def plot_sensitivity_phi_beta():
    phi_values = np.linspace(0.0001, 0.01, 100)
    beta_values = np.linspace(0.00001, 0.01, 100)

    R0_values_phi = []
    R0_values_beta = []

    # Sensitivity to phi
    for phi_val in phi_values:
        global phi
        phi = phi_val
        R0 = calculate_R0(0, 0)
        R0_values_phi.append(R0)

    phi = 0.001  # Reset to original value

    # Sensitivity to beta
    for beta_val in beta_values:
        global beta
        beta = beta_val
        R0 = calculate_R0(0, 0)
        R0_values_beta.append(R0)

    beta = 0.0001  # Reset to original value

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(phi_values, R0_values_phi, label=r'$R_0$ vs $\phi$', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='$R_0 = 1$')
    plt.xlabel(r'$\phi$ (Within-classroom transmission rate)')
    plt.ylabel(r'$R_0$')
    plt.title('Sensitivity of $R_0$ to $\\phi$')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(beta_values, R0_values_beta, label=r'$R_0$ vs $\beta$', color='green')
    plt.axhline(y=1, color='r', linestyle='--', label='$R_0 = 1$')
    plt.xlabel(r'$\beta$ (Community transmission rate)')
    plt.ylabel(r'$R_0$')
    plt.title('Sensitivity of $R_0$ to $\\beta$')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png")
    plt.show()



# Calculate and plot the stationary distribution of the transition matrix
def plot_stationary_distribution(P, states):
    eigenvalues, eigenvectors = linalg.eig(P.T)
    stationary_index = np.argmin(np.abs(eigenvalues - 1))
    stationary = eigenvectors[:, stationary_index].real
    stationary /= stationary.sum()  # Normalize

    x_values = []
    y_values = []

    for i, prob in enumerate(stationary):
        x_values.append(states[i][0])  # community risk (example)
        y_values.append(states[i][1])  # infected individuals

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, c=stationary, cmap='viridis', edgecolor='black')
    plt.colorbar(label='Stationary Probability')
    plt.xlabel('Community Risk')
    plt.ylabel('Infected Individuals')
    plt.title('Stationary Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stationary_distribution.png")
    plt.show()

# Plot the transition matrix
def plot_transition_matrix(P):
    plt.figure(figsize=(10, 6))
    plt.imshow(P, cmap='gray', aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Transition Probability')
    plt.xlabel('Next State Index')
    plt.ylabel('Current State Index')
    plt.title('Transition Probability Matrix')
    plt.tight_layout()
    plt.savefig("transition_matrix.png")
    plt.show()
# Plot Lyapunov function behavior
def plot_lyapunov_function(P, states):
    V = np.array([0.5 * (i ** 2 + a ** 2) for (a, i) in states])
    expected_V_next = P @ V
    delta_V = expected_V_next - V

    dfe_indices = [i for i, (_, inf) in enumerate(states) if inf == 0]
    ee_indices = [i for i, (_, inf) in enumerate(states) if inf > 0]

    plt.figure(figsize=(10, 6))

    plt.scatter(V[dfe_indices], delta_V[dfe_indices], color='blue', s=5, label='DFE region (Infected = 0)')
    plt.scatter(V[ee_indices], delta_V[ee_indices], color='red', s=5, label='EE region (Infected > 0)')
    plt.axhline(y=0, color='black', linestyle='--', label=r'$\Delta V = 0$')

    plt.xlabel('Lyapunov Function Value $V$')
    plt.ylabel(r'$\Delta V$')
    plt.title('Lyapunov Function Behavior in DFE and EE Regions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lyapunov_function.png")
    plt.show()

# Simulate infection dynamics and calculate transition matrix P
def simulate_transition_matrix():
    states = [(100, i) for i in range(max_students + 1)]  # Possible states: (allowed, infected)
    num_states = len(states)
    P = np.zeros((num_states, num_states))  # Initialize the transition matrix

    for t in range(time_steps):
        community_risks = np.random.uniform(0, 1, num_states)  # Random community risk for each state

        for idx, (a, i) in enumerate(states):
            # Calculate new infections based on generalized R0
            R0 = calculate_R0(0, t)  # Example for one classroom
            delta_i = min(int(R0 * i), max_students)

            # Find the next state based on the number of new infections
            next_state = (a, delta_i)
            next_idx = states.index(next_state)

            # Update the transition matrix P
            P[idx, next_idx] += 1

    # Normalize the transition matrix
    P = P / P.sum(axis=1, keepdims=True)
    return P, states


def plot_threshold_behavior_alpha_beta():
    # Adjusted range of alpha (phi) and beta values
    phi_values = np.linspace(0.001, 2, 100)
    beta_values = np.linspace(0.001, 0.1, 100)

    # Create a meshgrid for alpha and beta
    phi_grid, beta_grid = np.meshgrid(phi_values, beta_values)

    # Initialize R0_grid
    R0_grid = np.zeros_like(phi_grid)

    # Use mean community risk for simplification
    c = np.mean(community_risk)

    # Calculate R0 for each combination of phi and beta
    for i in range(phi_grid.shape[0]):
        for j in range(phi_grid.shape[1]):
            phi = phi_grid[i, j]
            beta = beta_grid[i, j]

            # Within-classroom term
            within_classroom = phi + beta * c * max_students

            # Cross-classroom term (simplified for this plot)
            cross_classroom = np.sum(cross_classroom_students) * (phi + beta * c * max_students)

            # Calculate R0
            R0_grid[i, j] = within_classroom * (max_students - np.sum(cross_classroom_students)) + cross_classroom

    # Create a contour plot to visualize the relationship between phi, beta, and R0
    plt.figure(figsize=(10, 8))

    levels = np.linspace(0, np.max(R0_grid), 20)
    im = plt.contourf(phi_grid, beta_grid, R0_grid, cmap='viridis', levels=levels)
    plt.colorbar(im, label='$R_0$ Value')

    # Add contour line for R0 = 1
    R0_contour = plt.contour(phi_grid, beta_grid, R0_grid, levels=[1], colors='black', linestyles='--')
    plt.clabel(R0_contour, inline=1, fontsize=10, fmt={1: '$R_0 = 1$'})

    # Axis labels and plot settings
    plt.xlabel(r'$\phi$ (Within-classroom transmission rate)')
    plt.ylabel(r'$\beta$ (Community transmission rate)')
    plt.title(r'Threshold Behavior of $R_0$ for Different $\phi$ and $\beta$')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("threshold_behavior_alpha_beta.png", dpi=300)
    plt.show()


def plot_phase_diagram_stochastic(phi, beta, max_students, community_risk, num_trajectories=5, num_steps=50):
    """
    Plot a stochastic phase diagram showing infected vs allowed students with random community risk over time.

    Arguments:
    phi -- within-classroom transmission rate
    beta -- community transmission rate
    max_students -- maximum number of students per classroom
    community_risk -- community risk values (used to simulate randomness)
    num_trajectories -- number of trajectories to simulate (default: 5)
    num_steps -- number of steps for each trajectory (default: 50)
    """

    # Initialize the phase space grid
    u_range = np.linspace(0, max_students, 100)
    i_range = np.linspace(0, max_students, 100)
    u_grid, i_grid = np.meshgrid(u_range, i_range)

    def stochastic_step(u, i, phi, beta, c):
        """
        Perform one stochastic step based on current allowed students (u), infected students (i),
        transmission rates (phi, beta), and community risk (c).
        """
        if u > 0:
            # Introduce randomness in community risk for each step
            c_random = np.random.uniform(0, 1) * c
            alpha = max(0, phi * i / u + beta * c_random - i / u)
            delta_i = (u - i) * alpha - i  # Infection increment with recovery term
            i_new = i + delta_i
            i_new = max(0, min(i_new, u))  # Ensure infected doesn't exceed allowed or fall below 0
        else:
            i_new = 0

        return u, i_new

    # Create the phase plot
    plt.figure(figsize=(12, 10))

    # Simulate and plot trajectories
    colors = plt.cm.jet(np.linspace(0, 1, num_trajectories))

    for idx, color in enumerate(colors):
        u_0 = np.random.uniform(max_students / 2, max_students)  # Random initial allowed students
        i_0 = np.random.uniform(0, u_0 / 2)  # Random initial infected students
        u, i = u_0, i_0

        # Track the trajectory
        trajectory_u = [u_0]
        trajectory_i = [i_0]

        for step in range(num_steps):
            u, i = stochastic_step(u, i, phi, beta, np.mean(community_risk))
            trajectory_u.append(u)
            trajectory_i.append(i)

        # Plot the trajectory
        plt.plot(trajectory_u, trajectory_i, '-', color=color, linewidth=2, label=f'Trajectory {idx + 1}')
        plt.plot([u_0], [i_0], 'o', color=color, markersize=8)  # Mark starting point

    # Add labels and legend
    plt.xlabel('Allowed Students')
    plt.ylabel('Infected Students')
    plt.title(
        f'Stochastic Phase Diagram: Infected vs. Allowed Students\nφ={phi:.4f}, β={beta:.4f}, c={np.mean(community_risk):.2f}')
    plt.xlim(0, max_students)
    plt.ylim(0, max_students)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add diagonal line where infected = allowed
    plt.plot([0, max_students], [0, max_students], 'gray', linestyle=':', label='Infected = Allowed')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("stochastic_phase_diagram.png")
    plt.show()


max_students = 100


plot_phase_diagram_stochastic(phi, beta, max_students, community_risk)
P, states = simulate_transition_matrix()

# Run the simulation and plot the threshold behavior

R0_values = simulate_infection_dynamics()
plot_threshold_behavior_R0()
plot_threshold_behavior(R0_values)
plot_infection_dynamics()
plot_sensitivity_phi_beta()
plot_stationary_distribution(P, states)
plot_transition_matrix(P)
plot_lyapunov_function(P, states)
plot_threshold_behavior_alpha_beta()


