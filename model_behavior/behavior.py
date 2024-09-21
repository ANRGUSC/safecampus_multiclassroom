import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
SEED = 42
np.random.seed(SEED)

# Define the infection model function
def infection_model(allowed_students, infected, community_risk, classroom_idx, alpha_m=2, beta=0.001, phi=0.005):
    u_i = allowed_students[classroom_idx]  # Allowed students in the classroom
    i_i = infected[classroom_idx]  # Current infected students in the classroom
    c_i = community_risk[classroom_idx]  # Community risk for the classroom

    # Within-classroom term
    in_class_term = alpha_m * i_i * u_i

    # Community interaction term
    community_term = beta * c_i * u_i ** 2

    # Cross-classroom interaction term
    cross_class_term = 0
    for j in range(len(allowed_students)):
        if j != classroom_idx:
            u_j = allowed_students[j]
            i_j = infected[j]
            cross_class_term += phi * i_j * u_j

    # Total infections at the next time step, capped by the classroom capacity
    total_infections = min(in_class_term + community_term + cross_class_term, u_i)

    return total_infections

# Define the R0 calculation function
def calculate_R0(classroom_idx, t, allowed_students, community_risk, alpha_m=2, beta=0.001, phi=0.005):
    u_i = allowed_students[classroom_idx, t]
    c_i = community_risk[classroom_idx, t]

    # In-class transmission
    within_classroom = alpha_m * u_i

    # Community transmission
    community_term = beta * c_i * u_i ** 2

    # Cross-classroom transmission
    cross_classroom = 0
    for j in range(allowed_students.shape[0]):
        if j != classroom_idx:
            u_j = allowed_students[j, t]
            c_j = community_risk[j, t]
            cross_classroom += phi * u_j

    # Total R0
    R0 = within_classroom + community_term + cross_classroom
    return R0

# Define the simulation function
def simulate_infection_dynamics(allowed_students, current_infected, community_risk, num_classrooms, time_steps, alpha_m, beta=0.001, phi=0.005):
    R0_values = np.zeros((num_classrooms, time_steps))

    for t in range(1, time_steps):
        for i in range(num_classrooms):
            # Calculate R0 for the current classroom at time t
            R0_i = calculate_R0(i, t, allowed_students, community_risk, alpha_m, beta, phi)
            R0_values[i, t] = R0_i

            # Use the infection model to calculate the number of infected for the next time step
            expected_infections = infection_model(
                allowed_students=allowed_students[:, t],
                infected=current_infected[:, t - 1],
                community_risk=community_risk[:, t],
                classroom_idx=i,
                alpha_m=alpha_m,
                beta=beta,
                phi=phi
            )

            # Update the number of infected students, ensuring the number is capped by classroom capacity
            current_infected[i, t] = max(0, min(allowed_students[i, t], expected_infections))

    return R0_values

# Define the plotting function
def plot_threshold_behavior(R0_values, num_classrooms, time_steps):
    plt.figure(figsize=(10, 6))

    # Use a colormap to assign distinct colors to each classroom
    colors = plt.cm.viridis(np.linspace(0, 1, num_classrooms))

    # Plot R0 for each classroom with different colors
    for i in range(num_classrooms):
        plt.plot(R0_values[i], label=f"Classroom {i + 1}", color=colors[i])

    # Plot the threshold line at R0 = 1
    plt.axhline(y=1, color='r', linestyle='--', label='$R_0 = 1$ (Threshold)')

    # Fill regions for DFE (R_0 < 1) and EE (R_0 > 1)
    plt.fill_between(np.arange(time_steps), 0, 1, color='green', alpha=0.1, label="DFE Region (R_0 < 1)")
    plt.fill_between(np.arange(time_steps), 1, np.max(R0_values), color='orange', alpha=0.1, label="EE Region (R_0 > 1)")

    # Set plot labels and title
    plt.xlabel("Time Steps")
    plt.ylabel("$R_0$")
    plt.title("Threshold Behavior of the Model (R_0 over time)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("threshold_behavior.png")
    # plt.show()  # Uncomment this line to display the plot

def plot_infection_dynamics(simulated_infected, num_classrooms, time_steps):
    """
    Plot the infection dynamics over time for each classroom based on the simulation.

    Parameters:
    simulated_infected (array): Simulated number of infected students in each classroom over time.
    num_classrooms (int): Number of classrooms.
    time_steps (int): Number of time steps in the simulation.
    """
    plt.figure(figsize=(10, 6))

    # Use a colormap to assign distinct colors to each classroom
    colors = plt.cm.plasma(np.linspace(0, 1, num_classrooms))

    # Plot the infection dynamics for each classroom
    for i in range(num_classrooms):
        plt.plot(simulated_infected[i], label=f"Classroom {i + 1}", color=colors[i])

    # Set plot labels and title
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Infected Students")
    plt.title("Simulated Infection Dynamics in Classrooms Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("infection_dynamics.png")
    # plt.show()  # Uncomment this line to display the plot

def plot_allowed_vs_R0_curve(num_classrooms, alpha_m=2.0, beta=0.001, phi=0.005, max_students=100):
    """
    Plot Allowed vs R0 curve for each classroom, highlighting regions for DFE and EE with different shades.

    Parameters:
    num_classrooms (int): Number of classrooms.
    alpha_m (float): Transmission rate within the classroom.
    beta (float): Transmission rate from the community.
    phi (float): Cross-classroom transmission rate.
    max_students (int): Maximum number of allowed students.
    """
    # Calculate a mean community risk value from a fixed distribution
    community_risk_mean = np.mean(np.random.uniform(0.5, 1.0, size=100))

    # Generate the range of allowed student values
    allowed_values = np.linspace(0, max_students, 100)
    R0_values = np.zeros((num_classrooms, len(allowed_values)))

    for i in range(num_classrooms):
        for j, allowed in enumerate(allowed_values):
            # Ensure that infected does not exceed allowed
            infected = min(allowed, np.random.uniform(1, allowed))  # Random infected, constrained by allowed

            # Recalculate R0 based on the varying allowed student values and the fixed community risk mean
            within_classroom_R0 = alpha_m * infected  # Based on the number of infected
            community_R0 = beta * community_risk_mean * allowed**2
            cross_classroom_R0 = phi * infected  # Cross-classroom infections are still based on infected students

            # Total R0 for the classroom
            R0_values[i, j] = within_classroom_R0 + community_R0 + cross_classroom_R0

    # Plotting the curves
    plt.figure(figsize=(10, 6))

    # Use a colormap to assign distinct colors to each classroom
    colors = plt.cm.cool(np.linspace(0, 1, num_classrooms))
    dfe_colors = plt.cm.Greens(np.linspace(0.3, 0.7, num_classrooms))  # Lighter shades for DFE
    ee_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, num_classrooms))  # Lighter shades for EE

    # Plot R0 vs Allowed for each classroom
    for i in range(num_classrooms):
        plt.plot(allowed_values, R0_values[i], label=f"Classroom {i + 1}", color=colors[i])

        # Fill regions for DFE (R_0 < 1) and EE (R_0 >= 1) with different shades for each classroom
        plt.fill_between(allowed_values, 0, R0_values[i], where=(R0_values[i] < 1), color=dfe_colors[i], alpha=0.3,
                         label=f"DFE Region - Classroom {i + 1}")
        plt.fill_between(allowed_values, 1, R0_values[i], where=(R0_values[i] >= 1), color=ee_colors[i], alpha=0.3,
                         label=f"EE Region - Classroom {i + 1}")

    # Plot the threshold line at R0 = 1
    plt.axhline(y=1, color='red', linestyle='--', label='$R_0 = 1$ (Threshold)')

    # Set plot labels and title
    plt.xlabel("Allowed Students")
    plt.ylabel("$R_0$")
    plt.title(f"Allowed vs $R_0$ for Classrooms\n"
              f"($\\alpha_m$={alpha_m}, $\\beta$={beta}, $\\phi$={phi}, CR mean={community_risk_mean:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig("allowed_vs_R0_curve_fixed.png")
    # plt.show()  # Uncomment this line to display the plot




def calculate_R0_surface(alpha_m, beta, phi, allowed, community_risk_mean):
    """
    Calculate R0 based on transmission rates and return the R0 values.
    """
    within_classroom_R0 = alpha_m * allowed  # Based on allowed students
    community_R0 = beta * community_risk_mean * allowed**2
    cross_classroom_R0 = phi * allowed  # Cross-classroom infections

    # Total R0
    R0 = within_classroom_R0 + community_R0 + cross_classroom_R0
    return R0

def plot_R0_surface_3D(allowed=100, community_risk_mean=0.74):
    """
    Plot the 3D surface plot to show how R0 changes with alpha_m, beta, and phi.

    Parameters:
    allowed (int): Number of allowed students.
    community_risk_mean (float): The mean community risk factor.
    """
    # Create ranges for alpha_m, beta, and phi
    alpha_m_values = np.linspace(0.001, 0.5, 50)  # Range for alpha_m
    beta_values = np.linspace(0.0001, 0.01, 50)  # Range for beta
    phi_values = np.linspace(0.001, 0.5, 50)  # Range for phi

    # Create a meshgrid for alpha_m and beta
    alpha_m_grid, beta_grid = np.meshgrid(alpha_m_values, beta_values)

    # Calculate R0 values for each combination of alpha_m, beta, and phi
    R0_values = np.zeros((len(alpha_m_values), len(beta_values), len(phi_values)))

    for i in range(len(alpha_m_values)):
        for j in range(len(beta_values)):
            for k, phi in enumerate(phi_values):
                R0_values[i, j, k] = calculate_R0_surface(alpha_m_values[i], beta_values[j], phi, allowed, community_risk_mean)

    # Select a slice for phi to represent it on the z-axis
    phi_slice = phi_values  # z-axis will represent phi values

    # Calculate the mean R0 values for the 2D surface (averaging over phi)
    mean_R0_values = np.mean(R0_values, axis=2)

    # Create a figure for the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface with phi on the z-axis
    surf = ax.plot_surface(alpha_m_grid, beta_grid, phi_slice[:, None], facecolors=plt.cm.viridis(mean_R0_values / np.max(mean_R0_values)), edgecolor='none')
    ax.set_xlim(0, max(alpha_m_values))  # Set x-axis (alpha_m) limit from 0
    ax.set_ylim(0, max(beta_values))  # Set y-axis (beta) limit from 0
    ax.set_zlim(0, max(phi_values))

    # Add horizontal color bar for the R0 values
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(mean_R0_values)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, orientation='horizontal', pad=0.1, label='$R_0$')

    # Set axis labels
    ax.set_xlabel(r'$\alpha_m$ (Within-classroom transmission rate)')
    ax.set_ylabel(r'$\beta$ (Community transmission rate)')
    ax.set_zlabel(r'$\phi$ (Cross-classroom transmission rate)')
    ax.set_title('Surface Plot of $R_0$ Based on Transmission Rates')

    plt.tight_layout()
    plt.savefig("R0_surface_plot_3D_updated.png")
    # plt.show()  # Uncomment to display the plot


# Main function to run the simulation
def main():
    # Simulation parameters
    num_classrooms = 2  # Number of classrooms
    time_steps = 52  # Number of time steps
    max_students = 100  # Maximum number of students per classroom

    # Transmission rates
    alpha_m = 0.01  # Transmission rate within the classroom
    beta = 0.001  # Transmission rate from the community
    phi = 0.000001 # Cross-classroom transmission rate

    # Community risk values for each classroom over time (randomized for simulation)
    np.random.seed(42)  # For reproducibility
    community_risk = np.random.uniform(0.1, 1.0, (num_classrooms, time_steps))

    # Number of students allowed in each classroom (assumed to be max_students for simplicity)
    allowed_students = np.full((num_classrooms, time_steps), max_students)

    # Initialize infection dynamics for each classroom (initially 1 infected individual in each classroom)
    current_infected = np.zeros((num_classrooms, time_steps))
    current_infected[:, 0] = 1  # Start with 1 infected student per classroom

    # Run the simulation
    R0_values = simulate_infection_dynamics(
        allowed_students=allowed_students,
        current_infected=current_infected,
        community_risk=community_risk,
        num_classrooms=num_classrooms,
        time_steps=time_steps,
        alpha_m=alpha_m,
        beta=beta,
        phi=phi
    )

    # Plot the threshold behavior of R0 over time
    plot_threshold_behavior(R0_values, num_classrooms, time_steps)

    # Plot the infection dynamics using the simulated infected data
    plot_infection_dynamics(current_infected, num_classrooms, time_steps)

    # Plot Allowed vs R0 for both classrooms, showing DFE and EE regions
    plot_allowed_vs_R0_curve(num_classrooms, alpha_m, beta, phi, max_students)

    plot_R0_surface_3D(100, 0.4)


if __name__ == "__main__":
    main()
