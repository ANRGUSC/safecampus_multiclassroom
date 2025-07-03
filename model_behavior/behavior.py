import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import itertools
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Function from the provided code - modified slightly for this analysis
def simulate_infections_n_classrooms(n_classes, alpha_m, beta, phi, current_infected, allowed_students, community_risk):
    """
    Simulation function using the population game formulation.
    """
    new_infected = []
    for i in range(n_classes):
        current_inf = current_infected[i]
        allowed = allowed_students[i]
        comm_risk = community_risk[i]

        # Within-classroom infections
        in_class_term = alpha_m[i] * current_inf * allowed

        # Community risk infections
        community_term = beta[i] * comm_risk * (allowed ** 2)

        # Compute average infection proportion from the other classrooms
        other_props = []
        for j in range(n_classes):
            if i != j:
                if allowed_students[j] > 0:
                    other_props.append(current_infected[j] / allowed_students[j])
                else:
                    other_props.append(0)
        avg_prop = np.mean(other_props) if other_props else 0

        # Cross-classroom infections using the population game formulation
        cross_class_term = phi * allowed * avg_prop

        total_infected = in_class_term + community_term + cross_class_term

        # Ensure the new infections do not exceed the number of allowed students
        total_infected = np.minimum(total_infected, allowed)
        new_infected.append(int(total_infected))
    return new_infected


# Generate stochastic community risk patterns
def generate_community_risk_pattern(weeks=10, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    t = np.linspace(0, 2 * np.pi, weeks)
    risk_pattern = np.zeros(weeks)

    num_components = random.randint(1, 3)  # Use 1 to 3 sine components

    # Generate the sine wave-based risk pattern
    for _ in range(num_components):
        amplitude = random.uniform(0.2, 0.4)
        frequency = random.uniform(0.5, 2.0)
        phase = random.uniform(0, 2 * np.pi)
        risk_pattern += amplitude * np.sin(frequency * t + phase)

    # Normalize and scale the risk pattern to range [0.1, 0.9]
    risk_pattern = (risk_pattern - np.min(risk_pattern)) / (np.max(risk_pattern) - np.min(risk_pattern))
    risk_pattern = 0.8 * risk_pattern + 0.1  # Scale to range [0.1, 0.9]

    # Add some noise and clamp the values between 0.1 and 0.9
    risk_pattern = [max(0.1, min(0.9, risk + random.uniform(-0.1, 0.1))) for risk in risk_pattern]

    return risk_pattern


# Simulate over time with fixed parameters
def simulate_over_time(n_classes, alpha, beta, phi, initial_infected, allowed_students, community_risk_pattern,
                       weeks=10):
    """
    Simulate the infection spread over time.

    Parameters:
    - n_classes: Number of classrooms
    - alpha, beta, phi: Model parameters
    - initial_infected: Initial number of infected students per classroom
    - allowed_students: Number of students allowed per classroom
    - community_risk_pattern: Time series of community risk values
    - weeks: Number of weeks to simulate

    Returns:
    - List of infection counts for each classroom at each week
    """
    alpha_m = [alpha] * n_classes
    beta_m = [beta] * n_classes

    infections_over_time = [initial_infected.copy()]
    current_infected = initial_infected.copy()

    for week in range(weeks):
        # Use weekly community risk from pattern (cycling if needed)
        current_risk = [community_risk_pattern[week % len(community_risk_pattern)]] * n_classes

        current_infected = simulate_infections_n_classrooms(
            n_classes, alpha_m, beta_m, phi,
            current_infected, allowed_students, current_risk
        )

        infections_over_time.append(current_infected.copy())

    return infections_over_time


# Find fixed points of the system
def find_fixed_points(alpha, beta, phi, community_risk, allowed_students):
    """
    Find the fixed points (equilibria) of the system for given parameters.

    Returns:
    - Array of fixed points (one per classroom)
    """
    n_classes = len(allowed_students)

    # Define the fixed point equations
    def fixed_point_equations(infected):
        residuals = []
        for i in range(n_classes):
            # Average proportions from other classrooms
            other_props = []
            for j in range(n_classes):
                if i != j:
                    if allowed_students[j] > 0:
                        other_props.append(infected[j] / allowed_students[j])
                    else:
                        other_props.append(0)
            avg_prop = np.mean(other_props) if other_props else 0

            # Calculate expected new infections
            in_class_term = alpha * infected[i] * allowed_students[i]
            community_term = beta * community_risk[i] * (allowed_students[i] ** 2)
            cross_class_term = phi * allowed_students[i] * avg_prop

            new_inf = in_class_term + community_term + cross_class_term
            new_inf = min(new_inf, allowed_students[i])

            # Fixed point: current = new
            residuals.append(new_inf - infected[i])

        return residuals

    # Initial guess: 10% of allowed students are infected
    initial_guess = [0.1 * allowed for allowed in allowed_students]

    # Find roots of the equations
    fixed_points = fsolve(fixed_point_equations, initial_guess)

    # Ensure solutions are valid (non-negative and not exceeding allowed)
    fixed_points = np.maximum(0, fixed_points)
    fixed_points = np.minimum(fixed_points, allowed_students)

    return fixed_points


# Check stability of fixed points
def check_stability(fixed_point, alpha, beta, phi, community_risk, allowed_students):
    """
    Check if a fixed point is stable by examining the eigenvalues of the Jacobian.

    Returns:
    - is_stable: Boolean indicating stability
    - eigenvalues: Array of eigenvalues
    """
    n_classes = len(allowed_students)
    J = np.zeros((n_classes, n_classes))

    # Construct Jacobian matrix
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                # Self-influence term (diagonal)
                J[i, i] = alpha * allowed_students[i] - 1
            else:
                # Cross-influence term (off-diagonal)
                J[i, j] = phi * allowed_students[i] / allowed_students[j] / (n_classes - 1)

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    # Stable if all real parts are negative
    is_stable = all(np.real(eigenvalues) < 0)

    return is_stable, eigenvalues


# Analyze parameter space for uncontrolled system
def analyze_uncontrolled_system(total_students, alpha_values, beta_values, phi_values):
    """
    Analyze the equilibrium behavior of the system with all students allowed.

    Returns:
    - Dictionary of results for each parameter combination
    """
    n_classes = 2  # Example with 2 classrooms
    allowed_students = [total_students] * n_classes
    community_risk = [0.5] * n_classes  # Medium risk

    results = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for phi in phi_values:
                key = (alpha, beta, phi)

                # Find equilibrium infection levels
                equilibrium = find_fixed_points(alpha, beta, phi, community_risk, allowed_students)

                # Check stability
                is_stable, eigenvalues = check_stability(equilibrium, alpha, beta, phi,
                                                         community_risk, allowed_students)

                # Calculate metrics
                total_equilibrium_infections = sum(equilibrium)
                infection_proportion = total_equilibrium_infections / (n_classes * total_students)

                # Calculate effective R0
                r0 = alpha * total_students + beta * 0.5 * (total_students ** 2)

                # Is disease endemic?
                is_endemic = total_equilibrium_infections > 1

                results[key] = {
                    "equilibrium_infections": equilibrium,
                    "total_infections": total_equilibrium_infections,
                    "infection_proportion": infection_proportion,
                    "is_stable": is_stable,
                    "eigenvalues": eigenvalues,
                    "r0": r0,
                    "is_endemic": is_endemic
                }

    return results


# Test different control policies
def test_action_impact(alpha, beta, phi, community_risk_pattern, initial_infected, total_students=100, weeks=10):
    """
    Test how different control policies (allowed students) affect outcomes.
    """
    n_classes = len(initial_infected)

    policy_outcomes = {}
    for capacity_pct in [0, 25, 50, 75, 100]:
        capacity = int(total_students * capacity_pct / 100)
        allowed = [capacity] * n_classes

        timeline = simulate_over_time(
            n_classes, alpha, beta, phi,
            initial_infected, allowed, community_risk_pattern, weeks
        )

        final_infections = timeline[-1]
        total_final = sum(final_infections)

        policy_outcomes[capacity_pct] = {
            "capacity": capacity,
            "final_infections": final_infections,
            "total_infections": total_final,
            "infection_rate": total_final / max(sum(allowed), 1),
            "timeline": timeline
        }

    return policy_outcomes


# Evaluate control impact across parameter space
def evaluate_control_impact(alpha_values, beta_values, phi_values):
    """
    Evaluate how effective control policies are across parameter space.
    """
    n_classes = 2
    initial_infected = [5] * n_classes
    total_students = 100

    # Generate a fixed risk pattern for consistency
    risk_pattern = generate_community_risk_pattern(weeks=15, seed=42)

    results = {}

    for alpha in alpha_values:
        for beta in beta_values:
            for phi in phi_values:
                key = (alpha, beta, phi)

                # Test different policies
                policy_outcomes = test_action_impact(
                    alpha, beta, phi, risk_pattern, initial_infected, total_students
                )

                # Calculate the range of outcomes
                min_outcome = min([outcome["total_infections"] for outcome in policy_outcomes.values()])
                max_outcome = max([outcome["total_infections"] for outcome in policy_outcomes.values()])
                outcome_range = max_outcome - min_outcome

                # Calculate control effectiveness
                control_effectiveness = outcome_range / max(max_outcome, 1)

                results[key] = {
                    "policy_outcomes": policy_outcomes,
                    "outcome_range": outcome_range,
                    "control_effectiveness": control_effectiveness,
                    "min_outcome": min_outcome,
                    "max_outcome": max_outcome
                }

    return results


# Plot the phase diagram
def plot_phase_diagram(results, parameter_values, total_students):
    alpha_values, beta_values, phi_values = parameter_values

    # Create figure
    fig = plt.figure(figsize=(18, 6))

    # 1. Plot R0 values
    ax1 = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(alpha_values, beta_values)
    Z = np.zeros((len(beta_values), len(alpha_values)))

    # Use middle phi value for this plot
    mid_phi = phi_values[len(phi_values) // 2]

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            Z[i, j] = results[(alpha, beta, mid_phi)]["r0"]

    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Beta')
    ax1.set_zlabel('R0')
    ax1.set_title(f'Basic Reproduction Number (Phi={mid_phi})')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2. Plot equilibrium infection proportion
    ax2 = fig.add_subplot(132, projection='3d')
    Z = np.zeros((len(beta_values), len(alpha_values)))

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            Z[i, j] = results[(alpha, beta, mid_phi)]["infection_proportion"]

    surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('Beta')
    ax2.set_zlabel('Infection Proportion')
    ax2.set_title(f'Equilibrium Infection Proportion (Phi={mid_phi})')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    # 3. Plot phase space
    ax3 = fig.add_subplot(133)
    endemic_points = []
    disease_free_points = []

    for alpha in alpha_values:
        for beta in beta_values:
            r0 = results[(alpha, beta, mid_phi)]["r0"]
            if r0 > 1:
                endemic_points.append((alpha, beta))
            else:
                disease_free_points.append((alpha, beta))

    if endemic_points:
        endemic_points = np.array(endemic_points)
        ax3.scatter(endemic_points[:, 0], endemic_points[:, 1], color='red', label='Endemic')

    if disease_free_points:
        disease_free_points = np.array(disease_free_points)
        ax3.scatter(disease_free_points[:, 0], disease_free_points[:, 1], color='blue', label='Disease-Free')

    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Beta')
    ax3.set_title(f'Phase Diagram (Phi={mid_phi})')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=300)
    plt.show()


# Plot control effectiveness
def plot_control_effectiveness(control_results, parameter_values):
    alpha_values, beta_values, phi_values = parameter_values

    # Middle phi value for 2D plots
    mid_phi = phi_values[len(phi_values) // 2]

    # Create a heatmap for control effectiveness
    plt.figure(figsize=(15, 5))

    # 1. Control effectiveness heatmap
    plt.subplot(131)
    Z = np.zeros((len(beta_values), len(alpha_values)))

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            Z[i, j] = control_results[(alpha, beta, mid_phi)]["control_effectiveness"]

    plt.imshow(Z, extent=[min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Control Effectiveness')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title(f'Control Effectiveness (Phi={mid_phi})')

    # 2. Outcome range heatmap
    plt.subplot(132)
    Z = np.zeros((len(beta_values), len(alpha_values)))

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            Z[i, j] = control_results[(alpha, beta, mid_phi)]["outcome_range"]

    plt.imshow(Z, extent=[min(alpha_values), max(alpha_values), min(beta_values), max(beta_values)],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Outcome Range')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title(f'Range of Policy Outcomes (Phi={mid_phi})')

    # 3. Policy comparison for a specific parameter set
    plt.subplot(133)
    good_alpha = alpha_values[len(alpha_values) // 2]
    good_beta = beta_values[len(beta_values) // 2]
    good_phi = phi_values[len(phi_values) // 2]

    policy_outcomes = control_results[(good_alpha, good_beta, good_phi)]["policy_outcomes"]

    for capacity_pct, outcome in policy_outcomes.items():
        timeline = outcome["timeline"]
        total_infections = [sum(week) for week in timeline]
        plt.plot(range(len(timeline)), total_infections,
                 label=f'{capacity_pct}% Capacity')

    plt.xlabel('Weeks')
    plt.ylabel('Total Infections')
    plt.title(f'Policy Comparison (α={good_alpha}, β={good_beta}, φ={good_phi})')
    plt.legend()

    plt.tight_layout()
    plt.savefig('control_effectiveness.png', dpi=300)
    plt.show()


# Main analysis function
def run_analysis():
    print("Starting comprehensive epidemic model analysis...")

    # 1. Parameter ranges to explore
    alpha_values = np.logspace(-3, -1, 5)  # from 0.001 to 0.1
    beta_values = np.logspace(-4, -2, 5)  # from 0.0001 to 0.01
    phi_values = np.logspace(-5, -3, 3)  # from 0.00001 to 0.001
    parameter_values = (alpha_values, beta_values, phi_values)

    # 2. Analyze uncontrolled system (all students allowed)
    print("\nAnalyzing uncontrolled system...")
    uncontrolled_results = analyze_uncontrolled_system(100, alpha_values, beta_values, phi_values)

    # Print some key findings
    print("\nSummary of uncontrolled system analysis:")

    # Count endemic parameter combinations
    endemic_count = sum(1 for result in uncontrolled_results.values() if result["is_endemic"])
    total_combinations = len(uncontrolled_results)
    print(
        f"Endemic parameter combinations: {endemic_count}/{total_combinations} ({endemic_count / total_combinations * 100:.1f}%)")

    # Find highest and lowest infection proportions
    max_prop = max(uncontrolled_results.values(), key=lambda x: x["infection_proportion"])
    min_prop = min(uncontrolled_results.values(), key=lambda x: x["infection_proportion"])
    print(
        f"Highest infection proportion: {max_prop['infection_proportion']:.2f} at α={max_prop['equilibrium_infections'][0]:.4f}, β={max_prop['equilibrium_infections'][1]:.4f}")
    print(
        f"Lowest infection proportion: {min_prop['infection_proportion']:.2f} at α={min_prop['equilibrium_infections'][0]:.4f}, β={min_prop['equilibrium_infections'][1]:.4f}")

    # 3. Evaluate control impact
    print("\nEvaluating control impact across parameter space...")
    control_results = evaluate_control_impact(alpha_values, beta_values, phi_values)

    # Find parameter combinations with highest and lowest control effectiveness
    max_control = max(control_results.items(), key=lambda x: x[1]["control_effectiveness"])
    min_control = min(control_results.items(), key=lambda x: x[1]["control_effectiveness"])

    print("\nSummary of control impact analysis:")
    print(f"Highest control effectiveness: {max_control[1]['control_effectiveness']:.2f}")
    print(f"   at α={max_control[0][0]:.4f}, β={max_control[0][1]:.4f}, φ={max_control[0][2]:.6f}")
    print(f"   outcome range: {max_control[1]['outcome_range']:.1f}")

    print(f"Lowest control effectiveness: {min_control[1]['control_effectiveness']:.2f}")
    print(f"   at α={min_control[0][0]:.4f}, β={min_control[0][1]:.4f}, φ={min_control[0][2]:.6f}")
    print(f"   outcome range: {min_control[1]['outcome_range']:.1f}")

    # 4. Visualize results
    print("\nGenerating phase diagram...")
    plot_phase_diagram(uncontrolled_results, parameter_values, 100)

    print("\nGenerating control effectiveness visualization...")
    plot_control_effectiveness(control_results, parameter_values)

    # 5. Analyze RL suitability
    print("\nAnalyzing RL suitability...")
    suitable_params = []
    unsuitable_params = []

    for params, result in control_results.items():
        alpha, beta, phi = params

        # Check if system satisfies RL suitability criteria
        is_suitable = (
                result["control_effectiveness"] > 0.2 and  # Control matters
                uncontrolled_results[params]["r0"] > 1.0 and  # Disease can spread
                uncontrolled_results[params]["infection_proportion"] < 0.9  # Not overwhelming
        )

        if is_suitable:
            suitable_params.append(params)
        else:
            unsuitable_params.append(params)

    print(f"\nRL-suitable parameter combinations: {len(suitable_params)}/{total_combinations}")
    if suitable_params:
        print("\nExample suitable parameter combinations:")
        for i, params in enumerate(suitable_params[:3]):
            alpha, beta, phi = params
            print(f"{i + 1}. α={alpha:.4f}, β={beta:.4f}, φ={phi:.6f}")
            print(f"   R0: {uncontrolled_results[params]['r0']:.2f}")
            print(f"   Control effectiveness: {control_results[params]['control_effectiveness']:.2f}")
            print(f"   Equilibrium infection proportion: {uncontrolled_results[params]['infection_proportion']:.2f}")

    # 6. Detailed analysis of an RL-suitable parameter set
    if suitable_params:
        # Pick the most suitable parameter combination
        best_params = max(suitable_params, key=lambda p: control_results[p]["control_effectiveness"])
        alpha, beta, phi = best_params

        print(f"\nDetailed analysis of most suitable parameter combination:")
        print(f"α={alpha:.4f}, β={beta:.4f}, φ={phi:.6f}")

        # Generate a stochastic risk pattern
        risk_pattern = generate_community_risk_pattern(weeks=20, seed=42)

        # Test impact of stochastic risk
        n_classes = 2
        initial_infected = [5] * n_classes
        allowed_students = [50] * n_classes

        print("\nSimulating with stochastic community risk...")
        timeline = simulate_over_time(
            n_classes, alpha, beta, phi,
            initial_infected, allowed_students, risk_pattern, weeks=20
        )

        # Plot the timeline with stochastic risk
        plt.figure(figsize=(10, 6))
        total_infections = [sum(week) for week in timeline]
        plt.plot(range(len(timeline)), total_infections, label='Total Infections')

        # Also plot the risk pattern
        risk_scaled = [r * 100 for r in risk_pattern]  # Scale for visibility
        plt.plot(range(len(risk_pattern)), risk_scaled, 'r--', label='Community Risk (×100)')

        plt.xlabel('Weeks')
        plt.ylabel('Count')
        plt.title('Infection Timeline with Stochastic Community Risk')
        plt.legend()
        plt.savefig('stochastic_simulation.png', dpi=300)
        plt.show()

        # Final recommendations
        print("\nFinal Recommendations for RL Application:")
        print(f"1. Recommended parameter ranges:")
        print(f"   α: {min([p[0] for p in suitable_params]):.4f} - {max([p[0] for p in suitable_params]):.4f}")
        print(f"   β: {min([p[1] for p in suitable_params]):.4f} - {max([p[1] for p in suitable_params]):.4f}")
        print(f"   φ: {min([p[2] for p in suitable_params]):.6f} - {max([p[2] for p in suitable_params]):.6f}")

        print(f"\n2. Most promising parameter combination:")
        print(f"   α={best_params[0]:.4f}, β={best_params[1]:.4f}, φ={best_params[2]:.6f}")

        print(f"\n3. System characteristics with these parameters:")
        print(f"   - Basic reproduction number: {uncontrolled_results[best_params]['r0']:.2f}")
        print(f"   - Control effectiveness: {control_results[best_params]['control_effectiveness']:.2f}")
        print(f"   - Equilibrium infection proportion: {uncontrolled_results[best_params]['infection_proportion']:.2f}")

        print("\n4. Recommended RL approach:")
        print("   - Multi-agent RL with coordination mechanisms")
        print("   - State representation should include both local (classroom) and global (community risk) information")
        print("   - Reward shaping to balance education goals (max students) with health goals (min infections)")
        print("   - Algorithms should handle stochastic transitions due to varying community risk")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    run_analysis()