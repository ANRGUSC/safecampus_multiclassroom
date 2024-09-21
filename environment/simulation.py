import numpy as np
SEED = 42
np.random.seed(SEED)

def simulate_infections_n_classrooms(n_classes, alpha_m, beta, phi, current_infected, allowed_students, community_risk):

    """
    Simulate the number of new infections in each classroom using the updated infection model.

    Args:
        n_classes: Number of classrooms.
        alpha_m: In-class transmission rate.
        beta: Community infection rate.
        phi: Cross-classroom transmission rate.
        current_infected: List of current infected students per classroom.
        allowed_students: List of allowed students per classroom.
        community_risk: Community risk values for each classroom.

    Returns:
        List of new infections per classroom.
    """
    new_infected = []

    for i in range(n_classes):
        # Ensure that current_infected[i] and allowed_students[i] are scalar
        current_infected_i = current_infected[i]
        allowed_students_i = allowed_students[i]
        community_risk_i = community_risk[i]  # Ensure this is a scalar for each classroom
        # Within-classroom infections (scalar for each classroom)
        in_class_term = alpha_m[i] * current_infected_i * allowed_students_i
        # Community infections (scalar for each classroom)
        community_term = beta[i] * community_risk_i * allowed_students_i ** 2

        # Cross-class infections (aggregate scalar)
        cross_class_infections = 0
        for j in range(n_classes):
            if i != j:
                cross_class_infections += phi * current_infected[j] * allowed_students[j]


        # Total infected for this classroom (scalar for each classroom)
        total_infected = in_class_term + community_term + cross_class_infections
        # print(f'total_infected for class {i}: {total_infected}')  # Debugging output

        # Ensure total_infected is capped by allowed_students and convert to integer
        total_infected = np.minimum(total_infected, allowed_students_i)
        # print("total_infected after capping:", total_infected)  # Debugging output
        new_infected.append(int(total_infected))  # Convert scalar to integer for each classroom

    return new_infected





# def simulate_random_actions(n_classes, alpha_m, beta, phi, community_risks, max_steps=20):
#     """
#     Simulate infections over time with random actions (number of allowed students) in each classroom.
#     Allowed actions are restricted to the set {0, 50, 100}.
#
#     Args:
#         n_classes: Number of classrooms.
#         shared_matrix: Matrix of infection sharing between classrooms.
#         alpha_m: In-class transmission rate.
#         beta: Community infection rate.
#         phi: Cross-classroom transmission rate.
#         community_risks: Array of community risk values over time for each classroom.
#         max_steps: Number of steps (weeks) to simulate.
#
#     Returns:
#         Infected history: A list where each element is the number of infected students at each time step.
#     """
#     total_students = 100
#     allowed_actions = [0, 50, 100]  # The set of possible allowed students
#     current_infected = [np.random.randint(0, 5) for _ in range(n_classes)]  # Random initial infected students
#     infected_history = [current_infected[:]]  # Track infection count over time
#
#     # Simulate for max_steps
#     for step in range(max_steps):
#         # Select a random action from {0, 50, 100} for each classroom
#         allowed_students = [np.random.choice(allowed_actions) for _ in range(n_classes)]
#         current_risks = community_risks[:, step]  # Community risks for the current step
#
#         # Simulate infections for this step
#         new_infected = simulate_infections_n_classrooms(
#             n_classes, alpha_m, beta, phi, current_infected, allowed_students, current_risks
#         )
#
#         # Update current infected with the new values
#         current_infected = new_infected
#         infected_history.append(current_infected[:])
#
#         # Print the current step's results
#         print(f"Step {step + 1}: Allowed students = {allowed_students}, Infected = {new_infected}")
#
#     return infected_history
#
#
# # Example usage
# if __name__ == "__main__":
#     n_classes = 2  # Number of classrooms
#     s_shared = 10  # Shared infections
#     alpha_m = 0.005  # Infection rate per classroom
#     beta = 0.01  # Community infection rate
#     phi = 0.005  # Cross-classroom transmission rate
#     community_risks = np.random.uniform(0, 1, (n_classes, 20))  # Simulate community risks for 20 steps
#     shared_matrix = np.ones((n_classes, n_classes)) * s_shared  # Simple shared infection matrix
#
#     # Simulate with random actions from the set {0, 50, 100} for 20 steps
#     simulate_random_actions(n_classes, alpha_m, beta, phi, community_risks, max_steps=20)
