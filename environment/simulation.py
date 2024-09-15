import numpy as np


def simulate_infections_n_classrooms(n_classes, shared_matrix, alpha, beta, current_infected, allowed_students,
                                     community_risk):
    """
    Simulate the number of new infections in each classroom.

    Args:
        n_classes: Number of classrooms.
        shared_matrix: The matrix governing shared infections between classrooms.
        alpha: In-class infection rate per classroom.
        beta: Community infection rate per classroom.
        current_infected: List of current infected students per classroom.
        allowed_students: List of allowed students per classroom.
        community_risk: Community risk values for each classroom.

    Returns:
        List of new infections per classroom.
    """
    new_infected = []
    for i in range(n_classes):
        # In-class infections
        in_class_infections = alpha[i] * current_infected[i] * allowed_students[i]

        # Community infections
        community_infections = beta[i] * community_risk[i] * allowed_students[i] ** 2

        # Cross-class infections
        cross_class_infections = 0
        for j in range(n_classes):
            if i != j:
                # Shared infection risk between classrooms
                shared_infection_risk = 1 - (1 - alpha[i]) * (1 - alpha[j])
                cross_class_infections += shared_matrix[i, j] * shared_infection_risk * current_infected[j]

        # Total infected (capped by the number of allowed students)
        total_infected = int(in_class_infections + community_infections + cross_class_infections)
        total_infected = min(total_infected, allowed_students[i])

        new_infected.append(total_infected)

    return new_infected


# def simulate_random_actions(n_classes, shared_matrix, alpha, beta, community_risks, max_steps=20):
#     """
#     Simulate infections over time with random actions (number of allowed students) in each classroom.
#     Allowed actions are restricted to the set {0, 50, 100}.
#
#     Args:
#         n_classes: Number of classrooms.
#         shared_matrix: Matrix of infection sharing between classrooms.
#         alpha: In-class infection rate.
#         beta: Community infection rate.
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
#         new_infected = simulate_infections_n_classrooms(n_classes, shared_matrix, alpha, beta, current_infected,
#                                                         allowed_students, current_risks)
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
#     n_classes = 3  # Number of classrooms
#     s_shared = 10  # Shared infections
#     alpha = [0.005] * n_classes  # Infection rate
#     beta = [0.01] * n_classes  # Community infection rate
#     community_risks = np.random.uniform(0, 1, (n_classes, 20))  # Simulate community risks for 20 steps
#     shared_matrix = np.ones((n_classes, n_classes)) * s_shared  # Simple shared infection matrix
#
#     # Simulate with random actions from the set {0, 50, 100} for 20 steps
#     simulate_random_actions(n_classes, shared_matrix, alpha, beta, community_risks, max_steps=20)
