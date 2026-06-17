import numpy as np

SEED = 42
np.random.seed(SEED)


def simulate_infections_n_classrooms(n_classes, alpha_m, beta, current_infected, allowed_students,
                                     community_risk, shared_student_fraction=0.3):
    """
    Updated simulation function.
    For each classroom i:

    I(i) = alpha_m[i] * current_infected[i] * allowed_students[i]
           + beta[i] * community_risk[i] * allowed_students[i]^2
           + shared_students * p_i

    where:
      - shared_students = int(allowed_students[i] * shared_student_fraction)
      - p_i is the average infection proportion in the other classrooms.

    The cross-classroom coupling strength is the shared-student fraction f itself
    (the former separate transmission rate phi/delta has been removed).
    """
    current_inf = np.array(current_infected)
    allowed = np.array(allowed_students)
    comm_risk = np.array(community_risk)
    alpha_arr = np.array(alpha_m)
    beta_arr = np.array(beta)

    # Within-classroom infections
    in_class_term = alpha_arr * current_inf * allowed

    # Community risk infections
    community_term = beta_arr * comm_risk * (allowed ** 2)

    # Compute average infection proportion from the other classrooms
    prop = np.zeros(n_classes)
    mask = allowed > 0
    prop[mask] = current_inf[mask] / allowed[mask]
    
    total_prop = np.sum(prop)
    if n_classes > 1:
        avg_prop = (total_prop - prop) / (n_classes - 1)
    else:
        avg_prop = np.zeros(n_classes)

    # Compute number of shared students based on the fraction
    shared_students = (allowed * shared_student_fraction).astype(int)

    # Cross-classroom infections using the population game formulation
    cross_class_term = shared_students * avg_prop

    total_infected = in_class_term + community_term + cross_class_term

    # Ensure the new infections do not exceed the number of allowed students
    total_infected = np.minimum(total_infected, allowed)
    
    return total_infected.astype(int).tolist()

