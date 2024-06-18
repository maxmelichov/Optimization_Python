import numpy as np
import math


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    path_history = dict(path=[], values=[])
    t = 1

    # Get initial function value, gradient, and Hessian using the logarithmic barrier approach
    current_f_value, current_gradient, current_hessian = get_values_after_log_barrier(f, ineq_constraints, x0, t)
    
    # Store initial values
    current_point = x0
    path_history['path'].append(current_point.copy())
    path_history['values'].append(f(current_point.copy())[0])

    # Continue iterations until the ratio of the number of constraints to t is small enough
    while (len(ineq_constraints) / t) > 1e-8:
        for _ in range(10):
            # Calculate the descent direction
            direction = find_direction(current_hessian, eq_constraints_mat, current_gradient)
            # Find step length using the Wolfe conditions with backtracking
            step_length = wolfe_condition_with_backtracking(f, current_point, current_f_value, current_gradient, direction, ineq_constraints, t)
            # Update the point
            next_point = current_point + direction * step_length
            
            # Update function value, gradient, and Hessian
            next_f_value, next_gradient, next_hessian = get_values_after_log_barrier(f, ineq_constraints, next_point, t)

            # Check for convergence using the stopping criterion based on the lambda value
            lambda_value = np.sqrt(np.dot(direction, np.dot(next_hessian, direction.T)))
            if 0.5 * (lambda_value ** 2) < 1e-8:
                break

            # Update the current estimates
            current_point = next_point
            current_f_value = next_f_value
            current_gradient = next_gradient
            current_hessian = next_hessian
        
        # Update path history after completing inner loop iterations
        path_history['path'].append(current_point.copy())
        path_history['values'].append(f(current_point.copy())[0])
        t *= 10  # Increase the barrier parameter
    
    # Return the final point, function value at that point, and the path history
    return current_point, f(current_point.copy())[0], path_history


def log_barrier(inequality_constraints, x0):
    # Determine the dimensionality of x0
    x0_dim = x0.shape[0]
    # Initialize the log barrier function value, gradient, and Hessian
    log_barrier_value = 0
    log_barrier_gradient = np.zeros(x0_dim)
    log_barrier_hessian = np.zeros((x0_dim, x0_dim))

    # Iterate over each inequality constraint
    for constraint in inequality_constraints:
        f_value, gradient, hessian = constraint(x0)
        
        # Update the log barrier function value
        log_barrier_value += math.log(-f_value)
        # Update the gradient of the log barrier
        log_barrier_gradient += gradient / -f_value
        
        # Precompute the outer product of normalized gradient for Hessian update
        normalized_gradient = gradient / f_value
        outer_product = np.outer(normalized_gradient, normalized_gradient)
        
        # Update the Hessian of the log barrier
        log_barrier_hessian += (hessian / f_value - outer_product) / f_value
    
    # Return the negative of the log barrier function, gradient, and Hessian
    return -log_barrier_value, log_barrier_gradient, -log_barrier_hessian


def find_direction_eq(previous_hessian, A, previous_gradient):
    left_matrix = np.block([
        [previous_hessian, A.T],   # Top half: Hessian and transpose of Jacobian
        [A, np.zeros((A.shape[0], A.shape[0]))]  # Bottom half: Jacobian and zero matrix
    ])
    # Construct the right-hand side vector, considering the negative gradient and zero padding
    right_vector = np.concatenate([-previous_gradient, np.zeros(A.shape[0])])

    # Solve the linear system to find the primal and dual variables
    solution = np.linalg.solve(left_matrix, right_vector)

    # Extract the solution relevant to the direction in the primal space
    direction = solution[:previous_hessian.shape[0]]

    return direction


def find_direction_no_eq(previous_hessian, previous_gradient):
    return np.linalg.solve(previous_hessian, -previous_gradient)


def find_direction(previous_hassian, A, previous_gradiant):
    if A is not None:
        return find_direction_eq(previous_hassian, A, previous_gradiant)
    return find_direction_no_eq(previous_hassian, previous_gradiant)
 

def wolfe_condition_with_backtracking(f, x, val, gradient, direction, ineq_constraints, t, alpha=0.01, beta=0.5, max_iter=10):
    step_length = 1
    curr_val, _, _ = f(x + step_length * direction)  # Assuming f returns a tuple with at least three items

    iteration = 0
    while iteration < max_iter:
        if curr_val <= val + alpha * step_length * np.dot(gradient, direction):
            break  # Condition met, exit the loop
        step_length *= beta
        curr_val, _, _ = f(x + step_length * direction)
        iteration += 1

    return step_length


def get_values_after_log_barrier(f, ineq_constraints, x0, t):
    # Evaluate the original objective function at x0
    val, grad, hass = f(x0)
    # Evaluate the log barrier contribution at x0
    log_f, log_g, log_h = log_barrier(ineq_constraints, x0)
    # Combine the original function evaluation with the log barrier scaled by t
    prev_f = t * val + log_f
    prev_grad = t * grad + log_g
    prev_hass = t * hass + log_h

    return prev_f, prev_grad, prev_hass
