import numpy as np
import math


def interior_pt(f, ineq_constraints, eq_constraints_mat, x0):
    path_history = {'path': [], 'values': []}
    t = 1.0
    prev_x = x0.copy()
    path_history['path'].append(prev_x)
    path_history['values'].append(f(prev_x)[0])

    prev_f, prev_grad, prev_hass = get_values_after_log_barrier(f, ineq_constraints, x0, t)
    
    num_of_constraints = len(ineq_constraints)
    prev_x0 = x0

    path_history['path'].append(prev_x0.copy())
    path_history['values'].append(f(prev_x0.copy())[0])

    while (num_of_constraints / t) > 1e-8:
        for i in range(10):
            dir = find_direction(prev_hass, eq_constraints_mat, prev_grad)
            step_len = wolfe_condition_with_backtracking(f, prev_x0, prev_f, prev_grad, dir)
            next_x0 = prev_x0 + dir * step_len
            
            next_f, next_grad, next_hass = get_values_after_log_barrier(f, ineq_constraints, next_x0, t)

            if np.linalg.norm(next_grad, 2) ** 2 * 0.5 < 1e-8:
                break

            prev_x0 = next_x0
            prev_f = next_f
            prev_grad = next_grad
            prev_hass = next_hass
        
        path_history['path'].append(prev_x0.copy())
        path_history['values'].append(f(prev_x0.copy())[0])
        t *= 10
    
    return prev_x0, f(prev_x0.copy())[0], path_history


def log_barrier(ineq_constraints, x0):
    x0_dim = x0.shape[0]
    log_f = 0
    log_g = np.zeros((x0_dim))
    log_h = np.zeros((x0_dim, x0_dim))

    for constraint in ineq_constraints:
        f, g, h = constraint(x0)
        inv_f = -1.0 / f
        log_f += math.log(-f)
        log_g += inv_f * g

        grad = g / f
        grad_dim = grad.shape[0]
        grad_tile = np.tile(grad.reshape(grad_dim, -1), (1, grad_dim)) * np.tile(grad.reshape(grad_dim, -1).T, (grad_dim, 1))
        log_h += (h * f - grad_tile) / f ** 2
    
    return -log_f, log_g, -log_h


def find_direction_eq(previous_hessian, A, previous_gradiant):
    num_constraints = A.shape[0]

    # Construct the KKT matrix
    kkt_matrix = np.block([
        [previous_hessian, A.T],
        [A, np.zeros((num_constraints, num_constraints))]
    ])

    # Construct the right-hand side vector for the KKT system
    rhs_vector = np.concatenate([-previous_gradiant, np.zeros(num_constraints)])

    # Solve the KKT system
    solution = np.linalg.solve(kkt_matrix, rhs_vector)

    # Only the part of the solution corresponding to the primal variables is needed
    return solution[:A.shape[1]]


def find_direction_no_eq(previous_hassian, previous_gradiant):
    return np.linalg.solve(previous_hassian, -previous_gradiant)


def find_direction(previous_hassian, A, previous_gradiant):
    if A is not None:
        return find_direction_eq(previous_hassian, A, previous_gradiant)
    return find_direction_no_eq(previous_hassian, previous_gradiant)
 

def wolfe_condition_with_backtracking(f, x, val, gradient, direction, max_iter=10):
    alpha = 1.0
    f_x_0 = val
    dot_grad = np.dot(gradient, direction)

    iter_count = 0
    while iter_count < max_iter:
        curr_val = f(x + alpha * direction)[0]  # Assuming f returns function value as the first element

        if curr_val <= f_x_0 + 0.01 * alpha * dot_grad:
            return alpha

        alpha *= 0.5
        iter_count += 1

        # Break if alpha becomes too small
        if alpha < 1e-6:
            break

    return alpha


def get_values_after_log_barrier(f, ineq_constraints, x0, t):
    val, grad, hass = f(x0)

    # Get the logarithmic barrier terms for the inequality constraints at x0
    log_f, log_g, log_h = log_barrier(ineq_constraints, x0)

    # Combine the original function's outputs with the barrier terms
    prev_f = t * val + log_f
    prev_grad = t * grad + log_g
    prev_hass = t * hass + log_h

    return prev_f, prev_grad, prev_hass
