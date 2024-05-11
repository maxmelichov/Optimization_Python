import numpy as np


def safe_inverse(hessian, epsilon=1e-8):
        try:
            inv_hessian = np.linalg.inv(hessian)
            return inv_hessian
        except np.linalg.LinAlgError:
            # If the matrix is singular, add a small value to the diagonal and try again
            hessian_reg = hessian + epsilon * np.eye(hessian.shape[0])
            return np.linalg.inv(hessian_reg)


class LineSearchOptimizer:
    def __init__(self, f, x0 = None, obj_tol = 1e-12, param_tol = 1e-8, max_iter = 100):
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.iteration_path_gd = []
        self.iteration_path_nt = []

    def gradient_descent(self):
        x = self.x0
        f_x, grad_val, hess_val = self.f(x, hessian_needed = False)
        for i in range(self.max_iter):
            self.iteration_path_gd.append([x, f_x])
            print(f"Iteration {i}: x = {x}, f(x) = {f_x}")
            p = -grad_val
            step_size = self.line_search(x, p)
            print(f"Step size: {step_size}")
            x_new = x + step_size * p
            f_new, grad_new, hess_new = self.f(x_new, hessian_needed = False)
            if np.linalg.norm(x_new - x) < self.param_tol or \
               abs(f_new - f_x) < self.obj_tol:
                return x_new, f_new, True
            x = x_new
            f_x, grad_val, hess_val = f_new, grad_new, hess_new
        return x, f_x, False

    def newton_method(self):
        x = self.x0
        f_x, grad_val, hess_val = self.f(x, hessian_needed = True)
        for i in range(self.max_iter):
            self.iteration_path_nt.append([x, f_x])
            print(f"Iteration {i}: x = {x}, f(x) = {f_x}")
            hess_inv = safe_inverse(hess_val)
            hess_inv = np.linalg.pinv(hess_val)
            p = -np.dot(hess_inv, grad_val)
            step_size = self.line_search(x, p)
            x_new = x + step_size * p
            f_new, grad_new, hess_new = self.f(x_new, hessian_needed = True)
            lambda_squared = (0.5 * p.T @ (hess_new @ p)) ** 0.5
            if lambda_squared < self.obj_tol or \
                  np.linalg.norm(x - x_new) < self.param_tol:
                return x_new, f_new, True
            x = x_new
            f_x, grad_val, hess_val = f_new, grad_new, hess_new
        return x, f_new, False
    

    def line_search(self, x, p):
        alpha = 1.0
        beta = 0.5
        c1 = 0.01
        while (self.f(x + alpha * p)[0] > self.f(x)[0] + c1 * alpha * np.dot(self.f(x)[1], p)):
            alpha *= beta
            if alpha < 1e-6:
                break
        return alpha
