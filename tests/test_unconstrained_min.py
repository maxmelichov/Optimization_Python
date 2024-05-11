import unittest
import numpy as np
import sys
import os
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(os.path.join(path, 'src'))
sys.path.insert(0, os.path.join(path, 'src'))
from unconstrained_min import LineSearchOptimizer
from examples import (rosenbrock_function, boyds_function, quadratic_function_circle,
quadratic_function_ellipse, quadratic_function_rotated_ellipse, linear_function)
from utils import plot_contours, plot_contours_linear, plot_iterations

class TestLineSearchOptimization(unittest.TestCase):
    def setUp(self):
        # Initial points and other constants as per your assignment instructions
        self.x0_quadratic = np.array([1, 1])
        self.x0_rosenbrock = np.array([-1, 2])
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.max_iter_rosenbrock = 10000


    def test_quadratic_function_circle(self):
        # Define the quadratic function with Q as an identity matrix
        def func(x, hessian_needed=False):
            return quadratic_function_circle(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_quadratic, self.obj_tol, self.param_tol, self.max_iter)
        print(optimizer.f, optimizer.x0, optimizer.obj_tol, optimizer.param_tol, optimizer.max_iter)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()
        final_x_nt, final_f_nt, success_nt = optimizer.newton_method()

        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
            (optimizer.iteration_path_nt, 'Newton\'s Method')
        ]
        # plot results
        plot_contours(quadratic_function_circle, [-1.5, 1.5], [-1.5, 1.5], paths=paths, title="quadratic function circle")
        plot_iterations(optimizer.iteration_path_gd, optimizer.iteration_path_nt, title="quadratic function circle")


    def test_quadratic_function_ellipse(self):
        # Define the quadratic function with Q as a diagonal matrix
        def func(x, hessian_needed=False):
            return quadratic_function_ellipse(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_quadratic, self.obj_tol, self.param_tol, self.max_iter)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()
        final_x_nt, final_f_nt, success_nt = optimizer.newton_method()
        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
            (optimizer.iteration_path_nt, 'Newton\'s Method')
        ]
        plot_contours(func, [-1.5, 1.5], [-1.5, 1.5], paths=paths, title="quadratic function ellipse")
        plot_iterations(optimizer.iteration_path_gd, optimizer.iteration_path_nt, title="quadratic function ellipse")


    
    def test_quadratic_function_rotated_ellipse(self):
        # Define the quadratic function with Q as a rotated matrix
        def func(x, hessian_needed=False):
            return quadratic_function_rotated_ellipse(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_quadratic, self.obj_tol, self.param_tol, self.max_iter)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()
        final_x_nt, final_f_nt, success_nt = optimizer.newton_method()
        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
            (optimizer.iteration_path_nt, 'Newton\'s Method')
        ]
        plot_contours(func, [-1.5, 1.5], [-1.5, 1.5], paths=paths, title="Quadratic Function Contour")
        plot_iterations(optimizer.iteration_path_gd, optimizer.iteration_path_nt, title="Quadratic Function Contour")

    
    def test_rosenbrock_function(self):
        # Define the Rosenbrock function
        def func(x, hessian_needed=False):
            return rosenbrock_function(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_rosenbrock, self.obj_tol, self.param_tol, self.max_iter_rosenbrock)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()
        final_x_nt, final_f_nt, success_nt = optimizer.newton_method()

        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
            (optimizer.iteration_path_nt, 'Newton\'s Method')
        ]
        # Check if the optimization was successful

        # Optionally, plot results
        plot_contours(func, [-2.5, 2.5], [-2.5, 2.5], paths=paths, title="Rosenbrock Function Contour")
        plot_iterations(optimizer.iteration_path_gd, optimizer.iteration_path_nt, title="Rosenbrock Function Contour")

        
    def test_boyds_function(self):
        # Define Boyd's function
        def func(x, hessian_needed=False):
            return boyds_function(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_quadratic, self.obj_tol, self.param_tol, self.max_iter)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()
        final_x_nt, final_f_nt, success_nt = optimizer.newton_method()

        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
            (optimizer.iteration_path_nt, 'Newton\'s Method')
        ]
        plot_contours(func, [-1.5, 1.5], [-1.5, 1.5], paths=paths, title="Boyd's Function Contour")
        plot_iterations(optimizer.iteration_path_gd, optimizer.iteration_path_nt, title="Boyd's Function Contour")

    
    def test_linear_function(self):
        # Define a linear function
        
        def func(x, hessian_needed=False):
            return linear_function(x, hessian_needed)
        optimizer = LineSearchOptimizer(func, self.x0_quadratic, self.obj_tol, self.param_tol, self.max_iter)
        final_x_gd, final_f_gd, success_gd = optimizer.gradient_descent()

        paths = [
            (optimizer.iteration_path_gd, 'Gradient Descent'),
        ]
        plot_contours_linear(func, [-1.5, 1.5], [-1.5, 1.5], paths=paths, title="Linear Function Contour")
        plot_iterations(optimizer.iteration_path_gd, [], title="Linear Function Contour")


if __name__ == '__main__':
    unittest.main()
