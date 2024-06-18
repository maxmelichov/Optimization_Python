import unittest
import numpy as np
import sys
sys.path.insert(0, r'src')
from utils import plot_results_qp, plot_results_lp, plot_values_graph
from constrained_min import interior_pt
sys.path.insert(0, r'tests')
from examples import *


class TestMinimize(unittest.TestCase):
    def test_qp(self):
        # Setup test data for QP
        ineq_constraints_qp = [qp_ineq1, qp_ineq2, qp_ineq3]
        A = np.array([1, 1, 1]).reshape(1, 3)
        x0 = np.array([0.1, 0.2, 0.7])

        final_candidate, final_obj, history = interior_pt(qp_function, ineq_constraints_qp, A, 0, x0)
        
        qp_ineq_constraints_at_final = [c(final_candidate)[0] for c in ineq_constraints_qp]

        # Assertions to validate the results
        self.assertTrue(all(c <= 0 for c in qp_ineq_constraints_at_final), "Inequality constraints should be non-positive")


        # For the Quadratic Programming (QP) test
        plot_values_graph(history['values'], 'Iteration-wise Objective Values for QP')
        plot_results_qp(history['path'], 'Algorithm Path within QP Feasible Region')


    def test_lp(self):
        # Setup test data for LP
        ineq_constraints_lp = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
        A = None
        x0 = np.array([0.5, 0.75])

        # Perform the optimization
        final_candidate, final_obj, history = interior_pt(lp_function, ineq_constraints_lp, A, 0, x0)

        # Check inequality constraints at the final point
        lp_ineq_constraints_at_final = [c(final_candidate)[0] for c in ineq_constraints_lp]

        self.assertTrue(all(c <= 0 for c in lp_ineq_constraints_at_final), "Inequality constraints should be non-positive")

        # For the Linear Programming (LP) test
        plot_values_graph(history['values'], 'Iteration-wise Objective Values for LP')
        plot_results_lp(history['path'], 'Algorithm Path within LP Feasible Region')


