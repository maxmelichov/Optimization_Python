This project is focused on implementing and evaluating numerical optimization techniques in Python, specifically targeting unconstrained minimization problems. The core algorithms implemented include Gradient Descent and Newton's Method, accompanied by robust line search strategies to ensure optimal step sizes.

Repository Structure
src/: Contains the core modules for the optimization algorithms and utility functions.
tests/: Includes test modules that apply the optimization algorithms to a variety of test functions and plots the results.
Installation and Usage
Clone the repository using:

bash
Copy code
git clone https://github.com/your-username/your-repository.git
To run the optimization tests and visualize the function minimizations, navigate to the project directory and execute:

bash
Copy code
python test_unconstrained_min.py
Implemented Functions
The optimization algorithms are tested on various functions including quadratic forms and the Rosenbrock function. Detailed explanations and visual results for each test case are provided below:

1. Quadratic Function: Circle
This function tests the basic behavior of both algorithms on a simple quadratic function with a circular level set.
[plot_contours]("plots\Circle\circle plot.png")
[iteration]("plots\Circle\Figure_1.png")

2. Quadratic Function: Ellipce

3. Quadratic Function: Contour


4. Quadratic Function: Boyd's
Evaluates the algorithms on an elliptical quadratic function, highlighting their behavior on elongated level sets.
[plot_contours]("plots\Circle\circle plot.png")
[iteration]("plots\Circle\Figure_1.png")


5. Rosenbrock Function
A classic test function for optimization algorithms, used here to demonstrate the convergence characteristics of the implemented methods.
[plot_contours]("plots\Circle\circle plot.png")
[iteration]("plots\Circle\Figure_1.png")


Discussion
Newton's Method generally achieves faster convergence due to its use of second-order information. However, it requires the computation of the Hessian matrix, which can be computationally expensive and problematic in cases where the Hessian is nearly singular. Gradient Descent, using only first-order derivatives, is more robust in such scenarios but may require more iterations to converge.
