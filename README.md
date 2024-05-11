# Numerical Optimization of Unconstrained Functions

## Project Overview
This repository contains Python implementations of numerical optimization algorithms designed for unconstrained minimization problems. The primary focus is on Gradient Descent and Newton's Method, enhanced with robust line search techniques to determine optimal step sizes effectively.

## Repository Structure
- **src/**: Houses core modules that implement the optimization algorithms and auxiliary utility functions.
- **tests/**: Contains test scripts that apply the optimization techniques across various function scenarios and generate visual outputs for analysis.

## Installation and Usage
To get started with the optimization tests and view results:
1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repository.git
   ```
2. Execute the test script:
   ```
   python test_unconstrained_min.py
   ```

## Implemented Functions
The algorithms are rigorously tested on multiple function types, including quadratic forms and the Rosenbrock function. Descriptions and visual results for each scenario are outlined below:

### 1. Quadratic Function: Circle
Tests basic algorithm behavior on a simple quadratic function with circular level sets.
- **Output Visuals**:
  - ![Circle Function Contours](plots/Circle/circle_plot.png)

  - ![Circle Function Iterations](plots/Circle/Figure_1.png)

### 2. Quadratic Function: Ellipse
Demonstrates the algorithms' performance on an elliptical quadratic function, emphasizing behavior on elongated level sets.
- **Output Visuals**:
  - ![Ellipse Function Contours](plots/Ellipse/ellipse.png)

  - ![Ellipse Function Iterations](plots/Ellipse/Figure_1.png)

### 3. Quadratic Function: Contour
Further explores optimization paths on various contour configurations.
- **Output Visuals**:
  - ![Contour Function Plots](plots/Contour/contour.png)

  - ![Contour Function Iterations](plots/Contour/Figure_1.png)

### 4. Quadratic Function: Boyd's
Focuses on a specialized Boyd’s quadratic function, testing edge cases in optimization.
- **Output Visuals**:
  - ![Boyd's Function Contours](plots/Boyd/boyd_plot.png)

  - ![Boyd's Function Iterations](plots\Boyd's\iteration.png)

### 5. Rosenbrock Function
A classic optimization test function used to evaluate convergence properties of the methods.
- **Output Visuals**:
  - ![Rosenbrock Function Contours](plots/Rosenbrock/rosenbrock_plot.png)

  - ![Rosenbrock Function Iterations](plots/Rosenbrock/iteration.png)

## Discussion
Newton's Method typically achieves faster convergence leveraging second-order information, albeit at the cost of higher computational demand and challenges with nearly singular Hessians. Gradient Descent, a first-order method, provides more consistent results across a broader range of conditions, though it may be slower, requiring more iterations for comparable precision.
