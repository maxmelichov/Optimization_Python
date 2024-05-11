import numpy as np

def quadratic_function_circle(x, hessian_needed=False):
    Q = Q_circle
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if hessian_needed else None
    return f, g, h

def quadratic_function_ellipse(x, hessian_needed=False):
    Q = Q_ellipse
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if hessian_needed else None
    return f, g, h

def quadratic_function_rotated_ellipse(x, hessian_needed=False):
    Q = Q_rotated_ellipse
    f = np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q + Q.T, x)
    h = Q + Q.T if hessian_needed else None
    return f, g, h


def rosenbrock_function(x, flag_hessian):
    f = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
    g = np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
 
    if flag_hessian:
        return f, g, np.array([[-400 * x[1] + 1200 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200]]).squeeze()
 
    return f, g, None

def linear_function(x, hessian_needed=False):
    a = np.array([1, 2])
    f = np.dot(a.T, x)
    g = a
    h = np.zeros((len(x), len(x))) if hessian_needed else None
    return f, g, h

def boyds_function(x, hessian_needed=False):
    x1, x2 = x[0], x[1]
    f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
                  3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)])
    if hessian_needed:
        h = np.array([
            [np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1), 
             3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)],
            [3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1),
             9 * (np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1))]
        ])
    else:
        h = None
    return f, g, h

# Examples of Q matrices for quadratic functions
Q_circle = np.array([[1, 0], [0, 1]])
Q_ellipse = np.array([[1, 0], [0, 100]])
Q_rotated_ellipse = np.dot(np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]]).T,
                           np.dot(np.array([[100, 0], [0, 1]]),
                                  np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])))

# Example usage:

