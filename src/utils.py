import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_contours(func, x_limits, y_limits, paths=None, title="Objective Function"):
    x = np.linspace(x_limits[0], x_limits[1], 400)
    y = np.linspace(y_limits[0], y_limits[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 35), cmap='viridis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if paths:
        for path, name in paths:
            for xy in path:
                plt.plot(xy[0][0], xy[0][1], marker='o' if name == 'Newton\'s Method' else 'x', markersize=3,
                        color='red' if name == 'Newton\'s Method' else 'blue')
            
        plt.legend()

    plt.show()
    
def plot_contours_linear(func, x_limits, y_limits, paths=None, title="Objective Function"):
    x = np.linspace(x_limits[0], x_limits[1], 400)
    y = np.linspace(y_limits[0], y_limits[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if paths:
        for path, name in paths:
            for xy in path:
                plt.plot(xy[0][0], xy[0][1], marker='o', markersize=3,
                        color='red' if name == 'Newton\'s Method' else 'blue')
            
        plt.legend()

    plt.show()


def plot_iterations(iteration_path_gd, iteration_path_nt, title="Objective Function"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot([i for i in range(len(iteration_path_gd))], [x[1] for x in iteration_path_gd], 'b-', label='Gradient Descent')
    plt.plot([i for i in range(len(iteration_path_nt))], [x[1] for x in iteration_path_nt], 'r-', label='Newton\'s Method')
    plt.legend()
    plt.show()