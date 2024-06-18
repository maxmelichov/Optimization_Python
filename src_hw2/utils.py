import numpy as np
import matplotlib.pyplot as plt

def plot_results_qp(path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    # Plotting the background feasible region as a triangle
    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    # Plot the path through the feasible region
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path', color='blue')
    # Highlight the final point
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, color='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(45, 45)  # Sets the initial viewing angle
    plt.show()

def plot_results_lp(path, title):
    fig, ax = plt.subplots()
    path = np.array(path)

    # Define lines for inequality constraints
    constraints_ineq = {
        'y = 0': (np.linspace(-1, 3, 1000), np.zeros(1000)),
        'y = 1': (np.linspace(-1, 3, 1000), np.ones(1000)),
        'x = 2': (np.full(1000, 2), np.linspace(-2, 2, 1000)),
        'y = -x + 1': (np.linspace(-1, 3, 1000), -np.linspace(-1, 3, 1000) + 1)
    }

    # Plot each constraint line
    for label, (x_vals, y_vals) in constraints_ineq.items():
        ax.plot(x_vals, y_vals, label=label)

    # Fill the feasible region defined by the intersection of constraints
    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], 'lightgray', label='Feasible region', alpha=0.5)
    # Plot the optimization path
    ax.plot(path[:, 0], path[:, 1], color='black', label='Path')
    # Highlight the final point
    ax.scatter(path[-1][0], path[-1][1], s=50, color='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()

def plot_values_graph(values, title):
    fig, ax = plt.subplots()
    ax.plot(values, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Objective values')
    plt.grid(True)
    plt.show()
