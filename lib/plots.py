import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_plots(names, feature_names, X, y, fixed_input = None, epsilon = None, title = ''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax1.plot(X_plot[:, 0], X_plot[:, 1], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.axis('equal')
    ax1.legend()

    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax2.plot(X_plot[:, 2], X_plot[:, 3], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])
    ax2.axis('equal')
    ax2.legend()

    if fixed_input is not None and epsilon is not None:
    #add rectangle to plot -> shows infinity norm 
        ax1.add_patch(Rectangle((fixed_input[0] - epsilon, fixed_input[1] - epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='pink',
                                facecolor='none',      
                                lw=4))
        ax1.set_aspect("equal", adjustable="datalim")

        ax2.add_patch(Rectangle((fixed_input[2]-epsilon, fixed_input[3]-epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='pink',
                                facecolor='none',      
                                lw=4))
        ax2.set_aspect("equal", adjustable="datalim")