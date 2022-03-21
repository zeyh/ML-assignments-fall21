import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def plot_decision_regions(
        features,
        targets,
        model,
        title: str = 'Decision Regions'):
    """
    This function produces a single plot containing a scatter plot of the
    features, targets, and decision regions of the model. It assumes a 
    "positive" class (1) and a "negative" class (0 or -1) in the targets.

    Args:
        features (np.ndarray): 2D array containing real-valued inputs.
        targets (np.ndarray): 1D array containing binary targets.
        model: a learner with .predict() method
        title: title of the plot
    Returns:
        None (plots to the active figure)
    """

    # define bounds of the domain
    min1, max1 = features[:, 0].min()-1, features[:, 0].max()+1
    min2, max2 = features[:, 1].min()-1, features[:, 1].max()+1

    # define grid for visualizing decision regions
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontally stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))

    # generate predictions over grid
    yhat = model.predict(grid)
    # print(grid[-10:-1])
    # print(yhat[-10:-1])
    
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # binary_cmap = matplotlib.colors.ListedColormap(['#9ce8ff', '#ffc773'])
    if len(np.unique(yhat)) == 1:
        if (np.unique(yhat) < 0.5).all():
            binary_cmap = matplotlib.colors.ListedColormap(['#9ce8ff'])
        else:
            binary_cmap = matplotlib.colors.ListedColormap(['#ffc773'])
    
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap=binary_cmap, alpha=0.7)

    # plot "negative" class:
    row_idx_neg = np.where(targets < 0.5)[0]
    plt.scatter(features[row_idx_neg, 0], features[row_idx_neg, 1], cmap=binary_cmap, label='negative')

    # plot "positive" class:
    row_idx_pos = np.where(targets > 0.5)[0]
    plt.scatter(features[row_idx_pos, 0], features[row_idx_pos, 1], cmap=binary_cmap, label='positive')

    plt.title(title)
    plt.xlim(min1, max1)
    plt.ylim(min2, max2)

    plt.legend()
    plt.show()



