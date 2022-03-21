import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            res[i][j] = np.linalg.norm(X[i]-Y[j])
    return res


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            res[i][j] = np.linalg.norm(X[i]-Y[j], 1)
    return res

# if __name__ == "__main__":
#     x = np.random.rand(3, 3)
#     y = np.random.rand(3,3)
#     print(x)
#     print(y)
#     test1 = euclidean_distances1(x, y)
#     test2 = euclidean_distances(x, y)
#     print(test1)
#     print(test2)