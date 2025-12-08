import numpy as np


def z_scoring(matrix):
    new_matrix = np.asmatrix(matrix)
    minimum = np.asarray(new_matrix.min(0))  # column wise
    for i in range(minimum.shape[1]):
        if minimum[0, i] > 0:
            new_matrix[:, i] = np.log10(new_matrix[:, i])
        elif minimum[0, i] == 0:
            new_matrix[:, i] = np.log10(new_matrix[:, i] + 0.1)
        if new_matrix[:, i].std() > 0:
            new_matrix[:, i] = (new_matrix[:, i] - new_matrix[:, i].mean()) / new_matrix[:, i].std()
    return new_matrix