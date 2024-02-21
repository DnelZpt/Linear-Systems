"""
This program implements three different algorithms to solve
over-determined linear systems

Author: Daniel Zapata
April 2023
"""
import numpy as np


def moure_penrose(A_matrix, b_vector):
    """
    Function to solve a linear system using the Moore-Penrose pseudo inverse
    :param A_matrix: mxn array with linear system parameters
    :param b_vector: mx1 vector with b values
    :return: nx1 vector with 'system solution'
    """
    A_pseudo = np.linalg.inv(A_matrix.T @ A_matrix) @ A_matrix.T
    x = np.dot(A_pseudo, b_vector)
    return x


def solver_full_rank(A_matrix, b_vector):
    """
    Estimates x vector using SVD for over-determined linear systems
    and full rank A arrays
    :param A_matrix: mxn array with linear system parameters
    :param b_vector: mx1 vector with b values
    :return: nx1 vector with 'system solution'
    """
    rank = np.linalg.matrix_rank(A_matrix)  # Get A rank to determinate if system is full rank
    if rank < A_matrix.shape[1]:  # If system is not full rank, raise a value error
        raise ValueError('Matrix A must be full range..')
    # Find SVD for A_matrix
    u, d, v_t = np.linalg.svd(A_matrix)  # v is already transposed
    b_prime = u.T @ b_vector  # Calculates b'
    n = A_matrix.shape[1]  # Number of unknowns
    y = np.zeros((n, 1))  # Initialize 'y' as a nx1 zeros vector
    for i in range(n):
        # To avoid division by zero, pass to the next iteration if d_i value
        # is zero. So, yi value will be zero too.
        if d[i] == 0:
            continue

        y[i, 0] = b_prime[i, 0] / d[i]

    x = v_t.T @ y  # Transpose v_t to get V

    return x


def solver_deficient_rank(A_matrix, b_vector):
    """
    Estimates x vector using SVD for over-determined linear systems
    and deficient rank A arrays
    :param A_matrix: mxn array with linear system parameters
    :param b_vector: mx1 vector with b values
    :return: nx1 vector with 'system solution'
    """
    rank = np.linalg.matrix_rank(A_matrix)  # Get A rank
    if rank == A_matrix.shape[1]:  # If system is full rank, raise a value error
        raise ValueError('Matrix A must be deficient range.')
    # Find SVD for A_matrix
    u, d, v_t = np.linalg.svd(A_matrix)  # v is transposed
    b_prime = u.T @ b_vector
    y = np.zeros((A_matrix.shape[1], 1))
    for i in range(rank):
        # To avoid division by zero, pass to the next iteration if d_i value
        # is zero. So, yi value will be zero too.
        if d[i] == 0:
            continue
        y[i, 0] = b_prime[i, 0] / d[i]

    x = v_t.T @ y  # Transpose v_t to get V

    return x


def solver_homogeneous(A_matrix):
    """
    Takes a homogenous system, with a full rank params matrix, and calculates
    its solution using the last V column of SVD
    :param A_matrix: mxn full rank matrix
    :return: nx1 solutions vector
    """
    rank = np.linalg.matrix_rank(A_matrix)
    if rank < A_matrix.shape[1]:
        raise ValueError('Matrix A must be full range.')

    _, _, v_t = np.linalg.svd(A_matrix)  # Calculates only V vector (remember it's transposed)
    v = v_t.T  # Transpose 'v_t' to get the original vector V
    x = v[:, -1].reshape((A_matrix.shape[1], 1))  # Gets only the last V column

    return x


if __name__ == '__main__':
    # -------------------------------------------------------------------------------
    # Example 1: Full Rank
    # y = a1x^2 + a2x + a3
    # -------------------------------------------------------------------------------
    params_1 = np.array([[4, -4, 1], [1, -1, 1], [0, 0, 1], [1, 1, 1], [4, 2, 1]])
    b_values_1 = np.array([[5], [3], [0.5], [3.2], [6.1]])
    solution_1 = solver_full_rank(params_1, b_values_1)
    print("x values Example 1:")
    print("\ta1 = {:.4f}\n\ta2 = {:.4f}\n\ta3 = {:.4f}".format(solution_1[0][0], solution_1[1][0], solution_1[2][0]))
    print("\n-----------------------------------------------\n")

    # -------------------------------------------------------------------------------
    # Example 2: Deficient Rank
    # Let's take the past example and make a linear combination for some of its arrows:
    #   * Arrow 1 is arrow 4 + arrow 5
    #   * Arrow 2 is arrow 4 by 2
    #   * Arrow 3 is arrow 5 by 3
    # -------------------------------------------------------------------------------
    A_params_2 = np.array([[5, 3, 2], [2, 2, 2], [12, 6, 3], [1, 1, 1], [4, 2, 1]])
    b_values_2 = np.array([[9.3], [6.4], [18.3], [3.2], [6.1]])
    solution_2 = solver_deficient_rank(A_params_2, b_values_2)
    print("x values Example 2:")
    print("\ta1 = {:.4f}\n\ta2 = {:.4f}\n\ta3 = {:.4f}".format(solution_2[0][0], solution_2[1][0], solution_2[2][0]))
    print("\n-----------------------------------------------\n")

    # -------------------------------------------------------------------------------
    # Example 3: Homogeneous System
    # Taking the system:
    #                    x + y + 2z  = 0
    #                   3x - y - 2z  = 0
    #                   -x + 2y + z = 0
    # -------------------------------------------------------------------------------
    A_params_3 = np.array([[1, 1, 2], [3, -1, -2], [-1, 2, 1]])
    solution_3 = solver_homogeneous(A_params_3)
    print("x values Example 3:")
    print("\tx = {:.4f}\n\ty = {:.4f}\n\tz = {:.4f}".format(solution_2[0][0], solution_2[1][0], solution_2[2][0]))
    print("\n-----------------------------------------------\n")
