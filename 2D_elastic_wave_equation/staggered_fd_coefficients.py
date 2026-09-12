import numpy as np

def staggered_fd_coefficients(order):
    '''
    Parameters
    order : int
        Accuracy order of the finite-difference scheme.
        Must be an even positive integer, e.g. 2, 4, 6, 8, 10.

    Returns
    coeff : ndarray
        Finite-difference coefficients
        [C1, C2, ..., CN]
        where N = order // 2
    '''
    # Check input
    if order <= 0 or order % 2 != 0:
        raise ValueError("order must be a positive even integer.")

    # Half stencil width
    N = order // 2

    # Coefficient matrix
    A = np.zeros((N, N), dtype=float)

    # Right-hand side
    b = np.zeros(N, dtype=float)
    b[0] = 1.0

    # Build matrix A
    for q in range(1, N + 1):
        for m in range(1, N + 1):
            A[q-1, m-1] = (2*m-1) ** (2*q-1)

    # Solve for C1, C2, ..., CN
    coeff = np.linalg.solve(A, b)

    return coeff