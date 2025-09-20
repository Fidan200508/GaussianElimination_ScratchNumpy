# ============================================
# Math4AI - Programming Assignment 2
# Systems of Linear Equations & Model Fitting
# ============================================

# --- Imports ---
import numpy as np

# --- Example System ---
A = [
    [2, 1, 3],
    [4, 4, 7],
    [2, 5, 9]
]

b = [1, 1, 3]

# =====================================================
# PART 2.1: Gaussian Elimination from Scratch
# =====================================================

def gaussian_elimination(A, b):
        n = len(A)
        # Augmented matrix
        M = [A[i] + [b[i]] for i in range(n)]

        for i in range(n):
            # Partial pivoting
            pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
            if abs(M[pivot_row][i]) < 1e-12:  # sıfıra yaxın pivot → keç
                continue
            # Swap rows
            M[i], M[pivot_row] = M[pivot_row], M[i]

            # Eliminate below
            for j in range(i + 1, n):
                if abs(M[i][i]) < 1e-12:
                    continue
                factor = M[j][i] / M[i][i]
                M[j] = [M[j][k] - factor * M[i][k] for k in range(n + 1)]

        # Check for no solution (0 = nonzero)
        for i in range(n):
            if all(abs(M[i][j]) < 1e-12 for j in range(n)) and abs(M[i][-1]) > 1e-12:
                return "No solution"

        # Check for infinite solutions (zero row = 0)
        rank = sum(any(abs(M[i][j]) > 1e-12 for j in range(n)) for i in range(n))
        if rank < n:
            return "Infinite solutions"

        # Back substitution (unique solution)
        x = [0] * n
        for i in reversed(range(n)):
            x[i] = (M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]

        return x

# --- Solve using your function ---
solution_scratch = gaussian_elimination(A, b)
print("Solution (from scratch):", solution_scratch)


# =====================================================
# PART 2.2: NumPy Verification
# =====================================================

# Convert to NumPy arrays
np_A = np.array(A, dtype=float)
np_b = np.array(b, dtype=float)

try:
    np_solution = np.linalg.solve(np_A, np_b)    #NumPy’s linear algebra module  .solve solves equation like Ax=b
    print("Solution (NumPy):", np_solution)
except np.linalg.LinAlgError as e:
    print("NumPy could not solve the system:", e)


# =====================================================
# Verification
# =====================================================
if isinstance(solution_scratch, list) or isinstance(solution_scratch, np.ndarray):
    print(np.isclose(solution_scratch,np_solution))
