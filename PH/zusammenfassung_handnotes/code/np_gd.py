import numpy as np
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)  # 1. Spalte: Bias
y = np.array([1, 2, 3, 4], dtype=float)
theta, alpha = np.zeros(2), 0.1
for _ in range(1000):                       # Batch-Gradient-Descent
    grad = X.T @ (X @ theta - y) / len(y)   # (1/n) X^T (X theta - y)
    theta -= alpha * grad
print("theta GD    :", theta.round(3))
print("theta Normal:", np.linalg.solve(X.T @ X, X.T @ y).round(3))
